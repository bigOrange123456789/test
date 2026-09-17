"""远程模型 HTTPS 请求：有限重试、显式代理配置和脱敏连接日志。"""

from __future__ import annotations

import errno
import http.client
import json
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request


def proxy_summary(mode: str) -> str:
    """只输出代理的主机和端口，避免泄露代理用户名或密码。"""
    if mode == "direct":
        return "直连（本请求不使用系统代理）"
    proxies = urllib.request.getproxies()
    value = proxies.get("https") or proxies.get("http")
    if not value:
        return "系统代理模式，未检测到 HTTP(S) 代理"
    try:
        parsed = urllib.parse.urlsplit(value if "://" in value else f"http://{value}")
        return f"系统代理 {parsed.hostname}:{parsed.port or 80}"
    except ValueError:
        return "系统代理已配置（地址格式无法解析）"


def transient_error(error: Exception) -> bool:
    """仅重试暂时断连、超时或服务繁忙；证书及鉴权错误直接报告。"""
    if isinstance(error, urllib.error.HTTPError):
        return error.code in {429, 500, 502, 503, 504}
    reason = error.reason if isinstance(error, urllib.error.URLError) else error
    if isinstance(reason, ssl.SSLCertVerificationError):
        return False
    if isinstance(reason, (ConnectionResetError, ConnectionAbortedError, TimeoutError,
                           http.client.RemoteDisconnected, ssl.SSLEOFError)):
        return True
    if isinstance(reason, socket.gaierror):
        return reason.errno == socket.EAI_AGAIN
    codes = {errno.ECONNRESET, errno.ECONNABORTED, errno.ETIMEDOUT, 10054, 10053, 10060}
    return isinstance(reason, OSError) and (reason.errno in codes or getattr(reason, "winerror", None) in codes)


def post_json(endpoint: str, body: dict, api_key: str, cfg: dict, timeout: float, log, on_retry=None) -> str:
    """每次重试重新建连，保持完整图文请求不变；默认最多请求三次。"""
    mode = cfg.get("api_proxy_mode", "system")
    if mode not in {"system", "direct"}:
        raise ValueError("api_proxy_mode 只能为 system 或 direct。")
    attempts = cfg.get("api_max_attempts", 3)
    if type(attempts) is not int or not 1 <= attempts <= 5:
        raise ValueError("api_max_attempts 必须为 1 到 5 的整数。")
    timeout = float(cfg.get("api_timeout", timeout))
    if not 1 <= timeout <= 600:
        raise ValueError("api_timeout 必须在 1 到 600 秒之间。")
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    image_count = sum(part.get("type") == "image_url" for message in body.get("messages", [])
                      if isinstance(message.get("content"), list) for part in message["content"])
    network = proxy_summary(mode)
    log(f"远程请求 端点={endpoint} 请求字节={len(encoded)} 图片数={image_count} "
        f"超时={timeout:g}秒 最大尝试={attempts} 网络={network} TLS证书校验=开启")
    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        log(f"远程连接尝试 {attempt}/{attempts}")
        request = urllib.request.Request(endpoint, data=encoded, headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
            "Accept": "application/json", "Connection": "close",
        }, method="POST")
        proxy_handler = urllib.request.ProxyHandler({}) if mode == "direct" else urllib.request.ProxyHandler()
        opener = urllib.request.build_opener(proxy_handler)
        try:
            with opener.open(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
                log(f"远程响应成功 HTTP={response.status} 耗时={time.perf_counter() - started:.2f}秒 响应字符={len(raw)}")
                return raw
        except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
            if isinstance(error, urllib.error.HTTPError):
                try:
                    detail = error.read().decode("utf-8", errors="replace")[:600]
                finally:
                    error.close()
                message = f"HTTP {error.code} {error.reason}；{detail}"
            else:
                reason = error.reason if isinstance(error, urllib.error.URLError) else error
                message = f"{type(reason).__name__}: {reason}"
            log(f"远程连接失败 尝试={attempt}/{attempts} 耗时={time.perf_counter() - started:.2f}秒 {message}")
            if not transient_error(error) or attempt == attempts:
                hint = ""
                if not isinstance(error, urllib.error.HTTPError):
                    hint = ("；请检查网络和代理。可在当前模型配置中设置 api_proxy_mode 为 direct 测试直连，"
                            "或设为 system 使用系统代理。")
                raise RuntimeError(f"远程模型 {body.get('model')} 请求失败（{attempt} 次尝试，{network}）：{message}{hint}") from error
            delay = min(2 ** (attempt - 1), 4)
            if on_retry:
                on_retry(f"远程连接暂时失败，{delay} 秒后重试（{attempt + 1}/{attempts}）。",
                         {"attempt": attempt + 1, "max_attempts": attempts, "proxy_mode": mode})
            time.sleep(delay)
    raise RuntimeError("远程请求未返回结果。")

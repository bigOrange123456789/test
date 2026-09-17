"""对运行中的服务做真实图文联调；加 --remote 才额外调用配置的远程 API。"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inferenceValid.rag_mira import MiraRAG, reference_image_url
from inferenceValid.embed_mira_chroma import parse_image_paths, image_path


def analyze(url, payload):
    request = urllib.request.Request(url.rstrip("/") + "/api/analyze", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"}, method="POST")
    result = None
    started = time.perf_counter()
    # 本地服务始终直连，避免测试客户端继承系统代理。
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=300) as response:
        for line in response:
            text = line.decode("utf-8").strip()
            if not text.startswith("data:") or text == "data: [DONE]":
                continue
            event = json.loads(text[5:].strip())
            if event.get("type") == "error":
                raise RuntimeError(event["message"])
            if event.get("type") in {"progress", "heartbeat"}:
                print(event.get("stage", "心跳"), event.get("state", ""), event.get("message", ""), event.get("elapsed_seconds"), flush=True)
            if all(key in event for key in ("diagnosis", "findings", "analysis", "advice")):
                result = event
    if not result or not all(isinstance(value, str) and value for value in result.values()):
        raise AssertionError("服务未返回四个非空诊疗字段。")
    print("验证成功", payload["model"], "RAG=", payload["rag"]["enabled"], "图片=", bool(payload["image"]),
          "字段长度=", {k: len(v) for k, v in result.items()}, "耗时=", round(time.perf_counter() - started, 2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8771", help="已启动的本地服务地址。")
    parser.add_argument("--remote", action="store_true", help="额外执行一次远程通义视觉 + RAG 请求，可能产生 API 费用。")
    args = parser.parse_args()
    service = MiraRAG()
    service._open_index()
    source = service.collection.get(limit=1, include=["metadatas"])
    path = image_path(service.data_root, parse_image_paths(source["metadatas"][0]["image_paths"])[0])
    data_url, size = reference_image_url(path)
    image = {"name": path.name, "dataUrl": data_url, "type": data_url[5:].split(";", 1)[0], "size": size}
    inputs = {"caseInput": "软件联调用虚拟病例，无真实患者信息。当前材料不足，不要确诊疾病。上传图片仅用于测试，不属于患者。",
              "symptoms": "未提供", "exams": "未提供实际检查结果", "diagnosisReport": "请简短返回四个诊疗字段，注明信息不足。"}
    for uploaded, enabled in ((None, False), (image, False), (image, True)):
        analyze(args.url, {"model": "qwen3-vl-local", "inputs": inputs, "image": uploaded, "rag": {"enabled": enabled, "k": 3}})
    if args.remote:
        analyze(args.url, {"model": "remote-api-2", "inputs": inputs, "image": image, "rag": {"enabled": True, "k": 3}})


if __name__ == "__main__":
    main()

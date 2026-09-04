import argparse
import json
import os
import re
import sys
from collections import defaultdict
from threading import Thread
DEFAULT_SYSTEM_PROMPT = "你是一名谨慎的中文医疗问答助手。请基于用户问题给出准确、清晰、简洁的医学科普回答。"#"请用简洁、简短的语言回答用户的问题"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def configure_stdout():
  if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def mask_api_key(api_key):
  if not api_key:
    return "<empty>"
  if len(api_key) <= 8:
    return "<set>"
  return api_key[:4] + "..." + api_key[-4:]


def load_config(path=None):
  if path is None:
    path = os.path.join(SCRIPT_DIR, "config.json")
  with open(path, "r", encoding="utf-8") as file:
    return json.load(file)


def parse_config_bool(value, default=False):
  if value is None:
    return default
  if isinstance(value, bool):
    return value
  if isinstance(value, str):
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
      return True
    if normalized in ("0", "false", "no", "n", "off"):
      return False
  return bool(value)


def is_local_config(_model_id, cfg):
  return parse_config_bool(cfg.get("localhost"), default=False)


def resolve_existing_path(path):
  if not path or os.path.isabs(path):
    return path

  candidates = [
    os.path.abspath(path),
    os.path.abspath(os.path.join(SCRIPT_DIR, path)),
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", path)),
  ]
  for candidate in candidates:
    if os.path.exists(candidate):
      return candidate
  return candidates[0]


def print_config_summary(config):
  print("\n========== 配置摘要 ==========")
  for model_id, cfg in config.items():
    if is_local_config(model_id, cfg):
      print(
        f"[{model_id}] localhost=True "
        f"model_path={cfg.get('model_path', '')} "
        f"lora_adapter={cfg.get('lora_adapter', '')}"
      )
      continue

    print(
      f"[{model_id}] localhost=False "
      f"base_url={cfg.get('base_url', '')} "
      f"configured_model={cfg.get('model', '')} "
      f"api_key={mask_api_key(str(cfg.get('api_key', '')))}"
    )
  print("==============================\n")


def normalize_model_ids(models_response):
  model_ids = []
  for model in getattr(models_response, "data", []):
    model_id = getattr(model, "id", None)
    if model_id:
      model_ids.append(model_id)
  return sorted(set(model_ids))


def short_error_message(error):
  message = str(error).replace("\n", " ").strip()
  message = re.sub(r"\s+", " ", message)
  return message[:300] + ("..." if len(message) > 300 else "")


def print_available_remote_models(config):
  try:
    import openai # pip install openai
  except ImportError:
    print("当前环境未安装 openai，无法检查远程 API 模型列表。")
    return

  print("========== 远程 API 可用模型检查 ==========")

  grouped = defaultdict(list)
  for model_id, cfg in config.items():
    if is_local_config(model_id, cfg):
      continue
    api_key = cfg.get("api_key")
    base_url = cfg.get("base_url")
    if not api_key or not base_url:
      print(f"[{model_id}] 跳过：缺少 api_key 或 base_url")
      continue
    grouped[(base_url, api_key)].append(model_id)

  if not grouped:
    print("没有发现远程 API 配置。")
    print("==========================================\n")
    return

  for (base_url, api_key), model_ids in grouped.items():
    title = ", ".join(model_ids)
    print(f"\n[{title}]")
    print(f"base_url: {base_url}")
    print(f"api_key: {mask_api_key(str(api_key))}")

    try:
      client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=15)
      available_models = normalize_model_ids(client.models.list())
    except Exception as error:
      print("状态：查询失败，api_key 可能失效，或 base_url 不支持 /models。")
      print(f"错误：{short_error_message(error)}")
      continue

    if not available_models:
      print("状态：查询成功，但没有返回模型列表。")
      continue

    unavailable_model_ids = []

    for model_id in model_ids:
      configured_model = config[model_id].get("model", "")
      if configured_model in available_models:
        print(f"当前配置 [{model_id}].model = {configured_model}：可用")
      else:
        unavailable_model_ids.append(model_id)
        print(f"当前配置 [{model_id}].model = {configured_model}：未在可用模型列表中，请检查是否已下线或名称写错")

    if unavailable_model_ids:
      print(f"可用模型数量：{len(available_models)}")
      print("可用模型：")
      for available_model in available_models:
        print(f"  - {available_model}")

  print("\n==========================================\n")


def load_local_models(config):
  local_configs = [(model_id, cfg) for model_id, cfg in config.items() if is_local_config(model_id, cfg)]
  if not local_configs:
    return

  from transformers import AutoModelForCausalLM, AutoTokenizer # pip install transformers torch accelerate

  try:
    from peft import PeftModel # pip install peft
  except Exception:
    PeftModel = None

  for model_id, cfg in local_configs:
    model_path = resolve_existing_path(cfg.get("model_path"))
    if not model_path or not os.path.exists(model_path):
      cfg["load_error"] = f"本地模型路径不存在：{cfg.get('model_path')}"
      print(f"[{model_id}] {cfg['load_error']}")
      continue

    try:
      model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",          # 自动选择最佳数据类型（如fp16）
            device_map="cpu", #"cuda:0", #"auto",            # 自动分配到可用设备（GPU优先）
      )
      tokenizer = AutoTokenizer.from_pretrained(model_path)

      lora_adapter = cfg.get("lora_adapter")
      if lora_adapter:
        lora_adapter = resolve_existing_path(lora_adapter)
        if not os.path.exists(lora_adapter):
          raise FileNotFoundError(f"LoRA adapter 路径不存在：{cfg.get('lora_adapter')}")
        if PeftModel is None:
          raise ImportError("当前环境未安装 peft，无法加载 LoRA adapter")
        model = PeftModel.from_pretrained(model, lora_adapter)

      cfg["model"] = model
      cfg["tokenizer"] = tokenizer
      cfg["resolved_model_path"] = model_path
      if cfg.get("lora_adapter"):
        cfg["resolved_lora_adapter"] = lora_adapter
      print(f"[{model_id}] 本地模型加载完成：{model_path}")
    except Exception as error:
      cfg["load_error"] = short_error_message(error)
      print(f"[{model_id}] 本地模型加载失败：{cfg['load_error']}")


def parse_args():
  parser = argparse.ArgumentParser(description="多目标大模型对话脚本，支持本地模型和远程 API。")
  stream_group = parser.add_mutually_exclusive_group()
  stream_group.add_argument("--stream", action="store_true", dest="stream", help="开启流式输出。")
  stream_group.add_argument("--no-stream", action="store_false", dest="stream", help="关闭流式输出。")
  parser.add_argument("--max-new-tokens", type=int, default=1000, help="模型单次回答最多生成的 token 数。")
  parser.add_argument("--temperature", type=float, default=0.001, help="生成温度，数值越高随机性越强。")
  parser.add_argument("--top-p", type=float, default=0.95, help="本地模型 nucleus sampling 的 top_p 参数。")
  parser.add_argument("--repetition-penalty", type=float, default=1.1, help="本地模型重复惩罚系数。")
  parser.add_argument("--check-remote-models", action="store_true", help="启动时检查远程 API 当前配置模型是否可用。")
  parser.set_defaults(stream=True)
  return parser.parse_args()


configure_stdout()
args = parse_args()
config = load_config()
# print_config_summary(config)
if args.check_remote_models:
  print_available_remote_models(config)

param={
  "max_new_tokens":args.max_new_tokens,#256,#512      # 最大生成长度
  "temperature":args.temperature,#0.1,#0.7,#0.6           # 控制随机性（0=确定性，越高越随机）
  "top_p":args.top_p,
  "repetition_penalty":args.repetition_penalty,
  "stream" : args.stream,
  # "enable_thinking":False, 
}

load_local_models(config)
modelId=""
for modelId in config:
  config[modelId]["messages"]=[{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
  # config[modelId]["messages"]=[{"role": "system", "content": "请专业且详细的回答用户的问题"}]
def chat_with_model(modelId, question):
  if not question:
    return "Please enter a question~"
  try:
    cfg = config[modelId]
    cfg["messages"].append({"role": "user", "content": question})
    if is_local_config(modelId, cfg):
      if "load_error" in cfg:
        return f"本地模型未加载成功：{cfg['load_error']}"
      if "model" not in cfg or "tokenizer" not in cfg:
        return "本地模型未加载成功：缺少 model 或 tokenizer"

      model=cfg["model"]
      tokenizer=cfg["tokenizer"]
      # 应用聊天模板生成模型输入
      text = tokenizer.apply_chat_template(
          cfg["messages"],#history,
          tokenize=False,
          add_generation_prompt=True   # 为模型回复添加生成提示
      )
      # 分词并转移到模型所在设备
      inputs = tokenizer(text, return_tensors="pt").to(model.device)
      generation_kwargs = {
          **inputs,
          "max_new_tokens":param["max_new_tokens"],#256,#512      # 最大生成长度
          "temperature":param["temperature"],#0.1,#0.6           # 控制随机性（0=确定性，越高越随机）
          "top_p":param["top_p"],                  # 核采样阈值
          "do_sample":param["temperature"] > 0,              # 启用采样（否则为贪心解码）
          "repetition_penalty":param["repetition_penalty"],      # 重复惩罚
          "pad_token_id":tokenizer.eos_token_id #填充标记（
      }
      if param["stream"]:
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs["streamer"] = streamer
        generation_thread = Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
        generation_thread.start()
        response_parts = []
        print(" "+modelId+":", end="", flush=True)
        for content in streamer:
          response_parts.append(content)
          print(content, end="", flush=True)
        generation_thread.join()
        print()
        response = "".join(response_parts)
      else:
        # 生成回复
        outputs = model.generate(**generation_kwargs)
        # 解码生成部分（仅保留新生成的token）
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
    else:
      import openai # pip install openai

      client = openai.OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"]
      )
      message = client.chat.completions.create(
        model=cfg["model"],
        messages=cfg["messages"],
        extra_body=cfg.get("extra_body", {}),
        temperature=param["temperature"],
        max_tokens=param["max_new_tokens"], #1000
        stream=param["stream"],  # 开启流式
        # enable_thinking=param["enable_thinking"],
      )
      if param["stream"]:
        response = ""
        print(" "+modelId+":", end="", flush=True)
        for chunk in message:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                response += content
                print(content, end="", flush=True) 
        print()
      else:
        response = message.choices[0].message.content
      import re
      response = re.sub(r'.*?</think>\s*', '', response, flags=re.DOTALL) # 删除Chain of Thought
    cfg["messages"].append({"role": "assistant", "content": response})
    return response
  except Exception as e:
    return f"Failed to call model:{e}" 
def initUI():# Gradio UI
  import gradio as gr # pip install gradio
  with gr.Blocks() as demo:
    gr.Markdown("## 💬 多模型 LLM 聊天界面")
    model_selector = gr.Dropdown(choices=config.keys(), value=modelId, label="选择模型")
    with gr.Row():
      input_box = gr.Textbox(label="你的问题", placeholder="请输入你想问的问题", lines=1)
      send_button = gr.Button("发送")
    output_box = gr.Textbox(label="模型回答", interactive=False)
    send_button.click(chat_with_model, inputs=[model_selector, input_box], outputs=output_box)
  demo.launch(debug=True)
def initUI2():
  from flask import Flask, send_from_directory # pip install flask
  from flask_socketio import SocketIO, emit # pip install flask-socketio
  app = Flask(__name__)
  socketio = SocketIO(app)
  # @app.route("/")
  # def index():
  #   return send_from_directory(".", "inferenceValid/test.html") # return send_from_directory(".", "test.html")
  @app.route("/")
  def index():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(script_dir, "test.html")
  @socketio.on("chatMessage")
  def handle_chat_message(data):
    model_id = data["modelId"]
    question = data["question"]
    try:
      response = chat_with_model(model_id, question)
    except Exception as e:
      response = str(e)
    emit("chatResponse", {"response": response})
  socketio.run(app, host="0.0.0.0", port=3000)
while True:
  prompt = input("   our:")
  if len(prompt.split("switch-"))>1:
    str0 = prompt.split("switch-")[1]
    if str0 in config:
      modelId=str0
      print("已将modelId切换为:",modelId)
    elif str0=="gradio":
      print("[gradio]正在生成交互页面的链接...")
      initUI()
    elif str0=="html":
      print("[html]正在生成交互页面的链接...")
      initUI2()
    else:
      print("无法识别的切换目标:",str0)
      exit(0)
    continue
  response = chat_with_model(modelId,prompt)
  print("response",response)
  if not param["stream"]:
    print(" "+modelId+":", response)

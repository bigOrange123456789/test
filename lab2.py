import openai # pip install openai

import json
import os
import re
from collections import defaultdict


def mask_api_key(api_key):
  if not api_key:
    return "<empty>"
  if len(api_key) <= 8:
    return "<set>"
  return api_key[:4] + "..." + api_key[-4:]


def load_config(path="config.json"):
  with open(path, "r", encoding="utf-8") as file:
    return json.load(file)


def print_config_summary(config):
  print("\n========== 配置摘要 ==========")
  for model_id, cfg in config.items():
    if model_id == "localhost":
      print(f"[{model_id}] local model_path={cfg.get('model_path', '')}")
      continue

    print(
      f"[{model_id}] base_url={cfg.get('base_url', '')} "
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
  print("========== 远程 API 可用模型检查 ==========")

  grouped = defaultdict(list)
  for model_id, cfg in config.items():
    if model_id == "localhost":
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


config = load_config()
# print_config_summary(config)
if  False:
  print_available_remote_models(config)

param={
  "max_new_tokens":1000,#256,#512      # 最大生成长度
  "temperature":0.001,#0.1,#0.7,#0.6           # 控制随机性（0=确定性，越高越随机）
  "stream" : False 
}

if os.path.exists(config["localhost"]["model_path"]):
  from transformers import AutoModelForCausalLM, AutoTokenizer # pip install transformers torch accelerate
  config["localhost"]["model"] = AutoModelForCausalLM.from_pretrained(
        config["localhost"]["model_path"],
        torch_dtype="auto",          # 自动选择最佳数据类型（如fp16）
        device_map="cpu", #"cuda:0", #"auto",            # 自动分配到可用设备（GPU优先）
  )
  config["localhost"]["tokenizer"] = AutoTokenizer.from_pretrained(config["localhost"]["model_path"])
modelId=""
for modelId in config:
  config[modelId]["messages"]=[{"role": "system", "content": "请用简洁、简短的语言回答用户的问题"}]
  # config[modelId]["messages"]=[{"role": "system", "content": "请专业且详细的回答用户的问题"}]
def chat_with_model(modelId, question):
  if not question:
    return "Please enter a question~"
  try:
    config[modelId]["messages"].append({"role": "user", "content": question})
    if modelId =="localhost":
      model=config[modelId]["model"]
      tokenizer=config[modelId]["tokenizer"]
      # 应用聊天模板生成模型输入
      text = tokenizer.apply_chat_template(
          config[modelId]["messages"],#history,
          tokenize=False,
          add_generation_prompt=True   # 为模型回复添加生成提示
      )
      # 分词并转移到模型所在设备
      inputs = tokenizer(text, return_tensors="pt").to(model.device)
      # 生成回复
      outputs = model.generate(
          **inputs,
          max_new_tokens=256,#param["max_new_tokens"],#256,#512      # 最大生成长度
          temperature=param["temperature"],#0.1,#0.6           # 控制随机性（0=确定性，越高越随机）
          top_p=0.95,                  # 核采样阈值
          do_sample=True,              # 启用采样（否则为贪心解码）
          repetition_penalty=1.1,      # 重复惩罚
          pad_token_id=tokenizer.eos_token_id #填充标记（
      )
      # 解码生成部分（仅保留新生成的token）
      response = tokenizer.decode(
          outputs[0][inputs.input_ids.shape[1]:],
          skip_special_tokens=True
      )
    else:
      client = openai.OpenAI(
        api_key=config[modelId]["api_key"],
        base_url=config[modelId]["base_url"]
      )
      message = client.chat.completions.create(
        model=config[modelId]["model"],
        messages=config[modelId]["messages"],
        extra_body=config[modelId]["extra_body"],
        temperature=param["temperature"],
        max_tokens=param["max_new_tokens"], #1000
        stream=param["stream"]  # 开启流式
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
    config[modelId]["messages"].append({"role": "assistant", "content": response})
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
  @app.route("/")
  def index():
    return send_from_directory(".", "test.html")
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
  if not param["stream"]:
    print(" "+modelId+":", response)

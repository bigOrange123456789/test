import openai # pip install openai
config={
  "localhost":{
    "model_path":"DeepSeek-Model",
  }
}
param={
  "max_new_tokens":1000,#256,#512      # 最大生成长度
  "temperature":0.001,#0.1,#0.7,#0.6           # 控制随机性（0=确定性，越高越随机）
  "stream" : False 
}
import os
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
    else:
      print("无法识别的切换目标:",str0)
      exit(0)
    continue
  response = chat_with_model(modelId,prompt)
  if not param["stream"]:
    print(" "+modelId+":", response)
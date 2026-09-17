// 验证浏览器流解析器：中文跨字节分块、CRLF、JSON 回退和真实错误事件。
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const sandbox = {
  console: { info() {}, error() {} }, TextDecoder, TextEncoder,
  location: { hash: "#diagnosis" },
  document: { addEventListener() {}, getElementById() { return {}; }, querySelector() { return null; } },
  window: { addEventListener() {} }, requestAnimationFrame() {},
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(__dirname, "../../cardio_ai_platform/app.js"), "utf8"), sandbox);

function responseFrom(text, contentType = "text/event-stream", size = 1) {
  const bytes = new TextEncoder().encode(text);
  return new Response(new ReadableStream({
    start(controller) {
      for (let i = 0; i < bytes.length; i += size) controller.enqueue(bytes.slice(i, i + size));
      controller.close();
    },
  }), { headers: { "Content-Type": contentType } });
}

test("SSE 中文逐字节分块仍能即时回调进度并合并字段", async () => {
  const events = [];
  const report = { diagnosis: "诊断", findings: "发现", analysis: "分析", advice: "建议" };
  const frames = [
    { status: "started" },
    { type: "progress", stage: "embedding", state: "running", message: "编码中" },
    { type: "heartbeat", elapsed_seconds: 10 },
    ...Object.entries(report).map(([key, value]) => ({ delta: { [key]: value } })),
    report,
  ];
  const body = frames.map((frame) => `data: ${JSON.stringify(frame)}\r\n\r\n`).join("") + "data: [DONE]\r\n\r\n";
  const actual = await sandbox.readAnalysisResponse(responseFrom(body), (event) => events.push(event));
  assert.deepEqual(JSON.parse(JSON.stringify(actual)), report);
  assert.equal(events.length, 2);
  assert.equal(events[0].message, "编码中");
});

test("完整 JSON 和无末尾空行的 SSE 兼容", async () => {
  const report = { diagnosis: "结果", findings: "发现", analysis: "分析", advice: "建议" };
  for (const [body, type] of [[JSON.stringify(report), "application/json"], [`data: ${JSON.stringify(report)}`, "text/plain"]]) {
    const actual = await sandbox.readAnalysisResponse(responseFrom(body, type));
    assert.equal(actual.diagnosis, "结果");
  }
});

test("失败事件直接抛出详细信息，不把失败当作成功诊断", async () => {
  await assert.rejects(sandbox.readAnalysisResponse(responseFrom('data: {"type":"error","message":"参考图片丢失"}\n\n')),
    /参考图片丢失/);
});

test("完整但损坏的事件报错；缺少诊疗字段的流报错", async () => {
  await assert.rejects(sandbox.readAnalysisResponse(responseFrom('data: {broken}\n\n')), /无法解析/);
  await assert.rejects(sandbox.readAnalysisResponse(responseFrom('data: {"status":"started"}\n\n')), /没有包含/);
});

test("进度在响应结束前到达", async () => {
  let controller;
  const response = new Response(new ReadableStream({ start(value) { controller = value; } }),
    { headers: { "Content-Type": "text/event-stream" } });
  let progress;
  const arrived = new Promise((resolve) => { progress = resolve; });
  const pending = sandbox.readAnalysisResponse(response, progress);
  controller.enqueue(new TextEncoder().encode('data: {"type":"progress","stage":"retrieval"}\n\n'));
  assert.equal((await arrived).stage, "retrieval");
  controller.enqueue(new TextEncoder().encode('data: {"diagnosis":"结果"}\n\n'));
  controller.close();
  assert.equal((await pending).diagnosis, "结果");
});

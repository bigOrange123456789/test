// 无浏览器/模型依赖的模板回归测试：node --test inferenceValid/tests/test_frontend_templates.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const source = fs.readFileSync(path.join(__dirname, "../../cardio_ai_platform/app.js"), "utf8");

function loadApp(template) {
  const app = { innerHTML: "" };
  const report = { innerHTML: "" };
  const timers = new Map();
  let nextTimer = 1;
  const document = {
    addEventListener() {},
    getElementById(id) { return id === "app" ? app : null; },
    // 刻意不提供中间流程面板；更新仍须安全、报告仍须可渲染。
    querySelector(selector) { return selector === "[data-structured-report]" ? report : null; },
  };
  if (template !== undefined) document.documentElement = { dataset: { frontendTemplate: template } };
  const sandbox = {
    console: { info() {}, error() {} },
    TextDecoder, TextEncoder,
    location: { hash: "#diagnosis" },
    history: { replaceState() {} },
    document,
    window: { addEventListener() {} },
    requestAnimationFrame() {},
    setInterval(callback) { const id = nextTimer++; timers.set(id, callback); return id; },
    clearInterval(id) { timers.delete(id); },
  };
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox, { filename: "cardio_ai_platform/app.js" });
  return {
    sandbox, app, report,
    state: vm.runInContext("state", sandbox),
    routes: vm.runInContext("Object.keys(pageMeta)", sandbox),
    drainTimers() {
      for (let iteration = 0; timers.size && iteration < 10000; iteration += 1) {
        for (const callback of Array.from(timers.values())) callback();
      }
      assert.equal(timers.size, 0, "动画计时器应当正常完成");
    },
  };
}

function layoutOf(html) {
  const match = /<section class="(diagnosis-layout[^\"]*)">([\s\S]*?)<\/section>/.exec(html);
  assert.ok(match, "病例分析布局应存在");
  return { classes: match[1].split(/\s+/), html: match[2] };
}

function assertLayout(html, template) {
  const layout = layoutOf(html);
  assert.equal(layout.classes.includes("diagnosis-layout-compact"), template === "compact");
  assert.equal((layout.html.match(/<div class="panel">/g) || []).length, template === "compact" ? 2 : 3);
  assert.match(layout.html, /左侧：病例输入/);
  assert.match(layout.html, /右侧：结构化诊疗结果/);
  assert.equal(layout.html.includes("中间：AI 医学分析过程"), template !== "compact");
  assert.equal((layout.html.match(/data-analysis-image-input/g) || []).length, 1);
  assert.equal((layout.html.match(/<h3>医学影像上传<\/h3>/g) || []).length, 1);
  const uploadIndex = layout.html.indexOf("<h3>医学影像上传</h3>");
  const inputsIndex = layout.html.indexOf('class="input-grid"');
  assert.ok(inputsIndex >= 0);
  assert.ok(uploadIndex > layout.html.indexOf("左侧：病例输入"));
  assert.ok(uploadIndex < layout.html.indexOf('data-analysis-model'));
  if (template === "compact") assert.ok(uploadIndex < inputsIndex, "紧凑模板影像上传应在年龄等输入之前");
  else assert.ok(uploadIndex > layout.html.indexOf('data-analysis-input="diagnosisReport"'), "原模板影像上传位置不变");
  for (const selector of ["data-analysis-model", "data-analysis-rag", "data-rag-k", 'data-action="start-analysis"', "data-structured-report"]) {
    assert.ok(layout.html.includes(selector), `${selector} 控件应保留`);
  }
}

test("缺少模板配置时保持 original 布局", () => {
  const defaultApp = loadApp();
  assert.equal(defaultApp.app.innerHTML, loadApp("original").app.innerHTML);
  assertLayout(defaultApp.app.innerHTML, "original");
});

for (const template of ["original", "compact"]) {
  test(`${template}：病例分析面板、唯一上传控件及顺序正确`, () => {
    const app = loadApp(template);
    assertLayout(app.app.innerHTML, template);
    app.state.ragEnabled = true;
    app.sandbox.render();
    assertLayout(app.app.innerHTML, template);
    assert.equal(app.app.innerHTML.includes("data-rag-progress"), template === "original");
  });

  test(`${template}：影像预览、清除与页面重绘保持布局`, () => {
    const app = loadApp(template);
    app.state.uploadedImage = { name: "test.png", size: 100, type: "image/png", dataUrl: "data:image/png;base64,dGVzdA==" };
    app.state.uploadState = "已上传 test.png";
    app.state.selectedAnalysisModel = "qwen3-vl-local";
    app.sandbox.render();
    assert.match(app.app.innerHTML, /class="upload-preview"/);
    assert.match(app.app.innerHTML, /data-action="clear-image"/);
    assertLayout(app.app.innerHTML, template);
    app.sandbox.clearAnalysisImage();
    assert.equal(app.state.uploadedImage, null);
    assert.doesNotMatch(app.app.innerHTML, /class="upload-preview"/);
    assertLayout(app.app.innerHTML, template);
    app.sandbox.setRoute("cases");
    app.sandbox.setRoute("diagnosis");
    assertLayout(app.app.innerHTML, template);
  });

  test(`${template}：中间面板不存在时 RAG 进度、心跳和结果输出仍可工作`, async () => {
    const app = loadApp(template);
    app.state.ragEnabled = true;
    app.state.analysisRunId = 7;
    const matches = [{ id: "mira:train:0:open_ended:0", rank: 1, similarity: 0.9, image_count: 0, images: [] }];
    for (const [index, stage] of ["input", "index", "embedding", "retrieval", "sources", "context", "generation"].entries()) {
      app.sandbox.handleAnalysisEvent({
        type: "progress", stage, state: "complete", message: `${stage} 完成`, elapsed_seconds: index + 1,
        details: stage === "sources" ? { matches } : {},
      }, 7);
    }
    app.sandbox.handleAnalysisEvent({ type: "heartbeat", elapsed_seconds: 8 }, 7);
    assert.equal(app.state.analysisProgressLog.length, 7);
    assert.equal(app.state.analysisElapsed, 8);
    assert.equal(app.state.ragMatches[0].id, matches[0].id);
    app.sandbox.handleAnalysisEvent({ type: "heartbeat", elapsed_seconds: 999 }, 6);
    assert.equal(app.state.analysisElapsed, 8, "过期请求不能污染当前状态");

    const result = { diagnosis: "诊断结果", findings: "关键发现", analysis: "分析 <script>test</script>", advice: "临床建议" };
    const completed = app.sandbox.typeAnalysisResult(result, 7);
    app.drainTimers();
    await completed;
    assert.equal(app.state.analysisStatus, "complete");
    assert.match(app.report.innerHTML, /诊断结果/);
    assert.match(app.report.innerHTML, /关键发现/);
    assert.match(app.report.innerHTML, /分析 &lt;script&gt;test&lt;\/script&gt;/);
    assert.doesNotMatch(app.report.innerHTML, /<script>/);
    assert.match(app.report.innerHTML, /临床建议/);
    assertLayout(app.app.innerHTML, template);
    assert.equal(app.app.innerHTML.includes("data-rag-progress"), template === "original");
  });
}

test("模板切换不改变病例分析以外的任何页面", () => {
  const original = loadApp("original");
  const compact = loadApp("compact");
  for (const route of original.routes.filter((name) => name !== "diagnosis")) {
    original.sandbox.setRoute(route);
    compact.sandbox.setRoute(route);
    assert.equal(compact.app.innerHTML, original.app.innerHTML, `${route} 页面不应因模板变化而变化`);
  }
});

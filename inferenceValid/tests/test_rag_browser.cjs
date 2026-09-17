// 通过真实浏览器检查 RAG 控件、上传载荷、结果显示及桌面/手机布局。
// 替代分析与图片接口，不向远程模型发送测试数据。
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const os = require("node:os");

(async () => {
  const output = path.join(os.tmpdir(), "cardio-rag-check");
  await fs.mkdir(output, { recursive: true });
  const browser = await chromium.launch({ channel: "msedge", headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1050 } });
    const errors = [];
    const requests = [];
    let fail = false;
    page.on("pageerror", (error) => errors.push(error.message));
    const imageBytes = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII=", "base64");
    await page.route("**/api/rag/image?**", (route) => route.request().url().includes("test-missing=1")
      ? route.fulfill({status: 404, contentType: "application/json", body: '{"error":"测试图片缺失"}'})
      : route.fulfill({status: 200, contentType: "image/png", body: imageBytes}));
    await page.route("**/api/analyze", async (route) => {
      const payload = route.request().postDataJSON();
      requests.push(payload);
      const frames = [ { status: "started" } ];
      const stages = ["input", "index", "embedding", "retrieval", "sources", "context", "generation"];
      stages.forEach((stage, i) => frames.push({ type: "progress", stage, state: "complete",
        message: ["已接收病例文本与图片", "向量库连接完成", "2048 维图文编码完成", `Top-${payload.rag.k} 检索完成`, "已读取完整问答与图片", "上下文组装完成", "模型生成完成"][i],
        elapsed_seconds: i + 1,
        details: stage === "sources" ? { matches: [1,2,3].slice(0, payload.rag.k).map((rank) => ({ rank, id: `mira:train:${rank}:open_ended:0`,
          similarity: 0.93 - rank * 0.05, image_count: rank === 1 ? 2 : 1, source_csv: "train.csv", source_row: rank,
          category: "open_ended", preview: "旧的截短预览",
          document: `Question: 参考问题 ${rank}\nAnswer: 完整回答 <script>不应执行</script>\n${"完整资料内容。".repeat(180)}末尾校验文字`,
          images: Array.from({length: rank === 1 ? 2 : 1}, (_, index) => ({name: `参考影像-${rank}-${index}.png`,
            url: `/api/rag/image?id=mira:train:${rank}:open_ended:0&index=${index}`})) })) } :
          stage === "index" ? {count: 65000, index_complete: false} : stage === "embedding" ? {dimension: 2048, norm: 1} : {},
      }));
      if (fail) frames.push({type: "error", message: "参考图片缺失：请检查 MIRA-data 路径；" + "very-long-path/".repeat(30)});
      else frames.push({diagnosis: "测试诊断", findings: "参考 [MIRA-1]", analysis: "已结合原始病例与检索参考完成分析。", advice: "需由临床医生结合实际检查复核。"});
      await route.fulfill({ status: 200, contentType: "text/event-stream; charset=utf-8",
        body: frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join("") + "data: [DONE]\n\n" });
    });
    await page.goto(process.env.RAG_PREVIEW_URL || "http://127.0.0.1:8771/#diagnosis");
    assert.equal(await page.locator("[data-analysis-rag]").isChecked(), false);
    assert.equal(await page.locator("[data-rag-k]").inputValue(), "3");
    await page.locator("[data-analysis-model]").selectOption("remote-api-2");
    await page.locator("[data-analysis-rag]").check();
    await page.locator('[data-analysis-input="caseInput"]').fill("浏览器测试病例");
    await page.locator("[data-analysis-image-input]").setInputFiles({name: "test.png", mimeType: "image/png",
      buffer: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII=", "base64")});
    await page.locator(".upload-preview img").waitFor();
    await page.locator("[data-rag-k]").fill("2");
    await page.locator('[data-action="start-analysis"]').click();
    await page.waitForFunction(() => state.analysisStatus === "complete");
    assert.equal(requests[0].rag.k, 2);
    assert.equal(requests[0].rag.enabled, true);
    assert.equal(requests[0].inputs.caseInput, "浏览器测试病例");
    assert.match(requests[0].image.dataUrl, /^data:image\/png;base64,/);
    assert.equal(await page.locator(".rag-match").count(), 2);
    assert.equal(await page.locator("[data-rag-progress] script").count(), 0);
    assert.match(await page.locator(".rag-document").first().innerText(), /末尾校验文字/);
    assert.equal(await page.locator(".rag-match").first().locator("[data-rag-image]").count(), 2);
    assert.equal(await page.locator("[data-rag-image]").count(), 3);
    for (const picture of await page.locator("[data-rag-image]").all()) {
      await picture.scrollIntoViewIfNeeded();
      await picture.evaluate((img) => img.decode());
    }
    const imageLink = await page.locator(".rag-image a").first().getAttribute("href");
    assert.match(imageLink, /&full=1$/);
    // 中间阶段更新不会替换资料 DOM，避免图片闪烁和阅读位置丢失。
    await page.evaluate(() => {
      const picture = document.querySelector("[data-rag-image]");
      picture.dataset.retained = "yes";
      handleAnalysisEvent({type: "progress", stage: "generation", state: "running", message: "继续生成", elapsed_seconds: 8}, state.analysisRunId);
    });
    assert.equal(await page.locator("[data-rag-image]").first().getAttribute("data-retained"), "yes");
    for (const [name, width, height] of [["desktop",1600,1050], ["mobile",390,844]]) {
      await page.setViewportSize({width,height});
      await page.evaluate(() => window.scrollTo(0, 0));
      const bounds = await page.evaluate(() => ({width: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth}));
      assert.ok(bounds.scroll <= bounds.width + 1, `${name} 出现横向溢出：${JSON.stringify(bounds)}`);
      await page.screenshot({path: path.join(output, `rag-${name}.png`), fullPage: true});
      await page.locator(".rag-controls").screenshot({path: path.join(output, `rag-controls-${name}.png`)});
      await page.locator("[data-rag-progress]").screenshot({path: path.join(output, `rag-progress-${name}.png`),
        style: ".topbar { visibility: hidden !important; }"});
    }
    // 单张图片失败仍保留其他图片、完整问答和分析结果。
    await page.locator("[data-rag-image]").first().evaluate((img) => { img.src += "&test-missing=1"; });
    await page.locator(".rag-image-error:not([hidden])").first().waitFor();
    assert.equal(await page.locator("[data-rag-image]").first().isHidden(), true);
    assert.match(await page.locator(".rag-document").first().innerText(), /末尾校验文字/);
    await page.locator("[data-analysis-model]").selectOption("qwen3-vl-local");
    assert.equal(await page.locator(".rag-model-note").count(), 0);
    assert.match(await page.locator(".upload-hint").innerText(), /当前视觉模型/);
    await page.locator('[data-action="start-analysis"]').click();
    await page.waitForFunction(() => state.analysisStatus === "complete");
    assert.equal(requests.at(-1).model, "qwen3-vl-local");
    assert.ok(requests.at(-1).image);
    await page.locator("[data-analysis-model]").locator("..").screenshot({path: path.join(output, "qwen-model-mobile.png")});
    await page.locator('[data-action="clear-image"]').click();
    await page.locator("[data-rag-k]").fill("3");
    await page.locator('[data-action="start-analysis"]').click();
    await page.waitForFunction(() => state.analysisStatus === "complete");
    assert.equal(requests.at(-1).image, null);
    assert.equal(requests.at(-1).rag.enabled, true);
    await page.locator("[data-analysis-rag]").uncheck();
    assert.equal(await page.locator("[data-rag-k]").isDisabled(), true);
    assert.equal(await page.locator(".rag-match").count(), 0);
    await page.locator('[data-action="start-analysis"]').click();
    await page.waitForFunction(() => state.analysisStatus === "complete");
    assert.equal(requests.at(-1).rag.enabled, false);
    await page.locator("[data-analysis-rag]").check();
    await page.locator("[data-rag-k]").fill("0");
    await page.locator('[data-action="start-analysis"]').click();
    assert.equal(requests.length, 4);
    await page.locator("[data-rag-k]").fill("3");
    fail = true;
    await page.locator('[data-action="start-analysis"]').click();
    await page.waitForFunction(() => state.analysisStatus === "error");
    assert.equal(await page.locator(".rag-match").count(), 3, "生成失败时仍应保留已检索的资料");
    assert.match(await page.locator("[data-structured-report]").innerText(), /参考图片缺失：请检查 MIRA-data 路径/);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1), "长错误消息导致横向溢出");
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({status:"通过", requests:requests.length, browserErrors:errors, screenshots:output}));
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });

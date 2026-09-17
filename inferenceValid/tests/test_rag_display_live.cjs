// 使用实际向量库、图片接口和本地 Qwen 验证检索资料展示，不调用远程 API。
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
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => { if (message.type() === "error") errors.push(message.text()); });
    await page.goto(process.env.RAG_PREVIEW_URL || "http://127.0.0.1:8771/#diagnosis");
    await page.locator("[data-analysis-model]").selectOption("qwen3-vl-local");
    await page.locator("[data-analysis-rag]").check();
    const inputs = {
      caseInput: "软件联调用虚拟病例，无真实患者信息。当前材料不足，不要确诊疾病。",
      symptoms: "未提供", exams: "未提供实际检查结果",
      diagnosisReport: "请简短返回四个诊疗字段，注明信息不足。检索资料仅用于验证软件。",
    };
    for (const [key, value] of Object.entries(inputs)) {
      await page.locator(`[data-analysis-input="${key}"]`).fill(value);
    }
    await page.locator('[data-action="start-analysis"]').click();
    await page.waitForFunction(() => state.ragMatches.length === 3 || state.analysisStatus === "error", null, { timeout: 300000 });
    const early = await page.evaluate(() => ({status: state.analysisStatus, count: state.ragMatches.length, error: state.analysisError}));
    assert.equal(early.count, 3, early.error);
    assert.equal(early.status, "running", "检索资料应在模型生成结束前出现");
    assert.equal(await page.locator(".rag-match").count(), 3);
    console.log(JSON.stringify({阶段: "检索完成，生成仍在进行", 资料组数: early.count}));
    await page.waitForFunction(() => ["complete", "error"].includes(state.analysisStatus), null, { timeout: 300000 });
    const result = await page.evaluate(() => ({status: state.analysisStatus, matches: state.ragMatches, error: state.analysisError}));
    assert.equal(result.status, "complete", result.error);
    let imageCount = 0;
    for (let index = 0; index < result.matches.length; index++) {
      const match = result.matches[index];
      const group = page.locator(".rag-match").nth(index);
      assert.equal(await group.locator(".rag-document").textContent(), match.document);
      assert.equal(await group.locator("[data-rag-image]").count(), match.image_count);
      for (const picture of await group.locator("[data-rag-image]").all()) {
        await picture.scrollIntoViewIfNeeded();
        const size = await picture.evaluate(async (img) => { await img.decode(); return [img.naturalWidth, img.naturalHeight]; });
        assert.ok(size.every((value) => value > 0 && value <= 960), `预览尺寸无效：${size}`);
        imageCount++;
      }
    }
    const originalURL = await page.locator(".rag-image a").first().getAttribute("href");
    const original = await page.request.get(new URL(originalURL, page.url()).href);
    assert.equal(original.status(), 200);
    assert.match(original.headers()["content-type"], /^image\//);
    for (const [name, width, height] of [["desktop",1600,1050], ["mobile",390,844]]) {
      await page.setViewportSize({width,height});
      for (const picture of await page.locator("[data-rag-image]").all()) {
        await picture.scrollIntoViewIfNeeded();
        await picture.evaluate((img) => img.decode());
      }
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1), `${name} 出现横向溢出`);
      // 先截真实视口，再在长图中隐藏吸顶导航，避免导航横跨多组资料。
      await page.locator(".rag-match").first().evaluate((element) => {
        const top = element.getBoundingClientRect().top + window.scrollY;
        const header = document.querySelector(".topbar").getBoundingClientRect().height;
        window.scrollTo({top: top - header - 16, behavior: "instant"});
      });
      await page.screenshot({path: path.join(output, `rag-live-viewport-${name}.png`)});
      await page.locator(".rag-matches").screenshot({path: path.join(output, `rag-live-${name}.png`),
        style: ".topbar { visibility: hidden !important; }"});
    }
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({状态: "通过", 组数: result.matches.length, 图片数: imageCount, 截图目录: output}));
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });

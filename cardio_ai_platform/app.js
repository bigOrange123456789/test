const state = {
  activeRoute: location.hash.replace("#", "") || "dashboard",
  selectedCaseId: "CV-DEMO-001",
  selectedPaperId: "paper-arcade",
  selectedKnowledgeNode: "冠心病",
  selectedSourceId: "source-papers",
  analysisIndex: -1,
  analyzing: false,
  analysisStatus: "idle",
  selectedAnalysisModel: "deepseek-original",
  analysisInputs: null,
  analysisResult: null,
  analysisDraft: {
    diagnosis: "",
    findings: "",
    analysis: "",
    advice: "",
  },
  analysisError: "",
  analysisRunId: 0,
  typingField: "",
  reportGenerated: false,
  uploadState: "等待上传",
  paperFilter: "全部",
  paperSearch: "",
};

let analysisTimer = null;
let typewriterTimer = null;

const ANALYSIS_API_URL = "/api/analyze";
const ANALYSIS_DEBUG = true;

const analysisModels = [
  { id: "deepseek-original", label: "原本deepseek（本地）" },
  { id: "deepseek-finetuned", label: "微调deepseek（LoRA）" },
  { id: "remote-api-1", label: "远程 API-1（deepseek）" },
  { id: "remote-api-2", label: "远程 API-2（通义）" },
];

const analysisReportFields = [
  { key: "diagnosis", label: "综合诊断结果" },
  { key: "findings", label: "关键发现" },
  { key: "analysis", label: "AI 分析" },
  { key: "advice", label: "临床建议" },
];

function analysisDebug(label, data) {
  if (!ANALYSIS_DEBUG) return;
  if (data === undefined) {
    console.info(`[CardioAI][病例分析] ${label}`);
    return;
  }
  console.info(`[CardioAI][病例分析] ${label}`, data);
}

function analysisInputLengths(inputs = {}) {
  return Object.fromEntries(
    Object.entries(inputs).map(([key, value]) => [key, String(value ?? "").length])
  );
}

function analysisReportLengths(report = {}) {
  return Object.fromEntries(
    analysisReportFields.map(({ key }) => [key, String(report[key] ?? "").length])
  );
}

function analysisPayloadSummary(payload) {
  return {
    model: payload?.model || "",
    inputKeys: Object.keys(payload?.inputs || {}),
    inputLengths: analysisInputLengths(payload?.inputs || {}),
  };
}

const navGroups = [
  {
    title: "系统总览",
    items: [{ id: "dashboard", label: "系统总览", icon: "H" }],
  },
  {
    title: "智能诊疗",
    items: [
      { id: "diagnosis", label: "病例分析", icon: "D" },
      { id: "prediction", label: "风险预测", icon: "R" },
      { id: "reports", label: "医学报告", icon: "P" },
    ],
  },
  {
    title: "AI 模型",
    items: [
      { id: "models", label: "模型中心", icon: "M" },
      { id: "training", label: "知识迁移", icon: "T" },
      { id: "fusion", label: "知识融合", icon: "F" },
    ],
  },
  {
    title: "医学知识",
    items: [
      { id: "knowledge", label: "知识库", icon: "K" },
      { id: "literature", label: "医学论文", icon: "L" },
      { id: "cases", label: "病例中心", icon: "C" },
    ],
  },
  {
    title: "证据与数据",
    items: [
      { id: "evidence", label: "证据一致性", icon: "E" },
      { id: "data", label: "数据中心", icon: "S" },
      { id: "settings", label: "系统设置", icon: "G" },
    ],
  },
];

const pageMeta = {
  dashboard: {
    title: "系统总览 Dashboard",
    eyebrow: "Data to Evidence to Clinical Decision",
  },
  diagnosis: {
    title: "智能诊疗分析",
    eyebrow: "Multimodal Case Reasoning",
  },
  prediction: {
    title: "心血管疾病风险预测",
    eyebrow: "Structured Task Prediction",
  },
  reports: {
    title: "智能医学报告",
    eyebrow: "Constrained Clinical Report Generation",
  },
  models: { title: "AI 模型中心", eyebrow: "Domain Adapted Medical LLM" },
  training: { title: "知识迁移与模型训练", eyebrow: "Instruction, Domain, Task" },
  fusion: { title: "多源医学知识融合", eyebrow: "Paper, Case, Guideline, Imaging" },
  knowledge: { title: "医学知识库", eyebrow: "Cardiovascular Knowledge Graph" },
  literature: { title: "医学学术文献", eyebrow: "Evidence and Literature Mining" },
  cases: { title: "病例中心", eyebrow: "Research Demo Case Registry" },
  evidence: { title: "证据一致性分析", eyebrow: "Evidence Grounding and Reliability" },
  data: { title: "数据中心", eyebrow: "Dataset Monitoring" },
  settings: { title: "系统设置", eyebrow: "Mock API Integration Surface" },
};

const externalSources = [
  {
    id: "src-angio-stenosis",
    type: "影像案例",
    title: "冠脉造影狭窄示例",
    source: "Wikimedia Commons / BMC Cancer case image",
    note: "公开冠脉造影图像，用于展示狭窄证据与影像分析卡片。",
    href: "https://commons.wikimedia.org/wiki/File:Angiography_coronary_stenosis_01.jpg",
    image:
      "https://commons.wikimedia.org/wiki/Special:Redirect/file/Angiography_coronary_stenosis_01.jpg",
    license: "CC BY 2.0",
  },
  {
    id: "src-ct-angio",
    type: "CT 影像",
    title: "双源 CT 冠脉成像示例",
    source: "Wikimedia Commons / Eur Radiol image",
    note: "公开 CT 血管成像图，用于表现多模态影像输入入口。",
    href: "https://commons.wikimedia.org/wiki/File:Ct-angiography.png",
    image: "https://commons.wikimedia.org/wiki/Special:Redirect/file/Ct-angiography.png",
    license: "CC BY 2.5",
  },
  {
    id: "src-physionet",
    type: "ECG 数据",
    title: "MIT-BIH Arrhythmia Database",
    source: "PhysioNet",
    note: "经典公开心电数据库，可作为心律失常识别、ECG 特征提取和模型评估示例。",
    href: "https://www.physionet.org/content/mitdb/1.0.0/",
    license: "Open Data Commons Attribution License",
  },
  {
    id: "src-arcade",
    type: "XCA 数据集",
    title: "ARCADE 冠脉造影标注数据集",
    source: "Scientific Data, 2024",
    note: "面向冠脉节段分类与狭窄检测的专家标注 XCA 图像数据集。",
    href: "https://www.nature.com/articles/s41597-023-02871-z",
    license: "Article and dataset source linked",
  },
  {
    id: "src-cadica",
    type: "ICA 数据集",
    title: "CADICA 冠心病造影数据集",
    source: "Mendeley Data, 2024",
    note: "包含 ICA 视频、病灶框标注和选定临床特征，可用于 CAD 辅助评估原型。",
    href: "https://data.mendeley.com/datasets/p9bpx9ctcv/2",
    license: "Mendeley Data source linked",
  },
  {
    id: "src-frontiers-ai-cvd",
    type: "综述文献",
    title: "AI applied in cardiovascular disease",
    source: "Frontiers in Cardiovascular Medicine, 2024",
    note: "用于展示 AI+心血管研究趋势、关键词和知识抽取模块。",
    href: "https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2024.1323918/full",
    license: "Open access article",
  },
];

const metrics = [
  { label: "知识库条目", value: "128,420", trend: "+2.8% 本周" },
  { label: "医学论文", value: "4,611", trend: "AI-CVD 文献池" },
  { label: "病例报告", value: "18,760", trend: "Research Demo" },
  { label: "证据一致性", value: "94.7%", trend: "Mock evaluation" },
  { label: "风险预测 AUC", value: "0.91", trend: "Validation mock" },
  { label: "报告完整性", value: "100%", trend: "Structured output" },
  { label: "当前模型", value: "CVD-LLM-R2", trend: "LoRA adapter" },
  { label: "任务队列", value: "7", trend: "2 running" },
];

const capabilities = [
  ["医学知识理解", "医学术语、指南片段、论文摘要与病例描述统一建模。"],
  ["医学影像分析", "冠脉造影、CT、X-Ray 等影像入口已预留。"],
  ["病例分析", "将病史、症状、检查、报告转化为结构化医学特征。"],
  ["风险预测", "输出疾病概率、风险等级、置信度和模型解释。"],
  ["医学报告生成", "按检查所见、异常、判断、建议和证据生成报告。"],
  ["证据验证", "核对 AI 结论是否由病例、影像、报告和知识支持。"],
  ["多步医学推理", "从输入解析、知识检索到结论追踪形成流程化展示。"],
];

const cases = [
  {
    id: "CV-DEMO-001",
    patient: "Research Demo Patient A",
    disease: "疑似冠心病",
    age: 58,
    sex: "男",
    bmi: 27.6,
    bloodPressure: "152/94 mmHg",
    heartRate: "86 bpm",
    risk: "高风险",
    confidence: 91.6,
    consistency: 94.7,
    status: "已完成 AI 分析",
    symptoms: ["间歇性胸痛", "胸闷", "活动后气短"],
    history: "高血压 8 年；LDL-C 升高；父亲有冠心病史。",
    exams: [
      "ECG: ST-T 改变，V4-V6 导联轻度压低",
      "Laboratory: LDL-C 4.2 mmol/L，hs-CRP 轻度升高",
      "Coronary CTA: 左前降支近段疑似钙化斑块",
      "Coronary angiography: 局部管腔狭窄影像证据",
    ],
    findings: ["冠状动脉局部狭窄", "LDL-C 升高", "高血压病史", "心电图异常"],
    evidence: [
      "LDL-C = 4.2 mmol/L",
      "既往高血压病史 8 年",
      "ECG 提示 ST-T 异常",
      "冠脉造影示局部狭窄影像证据",
    ],
    recommendation:
      "建议进一步完成冠脉相关检查，结合 ASCVD 风险评估，由心血管专科医生进行最终诊断与治疗决策。",
  },
  {
    id: "CV-DEMO-002",
    patient: "Research Demo Patient B",
    disease: "心力衰竭风险评估",
    age: 66,
    sex: "女",
    bmi: 24.1,
    bloodPressure: "138/82 mmHg",
    heartRate: "98 bpm",
    risk: "中风险",
    confidence: 84.2,
    consistency: 91.2,
    status: "待复核",
    symptoms: ["呼吸困难", "下肢水肿", "夜间阵发性气短"],
    history: "糖尿病 10 年；疑似慢性心衰病史；近期运动耐量下降。",
    exams: [
      "Echocardiography: EF 48%，左房轻度增大",
      "BNP: 356 pg/mL",
      "ECG: 窦性心律，偶发房早",
      "Chest X-Ray: 心影轻度增大",
    ],
    findings: ["BNP 升高", "EF 边界下降", "运动耐量下降"],
    evidence: ["BNP = 356 pg/mL", "EF = 48%", "患者主诉活动后呼吸困难"],
    recommendation:
      "建议动态复查 BNP 和超声心动图，评估容量负荷及合并症，并由临床医生综合判断。",
  },
  {
    id: "CV-DEMO-003",
    patient: "Research Demo Patient C",
    disease: "心律失常筛查",
    age: 44,
    sex: "男",
    bmi: 22.8,
    bloodPressure: "124/76 mmHg",
    heartRate: "112 bpm",
    risk: "低-中风险",
    confidence: 78.5,
    consistency: 88.6,
    status: "分析中",
    symptoms: ["心悸", "偶发胸闷"],
    history: "无明确冠心病史；近期睡眠不足；咖啡摄入增加。",
    exams: [
      "ECG: 窦性心动过速",
      "Holter: 偶发室上性早搏",
      "Laboratory: 电解质未见明显异常",
    ],
    findings: ["窦性心动过速", "偶发室上性早搏"],
    evidence: ["ECG 心率 112 bpm", "Holter 记录偶发室上性早搏"],
    recommendation:
      "建议评估诱因并随访 Holter，若症状加重需进一步心血管专科评估。",
  },
];

const papers = [
  {
    id: "paper-arcade",
    title:
      "Dataset for Automatic Region-based Coronary Artery Disease Diagnostics Using X-Ray Angiography Images",
    authors: "Popov et al.",
    year: "2024",
    field: "冠脉造影 / CAD",
    tags: ["XCA", "Stenosis Detection", "Dataset"],
    source: "Scientific Data",
    href: "https://www.nature.com/articles/s41597-023-02871-z",
    abstract:
      "ARCADE 数据集聚焦 X 射线冠脉造影图像，支持冠脉节段分类与狭窄检测两个任务，可作为影像 AI 和风险评估模块的公开案例。",
    extraction: ["冠脉节段", "狭窄斑块", "专家标注", "自动诊断基准"],
  },
  {
    id: "paper-frontiers",
    title: "Artificial intelligence applied in cardiovascular disease: a bibliometric and visual analysis",
    authors: "Zhang et al.",
    year: "2024",
    field: "AI + CVD 研究趋势",
    tags: ["Bibliometric", "Diagnosis", "Risk"],
    source: "Frontiers in Cardiovascular Medicine",
    href: "https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2024.1323918/full",
    abstract:
      "该文献从 2000-2023 年文献中分析心血管 AI 研究热点，为平台的研究态势、关键词和知识抽取展示提供依据。",
    extraction: ["诊断", "风险预测", "医学影像", "研究热点"],
  },
  {
    id: "paper-cadica",
    title: "CADICA: a new dataset for coronary artery disease",
    authors: "Mendeley Data contributors",
    year: "2024",
    field: "ICA 视频 / 病灶标注",
    tags: ["ICA", "Bounding Box", "Clinical Features"],
    source: "Mendeley Data",
    href: "https://data.mendeley.com/datasets/p9bpx9ctcv/2",
    abstract:
      "CADICA 包含冠脉造影视频、病灶框和部分临床特征，适合用于病例中心与影像证据追踪的演示数据源。",
    extraction: ["患者级数据", "ICA 视频", "病灶定位", "临床特征"],
  },
  {
    id: "paper-mitbih",
    title: "MIT-BIH Arrhythmia Database",
    authors: "Moody and Mark / PhysioNet",
    year: "2005 version",
    field: "心电图 / 心律失常",
    tags: ["ECG", "Arrhythmia", "Benchmark"],
    source: "PhysioNet",
    href: "https://www.physionet.org/content/mitdb/1.0.0/",
    abstract:
      "MIT-BIH 是经典心律失常心电数据库，可用于心电波形浏览、节律分类、风险预测和模型评估示例。",
    extraction: ["双通道 ECG", "心律失常", "标注节拍", "模型评估"],
  },
];

const knowledgeNodes = {
  冠心病: {
    type: "疾病",
    description: "冠状动脉粥样硬化导致供血不足，临床可表现为心绞痛、心肌梗死等。",
    relations: ["胸痛", "LDL-C 升高", "冠脉造影", "抗血小板治疗"],
  },
  胸痛: {
    type: "症状",
    description: "劳力性或间歇性胸痛是 CAD 风险分析中的关键症状证据。",
    relations: ["冠心病", "ECG", "肌钙蛋白"],
  },
  "LDL-C 升高": {
    type: "危险因素",
    description: "血脂异常与动脉粥样硬化进展相关，是风险分层中的重要变量。",
    relations: ["冠心病", "血脂管理指南"],
  },
  冠脉造影: {
    type: "检查",
    description: "冠脉造影可以直观呈现管腔狭窄、闭塞和介入治疗相关证据。",
    relations: ["冠心病", "狭窄比例", "影像证据"],
  },
  ECG: {
    type: "检查",
    description: "心电图可提供 ST-T 改变、心律失常和心肌缺血相关线索。",
    relations: ["胸痛", "心律失常", "风险预测"],
  },
  抗血小板治疗: {
    type: "治疗方案",
    description: "具体治疗方案必须由执业医师结合指南、禁忌证和患者情况决定。",
    relations: ["冠心病", "临床建议"],
  },
};

const workflow = [
  ["输入病例解析", "读取患者基本信息、症状、检查、报告和影像。", 100],
  ["医学知识检索", "检索到 12 条相关医学知识、3 篇论文和 2 条指南片段。", 96],
  ["领域知识匹配", "匹配冠心病、血脂异常、高血压、ST-T 改变等实体。", 93],
  ["病例特征分析", "检测到 5 项关键临床特征并生成结构化特征向量。", 90],
  ["疾病关联分析", "冠状动脉粥样硬化性心脏病关联强度较高。", 88],
  ["风险预测", "冠心病风险：High；Confidence：91.6%。", 92],
  ["证据一致性检查", "Evidence Consistency：94.7%，未发现冲突证据。", 95],
  ["生成医学报告", "按照九段式报告结构生成研究演示报告。", 100],
];

const sources = [
  {
    id: "source-instruction",
    title: "通用指令问答",
    count: "320K",
    quality: "96%",
    status: "已入库",
    examples: ["通用医学问答", "指令跟随", "自然语言推理"],
  },
  {
    id: "source-papers",
    title: "医学学术论文",
    count: "4,611",
    quality: "93%",
    status: "持续更新",
    examples: ["心血管疾病研究", "影像学文献", "AI 诊断综述"],
  },
  {
    id: "source-cases",
    title: "病例诊断报告",
    count: "18,760",
    quality: "91%",
    status: "脱敏演示",
    examples: ["冠心病病例", "心衰病例", "心电图报告"],
  },
  {
    id: "source-guideline",
    title: "临床指南",
    count: "86",
    quality: "98%",
    status: "专家审核",
    examples: ["冠心病指南", "高血压指南", "血脂管理指南"],
  },
  {
    id: "source-imaging",
    title: "透视成像规则",
    count: "1,240",
    quality: "89%",
    status: "规则标注",
    examples: ["冠脉投照体位", "影像质量规则", "病灶定位"],
  },
];

function h(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => {
    const map = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;",
    };
    return map[char];
  });
}

function currentCase() {
  return cases.find((item) => item.id === state.selectedCaseId) || cases[0];
}

function currentPaper() {
  return papers.find((item) => item.id === state.selectedPaperId) || papers[0];
}

function selectedAnalysisModelLabel() {
  return (
    analysisModels.find((model) => model.id === state.selectedAnalysisModel)?.label ||
    state.selectedAnalysisModel
  );
}

function routeLabel(routeId) {
  for (const group of navGroups) {
    const hit = group.items.find((item) => item.id === routeId);
    if (hit) return hit.label;
  }
  return pageMeta.dashboard.title;
}

function progress(value, variant = "") {
  return `<div class="progress-line ${variant}"><span style="width:${Math.max(
    0,
    Math.min(100, value)
  )}%"></span></div>`;
}

function tags(items, variant = "") {
  return `<div class="tag-row">${items
    .map((item) => `<span class="tag ${variant}">${h(item)}</span>`)
    .join("")}</div>`;
}

function emptyAnalysisReport() {
  return {
    diagnosis: "",
    findings: "",
    analysis: "",
    advice: "",
  };
}

function defaultAnalysisInputs(item = currentCase()) {
  return {
    age: String(item.age ?? ""),
    sex: item.sex || "",
    bmi: String(item.bmi ?? ""),
    bloodPressure: item.bloodPressure || "",
    heartRate: item.heartRate || "",
    familyHistory: "冠心病家族史：阳性",
    caseInput: item.history,
    symptoms: item.symptoms.join("、"),
    exams: item.exams.join("\n"),
    diagnosisReport: `主诉：${item.symptoms.join("、")}
病史：${item.history}
初步意见：需要结合心电图、血脂、冠脉影像和医生判断。`,
  };
}

function analysisInputValue(key, item = currentCase()) {
  const currentInputs = {
    ...defaultAnalysisInputs(item),
    ...(state.analysisInputs || {}),
  };
  return currentInputs[key] ?? "";
}

function collectAnalysisPayload() {
  const inputs = {
    age: document.querySelector('[data-analysis-meta="age"]')?.value || "",
    sex: document.querySelector('[data-analysis-meta="sex"]')?.value || "",
    bmi: document.querySelector('[data-analysis-meta="bmi"]')?.value || "",
    bloodPressure:
      document.querySelector('[data-analysis-meta="bloodPressure"]')?.value || "",
    heartRate: document.querySelector('[data-analysis-meta="heartRate"]')?.value || "",
    familyHistory:
      document.querySelector('[data-analysis-meta="familyHistory"]')?.value || "",
    caseInput: document.querySelector('[data-analysis-input="caseInput"]')?.value || "",
    symptoms: document.querySelector('[data-analysis-input="symptoms"]')?.value || "",
    exams: document.querySelector('[data-analysis-input="exams"]')?.value || "",
    diagnosisReport:
      document.querySelector('[data-analysis-input="diagnosisReport"]')?.value || "",
  };
  const checkedSymptoms = Array.from(
    document.querySelectorAll("[data-symptom-checkbox]:checked")
  )
    .map((input) => input.getAttribute("data-symptom-checkbox"))
    .filter(Boolean);
  if (!inputs.symptoms.trim() && checkedSymptoms.length) {
    inputs.symptoms = checkedSymptoms.join("、");
  }
  const model =
    document.querySelector("[data-analysis-model]")?.value || state.selectedAnalysisModel;
  return { model, inputs };
}

function stringifyAnalysisValue(value) {
  if (Array.isArray(value)) return value.map((item) => String(item)).join("\n");
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}

function pickAnalysisField(source, keys) {
  for (const key of keys) {
    if (Object.prototype.hasOwnProperty.call(source, key)) {
      return stringifyAnalysisValue(source[key]);
    }
  }
  return "";
}

function extractAnalysisReport(payload) {
  if (!payload || typeof payload !== "object") return null;
  const source =
    payload.data && typeof payload.data === "object"
      ? payload.data
      : payload.result && typeof payload.result === "object"
        ? payload.result
        : payload.delta && typeof payload.delta === "object"
          ? payload.delta
          : payload;
  const report = {
    diagnosis: pickAnalysisField(source, ["diagnosis", "综合诊断结果", "综合诊断", "result"]),
    findings: pickAnalysisField(source, ["findings", "关键发现", "key_findings", "keyFindings"]),
    analysis: pickAnalysisField(source, ["analysis", "AI分析", "AI 分析", "ai_analysis", "aiAnalysis"]),
    advice: pickAnalysisField(source, ["advice", "临床建议", "recommendation", "recommendations"]),
  };
  return Object.values(report).some((value) => value.trim()) ? report : null;
}

function normalizeAnalysisReport(payload) {
  const report = extractAnalysisReport(payload);
  if (!report) {
    throw new Error("API 响应中没有 diagnosis/findings/analysis/advice 字段");
  }
  return report;
}

function mergeAnalysisReports(base, update) {
  const merged = { ...emptyAnalysisReport(), ...(base || {}) };
  for (const { key } of analysisReportFields) {
    if (update?.[key]?.trim()) {
      merged[key] = update[key];
    }
  }
  return merged;
}

function collectJsonCandidates(text) {
  const trimmed = text.trim();
  if (!trimmed) return [];
  const candidates = [trimmed];
  for (const eventBlock of trimmed.split(/\r?\n\r?\n/)) {
    const data = eventBlock
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .filter((line) => line && line !== "[DONE]")
      .join("\n");
    if (data) candidates.push(data);
  }
  for (const line of trimmed.split(/\r?\n/)) {
    const candidate = line.trim().startsWith("data:")
      ? line.trim().slice(5).trim()
      : line.trim();
    if (candidate && candidate !== "[DONE]") candidates.push(candidate);
  }
  const firstBrace = trimmed.indexOf("{");
  const lastBrace = trimmed.lastIndexOf("}");
  if (firstBrace >= 0 && lastBrace > firstBrace) {
    candidates.push(trimmed.slice(firstBrace, lastBrace + 1));
  }
  return Array.from(new Set(candidates));
}

function parseAnalysisStreamPayload(text) {
  const candidates = collectJsonCandidates(text);
  let mergedReport = null;
  for (const candidate of candidates) {
    try {
      const payload = JSON.parse(candidate);
      const report = extractAnalysisReport(payload);
      if (report) mergedReport = mergeAnalysisReports(mergedReport, report);
    } catch (error) {
      // Partial stream chunks are expected to be invalid JSON until enough text arrives.
    }
  }
  return mergedReport;
}

function parseAnalysisResponseText(text, contentType = "") {
  const report = parseAnalysisStreamPayload(text);
  if (report) return report;

  const trimmed = text.trim();
  const isSse = contentType.includes("text/event-stream") || trimmed.startsWith("data:");
  if (isSse) {
    throw new Error("API 返回了 SSE 流，但没有包含 diagnosis/findings/analysis/advice 字段");
  }

  try {
    return normalizeAnalysisReport(JSON.parse(trimmed));
  } catch (error) {
    throw new Error(`API 响应不是有效的结构化 JSON：${error.message}`);
  }
}

async function readAnalysisResponse(response) {
  const contentType = response.headers?.get("Content-Type") || "";
  const reader = response.body?.getReader();
  analysisDebug("response:body-reader", {
    hasReader: Boolean(reader),
    contentType,
  });
  if (!reader) {
    const text = await response.text();
    analysisDebug("response:text-body", {
      chars: text.length,
      preview: text.slice(0, 320),
    });
    return parseAnalysisResponseText(text, contentType);
  }

  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let latestReport = null;
  let chunkIndex = 0;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    chunkIndex += 1;
    buffer += decoder.decode(value, { stream: true });
    const streamedReport = parseAnalysisStreamPayload(buffer);
    if (streamedReport) latestReport = mergeAnalysisReports(latestReport, streamedReport);
    analysisDebug("stream:chunk", {
      chunkIndex,
      bytes: value?.byteLength || 0,
      bufferChars: buffer.length,
      latestFieldLengths: latestReport ? analysisReportLengths(latestReport) : null,
    });
  }

  buffer += decoder.decode();
  const finalReport = parseAnalysisStreamPayload(buffer);
  analysisDebug("stream:done", {
    chunks: chunkIndex,
    bufferChars: buffer.length,
    hasFinalReport: Boolean(finalReport),
    hasLatestReport: Boolean(latestReport),
  });
  if (finalReport) {
    const merged = mergeAnalysisReports(latestReport, finalReport);
    analysisDebug("stream:report", {
      fieldLengths: analysisReportLengths(merged),
    });
    return merged;
  }
  if (latestReport) {
    analysisDebug("stream:report", {
      fieldLengths: analysisReportLengths(latestReport),
    });
    return latestReport;
  }
  analysisDebug("stream:unparsed-buffer-preview", buffer.slice(0, 600));
  return parseAnalysisResponseText(buffer, contentType);
}

async function requestAnalysis(payload) {
  const startedAt = performance.now();
  analysisDebug("request:start", {
    url: ANALYSIS_API_URL,
    ...analysisPayloadSummary(payload),
  });
  const response = await fetch(ANALYSIS_API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json, text/event-stream",
    },
    body: JSON.stringify(payload),
  });
  analysisDebug("response:headers", {
    ok: response.ok,
    status: response.status,
    statusText: response.statusText,
    contentType: response.headers?.get("Content-Type") || "",
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    analysisDebug("response:error-body", detail.slice(0, 600));
    throw new Error(
      `${ANALYSIS_API_URL} 返回 ${response.status} ${response.statusText}${
        detail ? `：${detail.slice(0, 240)}` : ""
      }`
    );
  }
  const report = await readAnalysisResponse(response);
  analysisDebug("request:complete", {
    elapsedMs: Math.round(performance.now() - startedAt),
    fieldLengths: analysisReportLengths(report),
  });
  return report;
}

function renderSidebar() {
  return `
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">AI</div>
        <div>
          <p class="brand-title">心血管疾病人工智能诊疗平台</p>
          <span class="brand-subtitle">Research Demo / Mock Data</span>
        </div>
      </div>
      ${navGroups
        .map(
          (group) => `
          <nav class="nav-group" aria-label="${h(group.title)}">
            <p class="nav-group-title">${h(group.title)}</p>
            ${group.items
              .map(
                (item) => `
                <button class="nav-button ${
                  state.activeRoute === item.id ? "active" : ""
                }" data-route="${h(item.id)}" title="${h(item.label)}">
                  <span class="nav-icon" aria-hidden="true">${h(item.icon)}</span>
                  <span class="nav-label">${h(item.label)}</span>
                  ${state.activeRoute === item.id ? '<span class="nav-status-dot"></span>' : ""}
                </button>`
              )
              .join("")}
          </nav>`
        )
        .join("")}
      <div class="sidebar-footer">
        本系统用于医学人工智能科研与临床辅助决策研究；所有病例与 AI 输出均为演示 Mock，不代表真实诊断。
      </div>
    </aside>
  `;
}

function renderTopbar() {
  const meta = pageMeta[state.activeRoute] || pageMeta.dashboard;
  const modelLabel =
    state.activeRoute === "diagnosis" ? selectedAnalysisModelLabel() : "CVD-LLM-R2";
  return `
    <header class="topbar">
      <div class="breadcrumb">
        <span class="eyebrow">${h(meta.eyebrow)}</span>
        <h1 class="page-title">${h(meta.title)}</h1>
      </div>
      <div class="topbar-actions">
        <span class="status-pill">Model: ${h(modelLabel)}</span>
        <span class="status-pill">System: Online</span>
        <span class="status-pill warning">Mode: Research Demo</span>
      </div>
    </header>
  `;
}

function renderLayout() {
  return `
    <div class="app-shell">
      ${renderSidebar()}
      <div class="content">
        ${renderTopbar()}
        <main class="page-main">${renderPage()}</main>
      </div>
    </div>
  `;
}

function renderDashboard() {
  return `
    <div class="page-grid">
      <section class="hero-workspace">
        <div class="research-frame">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">数据 → 知识 → 模型 → 推理 → 证据 → 诊疗</h2>
              <p class="panel-kicker">科研级多模态知识融合流程，突出结构化预测与证据一致性约束。</p>
            </div>
            <span class="tag">核心流程</span>
          </div>
          <div class="framework-map" aria-label="系统核心流程图">
            <span class="flow-link" style="left:25%;top:23%;width:25%;"></span>
            <span class="flow-link" style="left:50%;top:23%;width:25%;"></span>
            <span class="flow-link vertical" style="left:50%;top:27%;height:18%;"></span>
            <span class="flow-link vertical" style="left:50%;top:51%;height:17%;"></span>
            <span class="flow-link" style="left:31%;top:73%;width:39%;"></span>
            <div class="flow-node accent" style="grid-column:1;grid-row:1;"><strong>通用知识</strong><span>指令问答、推理、基础对话能力</span></div>
            <div class="flow-node accent" style="grid-column:2;grid-row:1;"><strong>医学论文</strong><span>心血管文献、临床指南、影像研究</span></div>
            <div class="flow-node accent" style="grid-column:3;grid-row:1;"><strong>病例报告</strong><span>症状、检查、诊断报告、影像描述</span></div>
            <div class="flow-node accent" style="grid-column:4;grid-row:1;"><strong>透视成像规则</strong><span>投照体位、管腔狭窄、病灶定位</span></div>
            <div class="flow-node primary" style="grid-column:2 / span 2;grid-row:2;"><strong>医学知识融合</strong><span>实体抽取、关系构建、证据索引、向量检索</span></div>
            <div class="flow-node primary" style="grid-column:2 / span 2;grid-row:3;"><strong>医疗大模型与领域知识迁移</strong><span>LoRA / Adapter / 医学任务适配</span></div>
            <div class="flow-node" style="grid-column:1;grid-row:4;"><strong>风险预测</strong><span>CAD、心衰、心梗、心律失常</span></div>
            <div class="flow-node" style="grid-column:2;grid-row:4;"><strong>结构化生成</strong><span>所见、异常、判断、建议、证据</span></div>
            <div class="flow-node warn" style="grid-column:3;grid-row:4;"><strong>证据一致性检查</strong><span>防止无依据结论和证据冲突</span></div>
            <div class="flow-node primary" style="grid-column:4;grid-row:4;"><strong>临床辅助决策</strong><span>科研输出，仅供专业人员参考</span></div>
          </div>
          <div class="framework-caption">
            ${tags(["General Knowledge", "Medical Knowledge", "Evidence Grounding", "Structured Report"])}
          </div>
        </div>
        <div class="page-grid">
          <div class="notice research">
            <strong>科研安全提示</strong>
            <span>本原型展示前端形态与交互流程，AI 结论为 Mock 数据，不替代执业医师诊断。</span>
          </div>
          <div class="panel">
            <div class="panel-header">
              <div>
                <h2 class="panel-title">当前系统状态</h2>
                <p class="panel-kicker">科研平台常用指标，等待接入真实后端。</p>
              </div>
            </div>
            <div class="page-grid grid-2">
              ${metrics
                .slice(0, 4)
                .map(
                  (item) => `
                    <div class="metric-card">
                      <span class="metric-label">${h(item.label)}</span>
                      <strong class="metric-value">${h(item.value)}</strong>
                      <span class="metric-trend">${h(item.trend)}</span>
                    </div>`
                )
                .join("")}
            </div>
          </div>
        </div>
      </section>
      <section class="page-grid grid-4">
        ${metrics
          .slice(4)
          .map(
            (item) => `
              <div class="metric-card">
                <span class="metric-label">${h(item.label)}</span>
                <strong class="metric-value">${h(item.value)}</strong>
                <span class="metric-trend">${h(item.trend)}</span>
              </div>`
          )
          .join("")}
      </section>
      <section class="page-grid grid-3">
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">核心数据来源</h2>
              <p class="panel-kicker">多源数据统一进入知识融合层。</p>
            </div>
          </div>
          <div class="page-grid">
            ${sources
              .slice(0, 3)
              .map(
                (source) => `
                <button class="source-card" data-route="fusion" data-source="${h(source.id)}">
                  <h3>${h(source.title)} <span class="tag gray">${h(source.count)}</span></h3>
                  <p>${h(source.examples.join(" / "))}</p>
                </button>`
              )
              .join("")}
          </div>
        </div>
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">模型能力</h2>
              <p class="panel-kicker">最终系统能力完整展示，当前由 Mock API 驱动。</p>
            </div>
          </div>
          <div class="page-grid">
            ${capabilities
              .slice(0, 4)
              .map(
                ([title, copy]) => `
                <div class="ability-card">
                  <h3>${h(title)}</h3>
                  <p>${h(copy)}</p>
                </div>`
              )
              .join("")}
          </div>
        </div>
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">公开案例资料</h2>
              <p class="panel-kicker">来自公开网页的影像、ECG 与文献入口。</p>
            </div>
          </div>
          <div class="page-grid">
            ${externalSources
              .slice(0, 4)
              .map(
                (src) => `
                <a class="source-card" href="${h(src.href)}" target="_blank" rel="noreferrer">
                  <h3>${h(src.title)} <span class="tag gray">${h(src.type)}</span></h3>
                  <p>${h(src.note)}</p>
                </a>`
              )
              .join("")}
          </div>
        </div>
      </section>
    </div>
  `;
}

function renderDiagnosis() {
  const item = currentCase();
  const finished = state.analysisIndex >= workflow.length;
  const analysisBusy = state.analysisStatus === "running" || state.analysisStatus === "typing";
  const statusLabel =
    state.analysisStatus === "typing"
      ? "Rendering"
      : state.analysisStatus === "complete"
        ? "Completed"
        : state.analysisStatus === "error"
          ? "Error"
          : state.analysisStatus === "running" && !state.analyzing
            ? "Waiting API"
            : state.analyzing
              ? "Running"
              : "Ready";
  return `
    <div class="page-grid">
      <div class="notice research">
        <strong>AI-generated / Research Demo</strong>
        <span>病例分析模块已接入 /api/analyze，当前输出由后端返回的结构化 JSON 动态驱动。</span>
      </div>
      <section class="diagnosis-layout">
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">左侧：病例输入</h2>
              <p class="panel-kicker">支持结构化信息、报告文本和医学影像上传入口。</p>
            </div>
          </div>
          <div class="input-grid">
            <div class="field"><label>年龄</label><input class="input" data-analysis-meta="age" value="${h(analysisInputValue("age", item))}" /></div>
            <div class="field"><label>性别</label><select class="select" data-analysis-meta="sex">
              ${["男", "女"]
                .map(
                  (sex) => `<option value="${h(sex)}" ${analysisInputValue("sex", item) === sex ? "selected" : ""}>${h(sex)}</option>`
                )
                .join("")}
            </select></div>
            <div class="field"><label>BMI</label><input class="input" data-analysis-meta="bmi" value="${h(analysisInputValue("bmi", item))}" /></div>
            <div class="field"><label>血压</label><input class="input" data-analysis-meta="bloodPressure" value="${h(analysisInputValue("bloodPressure", item))}" /></div>
            <div class="field"><label>心率</label><input class="input" data-analysis-meta="heartRate" value="${h(analysisInputValue("heartRate", item))}" /></div>
            <div class="field"><label>家族史</label><input class="input" data-analysis-meta="familyHistory" value="${h(analysisInputValue("familyHistory", item))}" /></div>
          </div>
          <div class="report-section">
            <h3>病例输入</h3>
            <textarea class="textarea" data-analysis-input="caseInput">${h(analysisInputValue("caseInput", item))}</textarea>
          </div>
          <div class="report-section">
            <h3>临床症状</h3>
            <textarea class="textarea compact-textarea" data-analysis-input="symptoms">${h(analysisInputValue("symptoms", item))}</textarea>
            <div class="check-list">
              ${["胸痛", "胸闷", "心悸", "呼吸困难", "晕厥", "活动后气短"]
                .map(
                  (symptom) => `
                  <label class="check-pill">
                    <input type="checkbox" data-symptom-checkbox="${h(symptom)}" ${item.symptoms.some((s) => s.includes(symptom)) ? "checked" : ""} />
                    ${h(symptom)}
                  </label>`
                )
                .join("")}
            </div>
          </div>
          <div class="report-section">
            <h3>检查结果</h3>
            <textarea class="textarea" data-analysis-input="exams">${h(analysisInputValue("exams", item))}</textarea>
          </div>
          <div class="report-section">
            <h3>病例诊断报告</h3>
            <textarea class="textarea" data-analysis-input="diagnosisReport">${h(analysisInputValue("diagnosisReport", item))}</textarea>
          </div>
          <div class="report-section">
            <h3>医学影像上传</h3>
            <div class="upload-zone">
              <div>
                <strong>${h(state.uploadState)}</strong>
                <span>支持 CT / 冠脉造影 / X-Ray / 透视截图，当前为前端占位。</span>
                <div class="button-row" style="margin-top:12px;justify-content:center;">
                  <button class="button secondary" data-action="mock-upload">模拟上传影像</button>
                </div>
              </div>
            </div>
          </div>
          <div class="report-section">
            <div class="field">
              <label for="analysis-model">AI 模型</label>
              <select id="analysis-model" class="select" data-analysis-model>
                ${analysisModels
                  .map(
                    (model) => `
                    <option value="${h(model.id)}" ${state.selectedAnalysisModel === model.id ? "selected" : ""}>${h(model.label)}</option>`
                  )
                  .join("")}
              </select>
            </div>
          </div>
          <div class="button-row">
            <button class="button" data-action="start-analysis" ${analysisBusy ? "disabled" : ""}>开始 AI 分析</button>
            <button class="button secondary" data-route="cases">选择病例</button>
          </div>
        </div>
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">中间：AI 医学分析过程</h2>
              <p class="panel-kicker">从输入解析到证据检查的医学推理流程可视化。</p>
            </div>
            <span class="tag ${finished ? "" : "gray"}">${h(statusLabel)}</span>
          </div>
          <div class="workflow-list">
            ${workflow
              .map(([title, copy, confidence], index) => {
                const status =
                  index < state.analysisIndex
                    ? "completed"
                    : state.analyzing && index === state.analysisIndex
                      ? "running"
                      : "";
                const mark = status === "completed" ? "✓" : index + 1;
                return `
                  <div class="workflow-step ${status}">
                    <div class="step-index">${h(mark)}</div>
                    <div>
                      <h3 class="step-title">${h(title)}</h3>
                      <p class="step-copy">${h(copy)}</p>
                    </div>
                    <div class="confidence">${status ? h(confidence) + "%" : "--"}</div>
                  </div>`;
              })
              .join("")}
          </div>
        </div>
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">右侧：结构化诊疗结果</h2>
              <p class="panel-kicker">由后端返回的 diagnosis / findings / analysis / advice 字段生成。</p>
            </div>
          </div>
          ${renderStructuredReport()}
        </div>
      </section>
    </div>
  `;
}

function renderStructuredReport() {
  const report = state.analysisDraft || emptyAnalysisReport();
  const placeholder =
    state.analysisStatus === "running"
      ? "AI 分析完成后显示结果。"
      : state.analysisStatus === "error"
        ? "API 请求失败，详情请查看浏览器控制台。"
        : "等待分析结果。";

  return analysisReportFields
    .map(({ key, label }) => {
      const value = report[key] || "";
      const isTyping = state.analysisStatus === "typing" && state.typingField === key;
      return `
        <div class="report-section">
          <h3>${h(label)}</h3>
          <p class="structured-output ${value ? "" : "placeholder"}">${h(value || placeholder)}${
            isTyping ? '<span class="typing-cursor" aria-hidden="true"></span>' : ""
          }</p>
        </div>
      `;
    })
    .join("");
}

function renderEvidence() {
  const item = currentCase();
  const matches = [
    ["冠状动脉局部狭窄", "冠脉造影影像证据", "支持证据", 96],
    ["LDL-C 升高", "实验室检查 LDL-C = 4.2 mmol/L", "支持证据", 99],
    ["高血压相关风险", "高血压病史 8 年", "支持证据", 94],
    ["心电图异常", "ECG ST-T 改变", "部分支持", 82],
    ["明显心肌梗死", "输入证据未出现肌钙蛋白升高", "Evidence Insufficient", 38],
  ];
  return `
    <div class="page-grid">
      <section class="split">
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">证据—结论关系图</h2>
              <p class="panel-kicker">医学证据 → 医学特征 → 疾病判断 → 风险结论。</p>
            </div>
            <span class="tag">Evidence Graph</span>
          </div>
          <div class="relation-graph">
            <svg viewBox="0 0 900 430" preserveAspectRatio="none">
              <line x1="150" y1="80" x2="360" y2="160" stroke="#9ed7d0" stroke-width="2" />
              <line x1="150" y1="210" x2="360" y2="160" stroke="#9ed7d0" stroke-width="2" />
              <line x1="150" y1="340" x2="360" y2="270" stroke="#9ed7d0" stroke-width="2" />
              <line x1="360" y1="160" x2="570" y2="210" stroke="#b7c6ec" stroke-width="2" />
              <line x1="360" y1="270" x2="570" y2="210" stroke="#b7c6ec" stroke-width="2" />
              <line x1="570" y1="210" x2="760" y2="210" stroke="#e6c275" stroke-width="2" />
            </svg>
            <div class="graph-node evidence" style="left:4%;top:10%;">冠脉造影</div>
            <div class="graph-node evidence" style="left:4%;top:40%;">LDL-C 4.2</div>
            <div class="graph-node evidence" style="left:4%;top:70%;">高血压史</div>
            <div class="graph-node feature" style="left:37%;top:26%;">狭窄特征</div>
            <div class="graph-node feature" style="left:37%;top:58%;">危险因素</div>
            <div class="graph-node risk" style="left:63%;top:42%;">冠心病判断</div>
            <div class="graph-node alert" style="left:78%;top:42%;">高风险结论</div>
          </div>
        </div>
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">Consistency Score</h2>
              <p class="panel-kicker">根据输入证据、报告文本和知识库约束计算。</p>
            </div>
          </div>
          <div class="evidence-score">
            <div class="evidence-score-inner">
              <div>
                <strong>${h(item.consistency)}%</strong>
                <span>一致性</span>
              </div>
            </div>
          </div>
          <div class="kv-list" style="margin-top:18px;">
            <div class="kv"><span>支持证据</span><strong>3</strong></div>
            <div class="kv"><span>部分支持</span><strong>1</strong></div>
            <div class="kv"><span>无支持</span><strong>1</strong></div>
            <div class="kv"><span>冲突证据</span><strong>0</strong></div>
          </div>
          <div class="notice" style="margin-top:14px;">
            <strong>Evidence Insufficient</strong>
            <span>“明显心肌梗死”结论缺少肌钙蛋白、动态 ECG 或影像证据，系统建议阻断该结论进入最终报告。</span>
          </div>
        </div>
      </section>
      <section class="panel">
        <div class="panel-header">
          <div>
            <h2 class="panel-title">AI 结论与输入证据匹配</h2>
            <p class="panel-kicker">点击后端接入时可追踪每条结论对应的证据来源。</p>
          </div>
        </div>
        <div class="page-grid grid-3">
          ${matches
            .map(
              ([claim, evidence, status, score]) => `
              <div class="evidence-card">
                <h3>${h(claim)} <span class="tag ${status.includes("Insufficient") ? "red" : status.includes("部分") ? "amber" : ""}">${h(status)}</span></h3>
                <p>${h(evidence)}</p>
                <div style="margin-top:12px;">${progress(score, score < 60 ? "red" : score < 90 ? "amber" : "")}</div>
              </div>`
            )
            .join("")}
        </div>
      </section>
    </div>
  `;
}

function renderKnowledge() {
  const node = knowledgeNodes[state.selectedKnowledgeNode] || knowledgeNodes["冠心病"];
  const nodePositions = [
    ["冠心病", "left:42%;top:40%;", "active"],
    ["胸痛", "left:11%;top:16%;", ""],
    ["LDL-C 升高", "left:11%;top:64%;", ""],
    ["冠脉造影", "left:39%;top:9%;", ""],
    ["ECG", "left:70%;top:18%;", ""],
    ["抗血小板治疗", "left:68%;top:66%;", ""],
  ];
  return `
    <div class="page-grid">
      <section class="split">
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">医学知识图谱</h2>
              <p class="panel-kicker">节点覆盖疾病、症状、检查、风险因素、治疗方案和医学证据。</p>
            </div>
            <span class="tag">Knowledge Graph</span>
          </div>
          <div class="knowledge-graph">
            <svg viewBox="0 0 900 430" preserveAspectRatio="none">
              <line x1="470" y1="190" x2="190" y2="95" stroke="#9ed7d0" stroke-width="2" />
              <line x1="470" y1="190" x2="190" y2="305" stroke="#9ed7d0" stroke-width="2" />
              <line x1="470" y1="190" x2="470" y2="68" stroke="#b7c6ec" stroke-width="2" />
              <line x1="470" y1="190" x2="720" y2="100" stroke="#b7c6ec" stroke-width="2" />
              <line x1="470" y1="190" x2="720" y2="310" stroke="#e6c275" stroke-width="2" />
            </svg>
            ${nodePositions
              .map(
                ([label, style]) => `
                <button class="graph-node clickable ${state.selectedKnowledgeNode === label ? "active" : ""}" style="${style}" data-node="${h(label)}">
                  ${h(label)}
                </button>`
              )
              .join("")}
          </div>
        </div>
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">${h(state.selectedKnowledgeNode)}</h2>
              <p class="panel-kicker">${h(node.type)} / 可接入知识库详情 API。</p>
            </div>
          </div>
          <p style="color:var(--muted);line-height:1.7;">${h(node.description)}</p>
          <div class="report-section">
            <h3>关联实体</h3>
            ${tags(node.relations)}
          </div>
          <div class="report-section">
            <h3>证据约束</h3>
            <p>结论必须被病例报告、检查数值、影像描述或指南知识至少一类证据支持；缺失时标记为 Evidence Insufficient。</p>
          </div>
        </div>
      </section>
      <section class="page-grid grid-3">
        <div class="panel">
          <div class="panel-header"><div><h2 class="panel-title">临床指南</h2><p class="panel-kicker">演示条目，等待后端接入。</p></div></div>
          ${tags(["冠心病指南", "高血压指南", "血脂管理指南", "心衰指南"])}
          <div class="kv-list" style="margin-top:12px;">
            <div class="kv"><span>规范化术语</span><strong>98%</strong></div>
            <div class="kv"><span>专家审核状态</span><strong>Ready</strong></div>
          </div>
        </div>
        <div class="panel">
          <div class="panel-header"><div><h2 class="panel-title">疾病知识</h2><p class="panel-kicker">冠心病结构化知识片段。</p></div></div>
          <div class="kv-list">
            <div class="kv"><span>危险因素</span><strong>高血压 / 高脂血症 / 糖尿病</strong></div>
            <div class="kv"><span>临床表现</span><strong>胸痛 / 胸闷 / 呼吸困难</strong></div>
            <div class="kv"><span>关键检查</span><strong>ECG / CTA / CAG</strong></div>
          </div>
        </div>
        <div class="panel">
          <div class="panel-header"><div><h2 class="panel-title">论文知识抽取</h2><p class="panel-kicker">从文献到知识库的路径。</p></div></div>
          <div class="workflow-list">
            ${["论文解析", "医学实体", "医学关系", "知识审核", "入库索引"]
              .map(
                (label, index) => `
                <div class="workflow-step completed">
                  <div class="step-index">✓</div>
                  <div><h3 class="step-title">${h(label)}</h3><p class="step-copy">Mock pipeline stage ${index + 1}</p></div>
                  <div class="confidence">${96 - index}%</div>
                </div>`
              )
              .join("")}
          </div>
        </div>
      </section>
    </div>
  `;
}

function renderLiterature() {
  const filtered = papers.filter((paper) => {
    const categoryOk = state.paperFilter === "全部" || paper.field.includes(state.paperFilter);
    const term = state.paperSearch.trim().toLowerCase();
    const searchOk =
      !term ||
      [paper.title, paper.authors, paper.field, paper.tags.join(" "), paper.abstract]
        .join(" ")
        .toLowerCase()
        .includes(term);
    return categoryOk && searchOk;
  });
  const paper = currentPaper();
  return `
    <section class="paper-browser">
      <div class="panel">
        <div class="panel-header">
          <div>
            <h2 class="panel-title">文献检索与分类</h2>
            <p class="panel-kicker">展示搜索、标签、详情和关键医学知识提取。</p>
          </div>
        </div>
        <div class="search-bar">
          <input class="input" placeholder="搜索论文、关键词、疾病领域" value="${h(state.paperSearch)}" data-paper-search />
          <button class="icon-button" title="搜索" aria-label="搜索">⌕</button>
        </div>
        <div class="tag-row" style="margin-bottom:12px;">
          ${["全部", "冠脉造影", "AI + CVD", "ICA", "心电图"]
            .map(
              (filter) => `
              <button class="button ${state.paperFilter === filter ? "" : "secondary"}" data-paper-filter="${h(filter)}">${h(filter)}</button>`
            )
            .join("")}
        </div>
        <div class="paper-list">
          ${filtered
            .map(
              (paperItem) => `
              <button class="paper-card ${state.selectedPaperId === paperItem.id ? "active" : ""}" data-paper="${h(paperItem.id)}">
                <h3>${h(paperItem.title)}</h3>
                <p>${h(paperItem.authors)} · ${h(paperItem.year)} · ${h(paperItem.source)}</p>
                <div style="margin-top:10px;">${tags(paperItem.tags, "gray")}</div>
              </button>`
            )
            .join("") || '<div class="empty-state">没有匹配的文献。</div>'}
        </div>
      </div>
      <div class="panel">
        <div class="panel-header">
          <div>
            <h2 class="panel-title">文献详情</h2>
            <p class="panel-kicker">论文内容 → 知识提取 → 医学实体 → 医学关系 → 知识库。</p>
          </div>
          <a class="button secondary" href="${h(paper.href)}" target="_blank" rel="noreferrer">打开来源</a>
        </div>
        <h3 class="detail-title">${h(paper.title)}</h3>
        <p class="detail-meta">${h(paper.authors)} · ${h(paper.year)} · ${h(paper.field)} · ${h(paper.source)}</p>
        <div class="report-section">
          <h3>摘要</h3>
          <p>${h(paper.abstract)}</p>
        </div>
        <div class="report-section">
          <h3>关键医学知识提取</h3>
          ${tags(paper.extraction)}
        </div>
        <div class="report-section">
          <h3>入库结构</h3>
          <div class="constraint-grid">
            ${["医学实体", "医学关系", "证据片段", "检索向量"]
              .map(
                (label, index) => `
                <div class="constraint-item">
                  <strong><span>${h(label)}</span><span>${95 - index}%</span></strong>
                  ${progress(95 - index)}
                </div>`
              )
              .join("")}
          </div>
        </div>
      </div>
    </section>
  `;
}

function renderCases() {
  const item = currentCase();
  return `
    <div class="page-grid">
      <section class="split">
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">病例列表</h2>
              <p class="panel-kicker">所有病例为 Research Demo，字段用于展示病例中心流程。</p>
            </div>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>病例</th><th>疾病</th><th>年龄</th><th>风险</th><th>状态</th><th>操作</th></tr>
              </thead>
              <tbody>
                ${cases
                  .map(
                    (caseItem) => `
                    <tr class="${caseItem.id === item.id ? "selected" : ""}">
                      <td>${h(caseItem.id)}</td>
                      <td>${h(caseItem.disease)}</td>
                      <td>${h(caseItem.age)}</td>
                      <td><span class="tag ${caseItem.risk.includes("高") ? "red" : caseItem.risk.includes("中") ? "amber" : ""}">${h(caseItem.risk)}</span></td>
                      <td>${h(caseItem.status)}</td>
                      <td><button class="button secondary" data-case="${h(caseItem.id)}">查看</button></td>
                    </tr>`
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        </div>
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">病例详情</h2>
              <p class="panel-kicker">${h(item.patient)} / ${h(item.id)}</p>
            </div>
            <button class="button" data-route="diagnosis" data-case="${h(item.id)}">导入诊疗分析</button>
          </div>
          <div class="kv-list">
            <div class="kv"><span>患者信息</span><strong>${h(item.sex)} / ${h(item.age)} 岁 / BMI ${h(item.bmi)}</strong></div>
            <div class="kv"><span>血压与心率</span><strong>${h(item.bloodPressure)} / ${h(item.heartRate)}</strong></div>
            <div class="kv"><span>风险预测</span><strong>${h(item.risk)} · ${h(item.confidence)}%</strong></div>
            <div class="kv"><span>证据一致性</span><strong>${h(item.consistency)}%</strong></div>
          </div>
          <div class="report-section">
            <h3>症状与病史</h3>
            <p>${h(item.symptoms.join("、"))}；${h(item.history)}</p>
          </div>
          <div class="report-section">
            <h3>检查与报告</h3>
            <ol>${item.exams.map((exam) => `<li>${h(exam)}</li>`).join("")}</ol>
          </div>
        </div>
      </section>
      <section class="page-grid grid-3">
        <figure class="image-card">
          <img src="/dataset_video/input/00045.png" alt="本地冠脉造影输入帧" />
          <figcaption><span>本地 XCA 输入帧：dataset_video/input/00045.png</span><a href="/dataset_video/input/00045.png" target="_blank">打开</a></figcaption>
        </figure>
        <figure class="image-card">
          <img src="/dataset_video/Ours/00045.png" alt="本地冠脉造影增强结果帧" />
          <figcaption><span>本地算法结果帧：dataset_video/Ours/00045.png</span><a href="/dataset_video/Ours/00045.png" target="_blank">打开</a></figcaption>
        </figure>
        <figure class="image-card">
          <video src="/merged_video.mp4" controls muted></video>
          <figcaption><span>本地合成视频：merged_video.mp4</span><a href="/merged_video.mp4" target="_blank">打开</a></figcaption>
        </figure>
      </section>
      <section class="page-grid grid-2">
        ${externalSources
          .filter((src) => src.image)
          .map(
            (src) => `
            <figure class="image-card">
              <img src="${h(src.image)}" alt="${h(src.title)}" loading="lazy" />
              <figcaption><span>${h(src.title)} · ${h(src.license)}</span><a href="${h(src.href)}" target="_blank" rel="noreferrer">来源</a></figcaption>
            </figure>`
          )
          .join("")}
      </section>
    </div>
  `;
}

function renderReports() {
  const item = currentCase();
  return `
    <div class="page-grid">
      <section class="split">
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">报告生成输入</h2>
              <p class="panel-kicker">病例、检查结果、影像和结构化约束统一进入生成器。</p>
            </div>
          </div>
          <div class="field"><label>病例摘要</label><textarea class="textarea">${h(item.history)}</textarea></div>
          <div class="field" style="margin-top:12px;"><label>检查结果</label><textarea class="textarea">${h(item.exams.join("\n"))}</textarea></div>
          <div class="upload-zone" style="margin-top:12px;">
            <div><strong>医学影像上传入口</strong><span>前端已预留，当前以 Mock 结果生成。</span></div>
          </div>
          <div class="button-row" style="margin-top:14px;">
            <button class="button" data-action="generate-report">生成报告</button>
            <button class="button secondary" data-route="diagnosis">回到病例分析</button>
          </div>
        </div>
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">标准化医学报告</h2>
              <p class="panel-kicker">九段式结构，绑定证据依据。</p>
            </div>
            <span class="tag ${state.reportGenerated ? "" : "gray"}">${state.reportGenerated ? "Generated" : "Preview"}</span>
          </div>
          ${state.reportGenerated ? renderFullReport(item) : '<div class="empty-state">点击“生成报告”查看标准化医学报告示例。</div>'}
        </div>
      </section>
      <section class="panel">
        <div class="panel-header">
          <div>
            <h2 class="panel-title">结构化生成约束</h2>
            <p class="panel-kicker">生成结果需满足报告完整性、术语规范性、证据覆盖和结论一致性。</p>
          </div>
        </div>
        <div class="constraint-grid">
          ${[
            ["Report Structure", 100],
            ["Medical Terminology", 96],
            ["Evidence Coverage", 94],
            ["Consistency", 97],
          ]
            .map(
              ([label, value]) => `
              <div class="constraint-item">
                <strong><span>${h(label)}</span><span>${h(value)}%</span></strong>
                ${progress(value)}
              </div>`
            )
            .join("")}
        </div>
      </section>
    </div>
  `;
}

function renderFullReport(item) {
  const sections = [
    ["一、患者信息", `${item.sex}，${item.age} 岁，BMI ${item.bmi}，血压 ${item.bloodPressure}，心率 ${item.heartRate}。`],
    ["二、临床表现", item.symptoms.join("、")],
    ["三、检查所见", item.exams.join("；")],
    ["四、关键异常", item.findings.join("；")],
    ["五、医学分析", "综合病史、症状、实验室检查和影像证据，模型提示需重点评估冠脉狭窄及 ASCVD 风险。"],
    ["六、风险评估", `${item.risk}，Confidence ${item.confidence}%。`],
    ["七、诊断结论", item.disease],
    ["八、临床建议", item.recommendation],
    ["九、证据依据", item.evidence.join("；")],
  ];
  return sections
    .map(
      ([title, copy]) => `
      <div class="report-section">
        <h3>${h(title)}</h3>
        <p>${h(copy)}</p>
      </div>`
    )
    .join("");
}

function renderPrediction() {
  const risks = [
    ["冠心病", 88, "High", "red"],
    ["心力衰竭", 46, "Medium", "amber"],
    ["心肌梗死", 34, "Low-Medium", "amber"],
    ["心律失常", 28, "Low", ""],
    ["高血压相关风险", 71, "Medium-High", "amber"],
  ];
  return `
    <div class="page-grid">
      <section class="page-grid grid-3">
        <div class="panel canvas-panel">
          <div class="panel-header"><div><h2 class="panel-title">Risk Radar</h2><p class="panel-kicker">多任务预测向量。</p></div></div>
          <canvas id="riskRadarCanvas" aria-label="风险雷达图"></canvas>
        </div>
        <div class="panel canvas-panel">
          <div class="panel-header"><div><h2 class="panel-title">ECG Waveform</h2><p class="panel-kicker">模拟 MIT-BIH 风格 ECG 片段。</p></div></div>
          <canvas id="ecgCanvas" aria-label="心电图波形"></canvas>
        </div>
        <div class="panel">
          <div class="panel-header"><div><h2 class="panel-title">预测摘要</h2><p class="panel-kicker">Risk Score / Probability / Confidence / Level。</p></div></div>
          <div class="kv-list">
            <div class="kv"><span>模型</span><strong>CVD-LLM-R2 + Task Head</strong></div>
            <div class="kv"><span>输入病例</span><strong>${h(currentCase().id)}</strong></div>
            <div class="kv"><span>总体置信度</span><strong>91.6%</strong></div>
            <div class="kv"><span>证据覆盖</span><strong>94.0%</strong></div>
          </div>
        </div>
      </section>
      <section class="panel">
        <div class="panel-header"><div><h2 class="panel-title">疾病风险预测任务</h2><p class="panel-kicker">当前为 Mock 输出，后续可接入真实预测 API。</p></div></div>
        <div class="page-grid grid-3">
          ${risks
            .map(
              ([label, score, level, variant]) => `
              <div class="metric-card">
                <span class="metric-label">${h(label)}</span>
                <strong class="metric-value">${h(score)}%</strong>
                <span class="tag ${h(variant)}">${h(level)}</span>
                <div style="margin-top:12px;">${progress(score, variant)}</div>
              </div>`
            )
            .join("")}
        </div>
      </section>
    </div>
  `;
}

function renderModels() {
  const stages = [
    ["General LLM", "通用语言理解、推理与指令遵循"],
    ["Medical Knowledge", "医学论文、指南、病例报告增强"],
    ["Domain Adaptation", "LoRA / Adapter 参数高效迁移"],
    ["Task Prediction", "风险预测、检查分析、诊断辅助"],
    ["Structured Generation", "报告段落、证据字段、风险等级"],
    ["Evidence Verification", "证据一致性与冲突检查"],
  ];
  return `
    <div class="page-grid">
      <section class="panel">
        <div class="panel-header">
          <div>
            <h2 class="panel-title">模型架构图</h2>
            <p class="panel-kicker">基础模型 → 医学知识增强 → 任务适配 → 结构化输出 → 证据验证。</p>
          </div>
        </div>
        <div class="model-chain">
          ${stages
            .map(
              ([title, copy], index) => `
              <div class="model-stage">
                <span class="stage-number">Stage ${index + 1}</span>
                <strong>${h(title)}</strong>
                <span>${h(copy)}</span>
              </div>`
            )
            .join("")}
        </div>
      </section>
      <section class="page-grid grid-3">
        ${capabilities.map(([title, copy]) => `<div class="ability-card"><h3>${h(title)}</h3><p>${h(copy)}</p></div>`).join("")}
      </section>
    </div>
  `;
}

function renderTraining() {
  return `
    <div class="page-grid">
      <section class="page-grid grid-3">
        ${[
          ["Stage 1", "通用指令学习", "通用指令问答数据 → 基础模型"],
          ["Stage 2", "医学领域知识迁移", "论文、病例报告、影像规则、临床指南 → 医学领域模型"],
          ["Stage 3", "目标域任务适配", "心血管疾病任务数据 → 诊疗模型"],
        ]
          .map(
            ([stage, title, copy]) => `
            <div class="panel">
              <span class="tag">${h(stage)}</span>
              <h2 class="panel-title" style="margin-top:12px;">${h(title)}</h2>
              <p class="panel-kicker">${h(copy)}</p>
            </div>`
          )
          .join("")}
      </section>
      <section class="page-grid grid-2">
        <div class="panel canvas-panel">
          <div class="panel-header"><div><h2 class="panel-title">Training / Validation Loss</h2><p class="panel-kicker">训练过程 Mock 曲线。</p></div></div>
          <canvas id="trainingLossCanvas" aria-label="训练损失曲线"></canvas>
        </div>
        <div class="panel">
          <div class="panel-header"><div><h2 class="panel-title">评估指标</h2><p class="panel-kicker">Accuracy / F1 / AUC / Evidence Consistency / Report Quality。</p></div></div>
          <div class="constraint-grid">
            ${[
              ["Accuracy", 89],
              ["F1", 87],
              ["AUC", 91],
              ["Evidence Consistency", 95],
              ["Report Quality", 93],
              ["Terminology", 96],
            ]
              .map(
                ([label, value]) => `
                <div class="constraint-item">
                  <strong><span>${h(label)}</span><span>${h(value)}%</span></strong>
                  ${progress(value)}
                </div>`
              )
              .join("")}
          </div>
        </div>
      </section>
    </div>
  `;
}

function renderFusion() {
  const source = sources.find((item) => item.id === state.selectedSourceId) || sources[1];
  return `
    <div class="page-grid">
      <section class="split">
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">多源医学知识融合</h2>
              <p class="panel-kicker">通用知识、医学论文、病例报告、临床指南和影像规则汇入医疗大模型。</p>
            </div>
          </div>
          <div class="fusion-graph">
            <svg viewBox="0 0 900 420" preserveAspectRatio="none">
              <line x1="210" y1="70" x2="460" y2="210" stroke="#9ed7d0" stroke-width="2" />
              <line x1="210" y1="160" x2="460" y2="210" stroke="#9ed7d0" stroke-width="2" />
              <line x1="210" y1="250" x2="460" y2="210" stroke="#9ed7d0" stroke-width="2" />
              <line x1="210" y1="340" x2="460" y2="210" stroke="#9ed7d0" stroke-width="2" />
              <line x1="460" y1="210" x2="700" y2="210" stroke="#e6c275" stroke-width="2" />
            </svg>
            <button class="graph-node clickable" data-source="source-instruction" style="left:7%;top:9%;">通用指令问答</button>
            <button class="graph-node clickable" data-source="source-papers" style="left:7%;top:31%;">医学论文</button>
            <button class="graph-node clickable" data-source="source-cases" style="left:7%;top:53%;">病例报告</button>
            <button class="graph-node clickable" data-source="source-guideline" style="left:7%;top:75%;">临床指南/影像规则</button>
            <div class="graph-node active" style="left:44%;top:43%;">医学知识融合</div>
            <div class="graph-node risk" style="left:72%;top:43%;">医疗大模型</div>
          </div>
        </div>
        <div class="panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">${h(source.title)}</h2>
              <p class="panel-kicker">点击左侧数据源查看入库状态。</p>
            </div>
            <span class="tag">${h(source.status)}</span>
          </div>
          <div class="kv-list">
            <div class="kv"><span>数据数量</span><strong>${h(source.count)}</strong></div>
            <div class="kv"><span>数据质量</span><strong>${h(source.quality)}</strong></div>
            <div class="kv"><span>数据来源</span><strong>Mock Registry</strong></div>
          </div>
          <div class="report-section">
            <h3>数据类型</h3>
            ${tags(source.examples)}
          </div>
        </div>
      </section>
      <section class="page-grid grid-4">
        ${sources
          .map(
            (src) => `
            <button class="source-card" data-source="${h(src.id)}">
              <h3>${h(src.title)}</h3>
              <p>数量 ${h(src.count)} · 质量 ${h(src.quality)} · ${h(src.status)}</p>
            </button>`
          )
          .join("")}
      </section>
    </div>
  `;
}

function renderDataCenter() {
  const datasets = [
    ["通用指令问答", "320K", "JSONL", "Ready"],
    ["医学论文", "4,611", "PDF / HTML", "Indexing"],
    ["病例诊断报告", "18,760", "Text / DICOM link", "De-identified"],
    ["ECG 数据", "48 records", "PhysioNet WFDB", "External"],
    ["冠脉造影图像", "3,000+", "PNG / DICOM", "External"],
  ];
  return `
    <div class="page-grid">
      <section class="panel">
        <div class="panel-header"><div><h2 class="panel-title">数据集监控</h2><p class="panel-kicker">用于后续 FastAPI / Flask 接口接入的数据资源面板。</p></div></div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>数据集</th><th>规模</th><th>格式</th><th>状态</th></tr></thead>
            <tbody>
              ${datasets.map(([name, size, format, status]) => `<tr><td>${h(name)}</td><td>${h(size)}</td><td>${h(format)}</td><td>${h(status)}</td></tr>`).join("")}
            </tbody>
          </table>
        </div>
      </section>
      <section class="page-grid grid-3">
        ${externalSources.map((src) => `<a class="source-card" href="${h(src.href)}" target="_blank" rel="noreferrer"><h3>${h(src.title)}</h3><p>${h(src.source)} · ${h(src.type)}</p></a>`).join("")}
      </section>
    </div>
  `;
}

function renderSettings() {
  return `
    <div class="page-grid">
      <section class="panel">
        <div class="panel-header">
          <div>
            <h2 class="panel-title">后端 API 接入面板</h2>
            <p class="panel-kicker">当前为 Mock API；后续替换 services 层即可接入真实模型服务。</p>
          </div>
          <span class="tag amber">Mock API</span>
        </div>
        <div class="constraint-grid">
          ${[
            ["/api/cases", "病例列表、详情、上传报告"],
            ["/api/analyze", "病例分析模块文本大模型流式输出"],
            ["/api/prediction/risk", "心血管风险预测"],
            ["/api/reports/generate", "结构化医学报告生成"],
            ["/api/evidence/check", "证据一致性分析"],
            ["/api/knowledge/search", "医学知识检索与知识图谱"],
          ]
            .map(
              ([endpoint, desc]) => `
              <div class="constraint-item">
                <strong><span>${h(endpoint)}</span><span>Ready</span></strong>
                <p class="panel-kicker">${h(desc)}</p>
              </div>`
            )
            .join("")}
        </div>
      </section>
      <section class="notice">
        <strong>医学安全设计</strong>
        <span>生产环境需要增加用户权限、审计日志、数据脱敏、医生复核、模型版本追踪和高风险结论阻断策略。</span>
      </section>
    </div>
  `;
}

function renderPage() {
  switch (state.activeRoute) {
    case "diagnosis":
      return renderDiagnosis();
    case "prediction":
      return renderPrediction();
    case "reports":
      return renderReports();
    case "models":
      return renderModels();
    case "training":
      return renderTraining();
    case "fusion":
      return renderFusion();
    case "knowledge":
      return renderKnowledge();
    case "literature":
      return renderLiterature();
    case "cases":
      return renderCases();
    case "evidence":
      return renderEvidence();
    case "data":
      return renderDataCenter();
    case "settings":
      return renderSettings();
    default:
      return renderDashboard();
  }
}

function setRoute(route) {
  state.activeRoute = route;
  history.replaceState(null, "", `#${route}`);
  render();
}

function stopAnalysisTimers() {
  clearInterval(analysisTimer);
  clearInterval(typewriterTimer);
  analysisTimer = null;
  typewriterTimer = null;
}

function runAnalysisWorkflow(runId) {
  return new Promise((resolve) => {
    analysisDebug("workflow:start", {
      runId,
      steps: workflow.length,
    });
    analysisTimer = setInterval(() => {
      if (runId !== state.analysisRunId) {
        clearInterval(analysisTimer);
        resolve();
        return;
      }
      state.analysisIndex += 1;
      if (state.analysisIndex >= workflow.length) {
        state.analyzing = false;
        clearInterval(analysisTimer);
        analysisTimer = null;
        analysisDebug("workflow:complete", { runId });
        resolve();
      }
      render();
    }, 650);
  });
}

function typeAnalysisResult(report, runId) {
  return new Promise((resolve) => {
    clearInterval(typewriterTimer);
    state.analysisDraft = emptyAnalysisReport();
    state.analysisStatus = "typing";
    state.typingField = analysisReportFields[0]?.key || "";
    let fieldIndex = 0;
    let charIndex = 0;

    typewriterTimer = setInterval(() => {
      if (runId !== state.analysisRunId) {
        clearInterval(typewriterTimer);
        resolve();
        return;
      }

      const field = analysisReportFields[fieldIndex];
      if (!field) {
        clearInterval(typewriterTimer);
        typewriterTimer = null;
        state.analysisDraft = { ...report };
        state.analysisStatus = "complete";
        state.typingField = "";
        render();
        resolve();
        return;
      }

      const text = report[field.key] || "";
      state.typingField = field.key;
      state.analysisDraft[field.key] = text.slice(0, charIndex);
      charIndex += 1;

      if (charIndex > text.length + 1) {
        fieldIndex += 1;
        charIndex = 0;
      }

      render();
    }, 22);
  });
}

async function startAnalysis() {
  const payload = collectAnalysisPayload();
  const runId = state.analysisRunId + 1;
  stopAnalysisTimers();
  state.analysisRunId = runId;
  state.analyzing = true;
  state.analysisStatus = "running";
  state.analysisIndex = 0;
  state.selectedAnalysisModel = payload.model;
  state.analysisInputs = { ...payload.inputs };
  state.analysisResult = null;
  state.analysisDraft = emptyAnalysisReport();
  state.analysisError = "";
  state.typingField = "";
  render();

  const requestPromise = requestAnalysis(payload);
  analysisDebug("request:dispatched", {
    runId,
    note: "fetch 已在中间动画启动前派发，模型推理不会等待动画结束。",
  });
  const workflowPromise = runAnalysisWorkflow(runId);
  try {
    const report = await requestPromise;
    analysisDebug("request:report-ready", {
      runId,
      fieldLengths: analysisReportLengths(report),
    });
    await workflowPromise;
    if (runId !== state.analysisRunId) return;
    state.analyzing = false;
    state.analysisIndex = workflow.length;
    state.analysisResult = report;
    await typeAnalysisResult(report, runId);
  } catch (error) {
    if (runId !== state.analysisRunId) return;
    console.error("病例分析 API 调用失败：", error);
    stopAnalysisTimers();
    state.analyzing = false;
    state.analysisStatus = "error";
    state.analysisError = error?.message || "未知错误";
    const friendlyError = state.analysisError.split("：")[0];
    state.analysisDraft = {
      diagnosis: "",
      findings: "",
      analysis: `AI 分析请求失败：${friendlyError}`,
      advice: "请确认 /api/analyze 后端服务已启动、接口路径一致，并查看浏览器控制台中的错误详情。",
    };
    render();
  }
}

function drawCanvasBase(canvas) {
  if (!canvas) return null;
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(320, rect.width * ratio);
  canvas.height = Math.max(180, rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  return { ctx, width: canvas.width / ratio, height: canvas.height / ratio };
}

function drawECG(id) {
  const base = drawCanvasBase(document.getElementById(id));
  if (!base) return;
  const { ctx, width, height } = base;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#e8eeee";
  ctx.lineWidth = 1;
  for (let x = 0; x < width; x += 18) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y < height; y += 18) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  ctx.strokeStyle = "#be4b52";
  ctx.lineWidth = 2;
  ctx.beginPath();
  const mid = height * 0.52;
  for (let x = 0; x < width; x++) {
    const beat = x % 92;
    let y = mid + Math.sin(x * 0.07) * 3;
    if (beat > 18 && beat < 24) y -= (beat - 18) * 2.4;
    if (beat >= 24 && beat < 29) y += (beat - 24) * 9.8 - 18;
    if (beat >= 29 && beat < 34) y -= (34 - beat) * 7.4 - 6;
    if (beat > 50 && beat < 70) y -= Math.sin((beat - 50) / 20 * Math.PI) * 12;
    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#61717a";
  ctx.font = "12px Segoe UI";
  ctx.fillText("Mock ECG waveform · inspired by public ECG benchmark patterns", 12, height - 12);
}

function drawLineChart(id) {
  const base = drawCanvasBase(document.getElementById(id));
  if (!base) return;
  const { ctx, width, height } = base;
  const train = [0.92, 0.78, 0.63, 0.51, 0.43, 0.37, 0.33, 0.3, 0.28];
  const valid = [0.98, 0.85, 0.72, 0.62, 0.56, 0.51, 0.48, 0.46, 0.45];
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#dce7e6";
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i++) {
    const y = 20 + (height - 44) * (i / 4);
    ctx.beginPath();
    ctx.moveTo(36, y);
    ctx.lineTo(width - 16, y);
    ctx.stroke();
  }
  function plot(series, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    series.forEach((value, index) => {
      const x = 36 + (width - 58) * (index / (series.length - 1));
      const y = 20 + (height - 54) * value;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  plot(train, "#0b8f86");
  plot(valid, "#5367a6");
  ctx.fillStyle = "#14242b";
  ctx.font = "12px Segoe UI";
  ctx.fillText("Training Loss", 42, height - 15);
  ctx.fillStyle = "#5367a6";
  ctx.fillText("Validation Loss", 146, height - 15);
}

function drawRadar(id) {
  const base = drawCanvasBase(document.getElementById(id));
  if (!base) return;
  const { ctx, width, height } = base;
  const labels = ["CAD", "HF", "MI", "Arr", "HTN"];
  const values = [0.88, 0.46, 0.34, 0.28, 0.71];
  const cx = width / 2;
  const cy = height / 2 + 8;
  const radius = Math.min(width, height) * 0.34;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#dce7e6";
  ctx.lineWidth = 1;
  for (let ring = 1; ring <= 4; ring++) {
    ctx.beginPath();
    labels.forEach((_, index) => {
      const angle = -Math.PI / 2 + (index * Math.PI * 2) / labels.length;
      const r = (radius * ring) / 4;
      const x = cx + Math.cos(angle) * r;
      const y = cy + Math.sin(angle) * r;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
  }
  ctx.beginPath();
  values.forEach((value, index) => {
    const angle = -Math.PI / 2 + (index * Math.PI * 2) / labels.length;
    const x = cx + Math.cos(angle) * radius * value;
    const y = cy + Math.sin(angle) * radius * value;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fillStyle = "rgba(11, 143, 134, 0.18)";
  ctx.fill();
  ctx.strokeStyle = "#0b8f86";
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.fillStyle = "#61717a";
  ctx.font = "12px Segoe UI";
  labels.forEach((label, index) => {
    const angle = -Math.PI / 2 + (index * Math.PI * 2) / labels.length;
    const x = cx + Math.cos(angle) * (radius + 20);
    const y = cy + Math.sin(angle) * (radius + 20);
    ctx.fillText(label, x - 12, y + 4);
  });
}

function afterRender() {
  requestAnimationFrame(() => {
    drawECG("ecgCanvas");
    drawLineChart("trainingLossCanvas");
    drawRadar("riskRadarCanvas");
  });
}

function render() {
  document.getElementById("app").innerHTML = renderLayout();
  afterRender();
}

document.addEventListener("click", (event) => {
  const routeButton = event.target.closest("[data-route]");
  if (routeButton) {
    const caseId = routeButton.getAttribute("data-case");
    const sourceId = routeButton.getAttribute("data-source");
    if (caseId && caseId !== state.selectedCaseId) {
      stopAnalysisTimers();
      state.analysisRunId += 1;
      state.selectedCaseId = caseId;
      state.analyzing = false;
      state.analysisInputs = null;
      state.analysisStatus = "idle";
      state.analysisDraft = emptyAnalysisReport();
      state.analysisIndex = -1;
    }
    if (sourceId) state.selectedSourceId = sourceId;
    setRoute(routeButton.getAttribute("data-route"));
    return;
  }

  const action = event.target.closest("[data-action]")?.getAttribute("data-action");
  if (action === "start-analysis") {
    startAnalysis();
    return;
  }
  if (action === "generate-report") {
    state.reportGenerated = true;
    render();
    return;
  }
  if (action === "mock-upload") {
    state.uploadState = "已接收 demo_angio_frame.png";
    render();
    return;
  }

  const caseButton = event.target.closest("[data-case]");
  if (caseButton) {
    const nextCaseId = caseButton.getAttribute("data-case");
    if (nextCaseId !== state.selectedCaseId) {
      stopAnalysisTimers();
      state.analysisRunId += 1;
      state.selectedCaseId = nextCaseId;
      state.analyzing = false;
      state.analysisInputs = null;
      state.analysisStatus = "idle";
      state.analysisDraft = emptyAnalysisReport();
      state.analysisIndex = -1;
    }
    render();
    return;
  }

  const paperButton = event.target.closest("[data-paper]");
  if (paperButton) {
    state.selectedPaperId = paperButton.getAttribute("data-paper");
    render();
    return;
  }

  const nodeButton = event.target.closest("[data-node]");
  if (nodeButton) {
    state.selectedKnowledgeNode = nodeButton.getAttribute("data-node");
    render();
    return;
  }

  const sourceButton = event.target.closest("[data-source]");
  if (sourceButton) {
    state.selectedSourceId = sourceButton.getAttribute("data-source");
    render();
  }
});

document.addEventListener("input", (event) => {
  if (event.target.matches("[data-analysis-input], [data-analysis-meta]")) {
    const key =
      event.target.getAttribute("data-analysis-input") ||
      event.target.getAttribute("data-analysis-meta");
    state.analysisInputs = {
      ...defaultAnalysisInputs(currentCase()),
      ...(state.analysisInputs || {}),
      [key]: event.target.value,
    };
  }

  if (event.target.matches("[data-paper-search]")) {
    const cursor = event.target.selectionStart || event.target.value.length;
    state.paperSearch = event.target.value;
    render();
    const input = document.querySelector("[data-paper-search]");
    if (input) {
      input.focus();
      input.setSelectionRange(cursor, cursor);
    }
  }
});

document.addEventListener("change", (event) => {
  if (event.target.matches("[data-analysis-model]")) {
    state.selectedAnalysisModel = event.target.value;
    render();
    return;
  }

  if (event.target.matches('[data-analysis-meta="sex"]')) {
    state.analysisInputs = {
      ...defaultAnalysisInputs(currentCase()),
      ...(state.analysisInputs || {}),
      sex: event.target.value,
    };
    return;
  }

  if (event.target.matches("[data-symptom-checkbox]")) {
    const symptoms = Array.from(document.querySelectorAll("[data-symptom-checkbox]:checked"))
      .map((input) => input.getAttribute("data-symptom-checkbox"))
      .filter(Boolean)
      .join("、");
    const textarea = document.querySelector('[data-analysis-input="symptoms"]');
    if (textarea) textarea.value = symptoms;
    state.analysisInputs = {
      ...defaultAnalysisInputs(currentCase()),
      ...(state.analysisInputs || {}),
      symptoms,
    };
  }
});

document.addEventListener("click", (event) => {
  const filter = event.target.closest("[data-paper-filter]");
  if (filter) {
    state.paperFilter = filter.getAttribute("data-paper-filter");
    render();
  }
});

window.addEventListener("hashchange", () => {
  const next = location.hash.replace("#", "") || "dashboard";
  if (pageMeta[next]) {
    state.activeRoute = next;
    render();
  }
});

window.addEventListener("resize", () => afterRender());

render();

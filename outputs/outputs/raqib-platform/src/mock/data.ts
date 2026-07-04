import { L, type ModelInfo, type RagMessage, type RiskFactor, type Severity } from "@/types";

// ── KPIs (with sparkline trends) ──────────────────────────────
export const kpis = {
  active: { value: 23, delta: 12, trend: [9, 12, 10, 15, 14, 19, 23] },
  pending: { value: 6, delta: -8, trend: [11, 10, 9, 8, 9, 7, 6] },
  responseMins: { value: 14, delta: -5, trend: [22, 20, 19, 18, 17, 15, 14] },
  highRisk: { value: 9, delta: 4, trend: [5, 6, 6, 7, 8, 8, 9] },
};

// ── Charts ────────────────────────────────────────────────────
export const trend7 = [
  { day: "السبت", dayEn: "Sat", reports: 14, dispatched: 11 },
  { day: "الأحد", dayEn: "Sun", reports: 19, dispatched: 15 },
  { day: "الإثنين", dayEn: "Mon", reports: 12, dispatched: 10 },
  { day: "الثلاثاء", dayEn: "Tue", reports: 22, dispatched: 18 },
  { day: "الأربعاء", dayEn: "Wed", reports: 17, dispatched: 14 },
  { day: "الخميس", dayEn: "Thu", reports: 25, dispatched: 20 },
  { day: "الجمعة", dayEn: "Fri", reports: 23, dispatched: 19 },
];

export const byType = [
  { key: "pothole", ar: "حُفر وتلف", en: "Potholes", value: 41, color: "#F07316" },
  { key: "sign_defect", ar: "عيوب إشارات", en: "Sign defects", value: 33, color: "#2F6FED" },
  { key: "accident", ar: "حوادث", en: "Accidents", value: 26, color: "#DC2A28" },
];

export const bySeverity: { level: Severity; ar: string; en: string; value: number; color: string }[] = [
  { level: "low", ar: "منخفضة", en: "Low", value: 28, color: "#1A9E54" },
  { level: "medium", ar: "متوسطة", en: "Medium", value: 39, color: "#E0A008" },
  { level: "high", ar: "عالية", en: "High", value: 22, color: "#F07316" },
  { level: "critical", ar: "حرجة", en: "Critical", value: 11, color: "#DC2A28" },
];

// ── Model registry (placeholders — wire endpoints via .env) ───
export const models: ModelInfo[] = [
  {
    key: "sign_defect",
    name: L("كشف عيوب إشارات المرور", "Traffic Sign Defect Detection"),
    family: "cv",
    task: L("كشف وتصنيف الإشارات التالفة/الساقطة/الباهتة", "Detect & classify damaged / fallen / faded signs"),
    status: "not_connected",
    version: "v0.1 · placeholder",
    accuracy: 92,
    metricLabel: L("mAP@0.5 (تقديري)", "mAP@0.5 (target)"),
    latencyMs: 38,
    throughput: L("~26 إطار/ث", "~26 fps"),
    lastRun: "—",
    endpointEnv: "VITE_MODEL_SIGN_DEFECT_URL",
    description: L(
      "موديل رؤية حاسوبية يكتشف إشارات المرور ويصنّف حالتها (سليمة، باهتة، مائلة، ساقطة، محجوبة).",
      "A CV model that detects traffic signs and classifies condition (intact, faded, tilted, fallen, obscured)."
    ),
  },
  {
    key: "pothole",
    name: L("كشف الحُفر وتلف السطح", "Pothole & Surface Damage Detection"),
    family: "cv",
    task: L("تحديد الحُفر والتشققات وتقدير شدتها", "Localize potholes & cracks and estimate severity"),
    status: "not_connected",
    version: "v0.1 · placeholder",
    accuracy: 89,
    metricLabel: L("mAP@0.5 (تقديري)", "mAP@0.5 (target)"),
    latencyMs: 44,
    throughput: L("~22 إطار/ث", "~22 fps"),
    lastRun: "—",
    endpointEnv: "VITE_MODEL_POTHOLE_URL",
    description: L(
      "يكشف الحُفر وتلف الإسفلت ويقدّر المساحة/الشدة لترتيب أولوية الصيانة.",
      "Detects potholes and asphalt damage and estimates area/severity to prioritize maintenance."
    ),
  },
  {
    key: "accident",
    name: L("كشف الحوادث المرورية", "Accident Detection"),
    family: "cv",
    task: L("رصد التصادمات والمركبات المتوقفة لحظيًا", "Spot collisions & stalled vehicles in real time"),
    status: "not_connected",
    version: "v0.1 · placeholder",
    accuracy: 87,
    metricLabel: L("F1 (تقديري)", "F1 (target)"),
    latencyMs: 51,
    throughput: L("~18 إطار/ث", "~18 fps"),
    lastRun: "—",
    endpointEnv: "VITE_MODEL_ACCIDENT_URL",
    description: L(
      "يحلّل الفيديو لرصد الحوادث والمركبات المتوقفة وإطلاق بلاغ فوري للجهات.",
      "Analyzes video to spot accidents and stalled vehicles and trigger an immediate alert."
    ),
  },
  {
    key: "risk",
    name: L("التنبؤ بخطورة الطريق", "Road Risk Prediction"),
    family: "ml",
    task: L("تقدير خطورة المقطع من الظروف البيئية والزمنية", "Estimate segment risk from environmental & temporal factors"),
    status: "not_connected",
    version: "v0.1 · placeholder",
    accuracy: 84,
    metricLabel: L("ROC-AUC (تقديري)", "ROC-AUC (target)"),
    latencyMs: 12,
    throughput: L("دفعات فورية", "instant batches"),
    lastRun: "—",
    endpointEnv: "VITE_MODEL_RISK_URL",
    description: L(
      "نموذج تعلُّم آلي يقدّر احتمال الخطورة وفق الإضاءة والطقس والوقت وكثافة المرور وحالة السطح.",
      "An ML model estimating risk likelihood from lighting, weather, time, traffic density and surface state."
    ),
  },
];

// ── Risk segments under watch ─────────────────────────────────
export const riskSegments = [
  { id: "SEG-114", name: L("طريق الملك فهد · شمال", "King Fahd Rd · North"), score: 81, level: "high" as Severity, trend: [62, 65, 70, 74, 78, 81] },
  { id: "SEG-207", name: L("الدائري الشرقي · مخرج 9", "Eastern Ring · Exit 9"), score: 68, level: "high" as Severity, trend: [55, 58, 60, 63, 66, 68] },
  { id: "SEG-051", name: L("طريق مكة السريع", "Makkah Expressway"), score: 47, level: "medium" as Severity, trend: [52, 50, 49, 48, 47, 47] },
  { id: "SEG-318", name: L("دائري أبها الجنوبي", "Abha Southern Ring"), score: 89, level: "critical" as Severity, trend: [70, 74, 79, 83, 86, 89] },
  { id: "SEG-076", name: L("طريق المدينة · جدة", "Madinah Rd · Jeddah"), score: 33, level: "low" as Severity, trend: [40, 38, 36, 35, 34, 33] },
];

// ── RAG canned answers (replace with real /rag/query) ─────────
const fallbackSources = [
  {
    refId: "RQ-2026-08842",
    title: L("بلاغ حفرة — طريق الملك فهد", "Pothole report — King Fahd Rd"),
    kind: "report" as const,
    snippet: L("حفرة بعمق 18سم في المسار الأيمن، خطورة عالية.", "18cm pothole in the right lane, high severity."),
  },
  {
    refId: "REG-TR-114",
    title: L("لائحة السلامة المرورية — مادة 114", "Road Safety Regulation — Article 114"),
    kind: "regulation" as const,
    snippet: L("تُستبدل الإشارة التالفة خلال 72 ساعة من الرصد.", "A damaged sign must be replaced within 72h of detection."),
  },
];

export function ragAnswer(query: string): { content: { ar: string; en: string }; sources: RagMessage["sources"] } {
  const q = query.toLowerCase();
  if (q.includes("حفرة") || q.includes("pothole") || q.includes("الملك فهد") || q.includes("king fahd")) {
    return {
      content: L(
        "هذا الأسبوع سُجِّلت 7 بلاغات حُفر على طريق الملك فهد، منها بلاغان بخطورة عالية (أبرزها RQ-2026-08842 بعمق 18سم). متوسط زمن الإرسال للجهة كان 14 دقيقة. أوصي بأولوية صيانة للمسار الأيمن شمالًا.",
        "This week 7 pothole reports were logged on King Fahd Rd, two of them high-severity (notably RQ-2026-08842, 18cm deep). Average dispatch time was 14 minutes. I'd prioritize maintenance for the northbound right lane."
      ),
      sources: fallbackSources,
    };
  }
  if (q.includes("قف") || q.includes("stop") || q.includes("إشارة") || q.includes("sign")) {
    return {
      content: L(
        "وفق المادة 114 من لائحة السلامة المرورية، تُستبدل إشارة قف التالفة خلال 72 ساعة من رصدها، ومؤقتًا تُركّب إشارة بديلة عاكسة. يوجد حاليًا بلاغ مفتوح RQ-2026-08825 لإشارة ساقطة على طريق العليا.",
        "Per Article 114 of the road-safety regulation, a damaged stop sign must be replaced within 72h of detection, with a temporary reflective sign installed meanwhile. There is an open report RQ-2026-08825 for a fallen sign on Olaya St."
      ),
      sources: fallbackSources,
    };
  }
  if (q.includes("ليل") || q.includes("night") || q.includes("حادث") || q.includes("accident")) {
    return {
      content: L(
        "أكثر المقاطع تكرارًا للحوادث ليلًا: الدائري الشرقي مخرج 9، وطريق مكة السريع. يرتبط ذلك بضعف الإضاءة وارتفاع السرعة. نموذج التنبؤ يضع SEG-207 عند خطورة 68/100.",
        "The segments with the most night-time accidents: Eastern Ring Exit 9 and the Makkah Expressway, linked to poor lighting and high speeds. The prediction model places SEG-207 at 68/100 risk."
      ),
      sources: fallbackSources,
    };
  }
  return {
    content: L(
      "بحثت في أرشيف البلاغات والأنظمة ومعايير السلامة. اطرح سؤالًا أكثر تحديدًا (طريق، نوع خطر، فترة زمنية) وسأرفق المصادر ذات الصلة. هذا رد توضيحي يُستبدل بمخرجات نظام RAG بعد ربطه.",
      "I searched the reports archive, regulations and safety standards. Ask something more specific (road, hazard type, time window) and I'll attach the relevant sources. This is an illustrative answer to be replaced by the RAG system once connected."
    ),
    sources: fallbackSources,
  };
}

export const featureImportanceBase: RiskFactor[] = [
  { name: L("الإضاءة", "Lighting"), weight: 0.3 },
  { name: L("الطقس", "Weather"), weight: 0.24 },
  { name: L("كثافة المرور", "Traffic"), weight: 0.2 },
  { name: L("وقت اليوم", "Time of day"), weight: 0.16 },
  { name: L("حالة السطح", "Surface"), weight: 0.1 },
];

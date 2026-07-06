import { L, type Authority, type HazardReport, type TimelineEvent } from "@/types";

const ago = (mins: number) => new Date(Date.now() - mins * 60000).toISOString();

export const authorities: Record<string, Authority> = {
  riyadhRoads: { id: "AUTH-RUH-01", name: L("إدارة صيانة الطرق — الرياض", "Road Maintenance — Riyadh"), region: L("منطقة الرياض", "Riyadh Region") },
  easternTraffic: { id: "AUTH-EP-02", name: L("مرور المنطقة الشرقية", "Eastern Region Traffic"), region: L("المنطقة الشرقية", "Eastern Region") },
  jeddahRoads: { id: "AUTH-JED-03", name: L("أمانة جدة — قطاع الطرق", "Jeddah Municipality — Roads"), region: L("منطقة مكة المكرمة", "Makkah Region") },
  civilDefense: { id: "AUTH-CD-04", name: L("الدفاع المدني — الطوارئ", "Civil Defense — Emergency"), region: L("وطني", "National") },
  roadSafety: { id: "AUTH-RS-05", name: L("الإدارة العامة للسلامة المرورية", "General Dept. of Road Safety"), region: L("وطني", "National") },
};

function tl(
  kind: TimelineEvent["kind"],
  mins: number,
  ar: string,
  en: string
): TimelineEvent {
  return { kind, at: ago(mins), label: L(ar, en) };
}

export const mockReports: HazardReport[] = [
  {
    id: "RQ-2026-08842",
    type: "pothole",
    severity: "high",
    status: "pending_dispatch",
    source: "manual",
    title: L("حفرة عميقة في المسار الأيمن", "Deep pothole in right lane"),
    description: L(
      "رصدت الرؤية الحاسوبية حفرة بعمق تقديري 18سم في المسار الأيمن من طريق الملك فهد باتجاه الشمال، تمثل خطرًا مباشرًا على المركبات السريعة. يُنصح بإغلاق المسار مؤقتًا وإصلاح فوري.",
      "Computer vision detected an ~18cm-deep pothole in the right northbound lane of King Fahd Rd, a direct hazard to fast-moving vehicles. Temporary lane closure and immediate repair advised."
    ),
    modelKey: "pothole",
    confidence: 0.94,
    createdAt: ago(12),
    location: { x: 53, y: 45, city: L("الرياض", "Riyadh"), road: L("طريق الملك فهد", "King Fahd Rd") },
    authority: authorities.riyadhRoads,
    detections: [{ label: L("حفرة", "Pothole"), confidence: 0.94, x: 0.36, y: 0.52, w: 0.26, h: 0.22 }],
    mediaKind: "image",
    timeline: [
      tl("capture", 13, "تم رفع اللقطة", "Capture uploaded"),
      tl("analysis", 12, "تحليل الموديل (كشف الحُفر)", "Model analysis (pothole)"),
      tl("report", 12, "تم توليد التقرير", "Report generated"),
    ],
  },
  {
    id: "RQ-2026-08839",
    type: "accident",
    severity: "critical",
    status: "dispatched",
    source: "field_camera",
    title: L("حادث تصادم على الطريق الدائري الشرقي", "Collision on Eastern Ring Rd"),
    description: L(
      "كشف النظام حادث تصادم بين مركبتين مع انسداد جزئي للمسار الأوسط. أُرسل بلاغ فوري للدفاع المدني والمرور مع تقدير شدة الحادث كحرج.",
      "A two-vehicle collision with partial blockage of the middle lane was detected. An immediate alert was dispatched to Civil Defense and Traffic with severity assessed as critical."
    ),
    modelKey: "accident",
    confidence: 0.89,
    createdAt: ago(28),
    location: { x: 55, y: 47, city: L("الرياض", "Riyadh"), road: L("الطريق الدائري الشرقي", "Eastern Ring Rd") },
    authority: authorities.civilDefense,
    detections: [
      { label: L("مركبة", "Vehicle"), confidence: 0.91, x: 0.3, y: 0.42, w: 0.22, h: 0.3 },
      { label: L("مركبة", "Vehicle"), confidence: 0.86, x: 0.55, y: 0.48, w: 0.2, h: 0.26 },
    ],
    mediaKind: "video",
    timeline: [
      tl("capture", 29, "التقاط من الكاميرا الميدانية", "Captured by field camera"),
      tl("analysis", 28, "تحليل الموديل (كشف الحوادث)", "Model analysis (accident)"),
      tl("report", 28, "تم توليد التقرير", "Report generated"),
      tl("dispatch", 27, "أُرسل للدفاع المدني والمرور", "Dispatched to Civil Defense & Traffic"),
    ],
  },
  {
    id: "RQ-2026-08835",
    type: "sign_defect",
    severity: "medium",
    status: "acknowledged",
    source: "citizen",
    title: L("إشارة قف باهتة وغير واضحة", "Faded, unclear stop sign"),
    description: L(
      "بلاغ مواطن يُظهر إشارة قف باهتة الألوان ومائلة عند تقاطع طريق المدينة، ما يقلل وضوحها ليلًا. صُنّف كعيب متوسط يستوجب الاستبدال.",
      "A citizen report shows a faded, tilted stop sign at the Madinah Rd intersection, reducing night visibility. Classified as a medium defect requiring replacement."
    ),
    modelKey: "sign_defect",
    confidence: 0.91,
    createdAt: ago(95),
    location: { x: 22, y: 50, city: L("جدة", "Jeddah"), road: L("طريق المدينة", "Madinah Rd") },
    authority: authorities.jeddahRoads,
    detections: [{ label: L("إشارة قف", "Stop sign"), confidence: 0.91, x: 0.58, y: 0.2, w: 0.2, h: 0.3 }],
    mediaKind: "image",
    timeline: [
      tl("capture", 100, "بلاغ مواطن", "Citizen submission"),
      tl("analysis", 99, "تحليل الموديل (عيوب الإشارات)", "Model analysis (sign defect)"),
      tl("report", 99, "تم توليد التقرير", "Report generated"),
      tl("dispatch", 92, "أُرسل لأمانة جدة", "Dispatched to Jeddah Municipality"),
      tl("ack", 70, "استلمت الجهة البلاغ", "Authority acknowledged"),
    ],
  },
  {
    id: "RQ-2026-08830",
    type: "pothole",
    severity: "medium",
    status: "new",
    source: "manual",
    title: L("تلف سطح الطريق قرب المخرج 12", "Surface damage near Exit 12"),
    description: L(
      "تشققات وتلف في سطح الإسفلت على طريق الملك سعود قرب المخرج 12. يُوصى بجدولة صيانة وقائية.",
      "Asphalt cracking and surface damage on King Saud Rd near Exit 12. Preventive maintenance scheduling recommended."
    ),
    modelKey: "pothole",
    confidence: 0.87,
    createdAt: ago(140),
    location: { x: 67, y: 39, city: L("الدمام", "Dammam"), road: L("طريق الملك سعود", "King Saud Rd") },
    authority: authorities.easternTraffic,
    detections: [{ label: L("تلف سطح", "Surface damage"), confidence: 0.87, x: 0.28, y: 0.55, w: 0.4, h: 0.2 }],
    mediaKind: "image",
    timeline: [
      tl("capture", 141, "تم رفع اللقطة", "Capture uploaded"),
      tl("analysis", 140, "تحليل الموديل", "Model analysis"),
      tl("report", 140, "تم توليد التقرير", "Report generated"),
    ],
  },
  {
    id: "RQ-2026-08825",
    type: "sign_defect",
    severity: "high",
    status: "pending_dispatch",
    source: "manual",
    title: L("إشارة تحذير ساقطة على طريق العليا", "Fallen warning sign on Olaya St"),
    description: L(
      "إشارة تحذير منعطف ساقطة بالكامل عن قاعدتها على طريق العليا. غياب التحذير يرفع خطورة المنعطف، صُنّف كعيب عالي الأولوية.",
      "A curve-warning sign has fully fallen from its base on Olaya St. The missing warning raises curve risk; classified high priority."
    ),
    modelKey: "sign_defect",
    confidence: 0.96,
    createdAt: ago(180),
    location: { x: 52, y: 44, city: L("الرياض", "Riyadh"), road: L("طريق العليا", "Olaya St") },
    authority: authorities.roadSafety,
    detections: [{ label: L("إشارة ساقطة", "Fallen sign"), confidence: 0.96, x: 0.2, y: 0.6, w: 0.3, h: 0.22 }],
    mediaKind: "image",
    timeline: [
      tl("capture", 181, "تم رفع اللقطة", "Capture uploaded"),
      tl("analysis", 180, "تحليل الموديل", "Model analysis"),
      tl("report", 180, "تم توليد التقرير", "Report generated"),
    ],
  },
  {
    id: "RQ-2026-08820",
    type: "accident",
    severity: "high",
    status: "resolved",
    source: "field_camera",
    title: L("مركبة متوقفة على كتف الطريق السريع", "Stalled vehicle on highway shoulder"),
    description: L(
      "مركبة متوقفة جزئيًا داخل المسار على طريق مكة السريع. عولج البلاغ وأُزيلت المركبة.",
      "A vehicle partly stalled within the lane on the Makkah highway. Report handled and the vehicle removed."
    ),
    modelKey: "accident",
    confidence: 0.85,
    createdAt: ago(360),
    location: { x: 25, y: 53, city: L("مكة المكرمة", "Makkah"), road: L("طريق مكة السريع", "Makkah Expressway") },
    authority: authorities.civilDefense,
    detections: [{ label: L("مركبة متوقفة", "Stalled vehicle"), confidence: 0.85, x: 0.4, y: 0.45, w: 0.24, h: 0.28 }],
    mediaKind: "video",
    timeline: [
      tl("capture", 361, "التقاط من الكاميرا", "Captured by camera"),
      tl("analysis", 360, "تحليل الموديل", "Model analysis"),
      tl("dispatch", 358, "أُرسل للجهة", "Dispatched"),
      tl("resolve", 300, "تمت المعالجة", "Resolved"),
    ],
  },
  {
    id: "RQ-2026-08815",
    type: "pothole",
    severity: "critical",
    status: "dispatched",
    source: "citizen",
    title: L("انهيار جزئي في سطح طريق دائري أبها", "Partial collapse on Abha Ring Rd"),
    description: L(
      "بلاغ مواطن يُظهر انهيارًا جزئيًا لسطح الطريق بعرض يقارب المتر على الدائري الجنوبي بأبها، خطر حرج على المركبات.",
      "A citizen report shows a near 1m-wide partial road-surface collapse on Abha's southern ring road — a critical vehicle hazard."
    ),
    modelKey: "pothole",
    confidence: 0.92,
    createdAt: ago(75),
    location: { x: 31, y: 65, city: L("أبها", "Abha"), road: L("الطريق الدائري الجنوبي", "Southern Ring Rd") },
    authority: authorities.roadSafety,
    detections: [{ label: L("انهيار سطح", "Surface collapse"), confidence: 0.92, x: 0.3, y: 0.5, w: 0.36, h: 0.3 }],
    mediaKind: "image",
    timeline: [
      tl("capture", 76, "بلاغ مواطن", "Citizen submission"),
      tl("analysis", 75, "تحليل الموديل", "Model analysis"),
      tl("report", 75, "تم توليد التقرير", "Report generated"),
      tl("dispatch", 73, "أُرسل للسلامة المرورية", "Dispatched to Road Safety"),
    ],
  },
  {
    id: "RQ-2026-08807",
    type: "accident",
    severity: "medium",
    status: "acknowledged",
    source: "field_camera",
    title: L("احتكاك مروري خفيف على طريق المدينة", "Minor traffic incident on Madinah Rd"),
    description: L(
      "احتكاك خفيف بين مركبتين دون إصابات ظاهرة، مع بطء حركة. أُبلغت الجهة وجارٍ المتابعة.",
      "A minor two-vehicle incident with no visible injuries and slowed traffic. Authority notified and following up."
    ),
    modelKey: "accident",
    confidence: 0.83,
    createdAt: ago(220),
    location: { x: 28, y: 40, city: L("المدينة المنورة", "Madinah"), road: L("الطريق الدائري", "Ring Rd") },
    authority: authorities.civilDefense,
    detections: [{ label: L("مركبة", "Vehicle"), confidence: 0.83, x: 0.42, y: 0.4, w: 0.22, h: 0.26 }],
    mediaKind: "video",
    timeline: [
      tl("capture", 221, "التقاط من الكاميرا", "Captured by camera"),
      tl("analysis", 220, "تحليل الموديل", "Model analysis"),
      tl("dispatch", 218, "أُرسل للجهة", "Dispatched"),
      tl("ack", 200, "استلمت الجهة", "Acknowledged"),
    ],
  },
];

export const mockReportById = (id: string) => mockReports.find((r) => r.id === id);

# رقيب · Raqib

**منصة رصد مخاطر الطرق** — واجهة احترافية جاهزة للإطلاق. كل مكان للموديل هو **تمبلت فاضي** بدون أي بيانات وهمية؛ أول ما تجهّز الموديلز، تربط روابط الـ API من ملف `.env` والمنصة تشتغل live على طول.

> A production-ready frontend skeleton. Every model slot is an empty template (no fabricated data). When your models are ready, set the API URLs in `.env` and the whole service goes live.

---

## الفكرة

- **بوابة الجهة المختصة** (`/app`): الكشف الذكي (رفع صورة/فيديو → موديلز الرؤية الحاسوبية)، البلاغات الواردة، والتنبؤ بخطورة الطريق.
- **بوابة المواطن** (`/citizen`): بلّغ عن خطر (رفع + موقع)، وافحص خطورة طريق.
- موديلز الـ CV الثلاثة (إشارات / حُفر / حوادث) في **مكان واحد** (صفحة الكشف الذكي).
- موديل الـ ML (خطورة الطريق) **تمبلت فاضي** — الخصائص قيد التحديد — وموجود للجهة وللمواطن.

كل الشاشات تعرض حالات فاضية نظيفة الآن، وتتحوّل لبيانات حقيقية تلقائيًا بمجرد ربط الـ API.

---

## التشغيل

```bash
npm install
cp .env.example .env     # ويندوز: copy .env.example .env
npm run dev              # http://localhost:5173
```

البناء: `npm run build` ثم `npm run preview`.

---

## الربط والإطلاق (أهم جزء)

1. في `.env` حوّل `VITE_USE_MOCK=false` وحط `VITE_API_BASE_URL`.
2. حط رابط كل موديل في متغيّره:

```env
VITE_API_BASE_URL=https://api.your-domain.com
VITE_USE_MOCK=false

VITE_MODEL_SIGN_DEFECT_URL=/models/sign-defect/infer
VITE_MODEL_POTHOLE_URL=/models/pothole/infer
VITE_MODEL_ACCIDENT_URL=/models/accident/infer
VITE_MODEL_RISK_URL=/models/risk/predict
```

3. خلاص — الكشف الذكي يبدأ يستدعي موديلزكم، والبلاغات والأرقام تتحسب من البيانات الحقيقية.

كل نداءات الـ API في مكان واحد: مجلد `src/api/` + `src/api/endpoints.ts`. عقود البيانات (شكل الـ JSON المتوقع) في `src/types/index.ts`.

| الخدمة | الطريقة | المسار |
|---|---|---|
| قائمة البلاغات | GET | `/reports` |
| كشف CV (رفع لقطة) | POST | `VITE_MODEL_*_URL` (multipart، الملف باسم `file`) |
| التنبؤ بالخطورة | POST | `VITE_MODEL_RISK_URL` |
| بلاغ مواطن | POST | `/citizen/reports` |

شكل رد موديل الـ CV المتوقع:

```ts
interface AnalyzeResult {
  type: "sign_defect" | "pothole" | "accident" | "other";
  severity: "low" | "medium" | "high" | "critical";
  confidence: number;             // 0..1
  detections: { label: {ar:string;en:string}; confidence:number; x:number;y:number;w:number;h:number }[];
  description: { ar: string; en: string };
}
```

---

## الهيكل

```
src/
├── api/         ← مكان ربط الـ API (client, endpoints, reports, models, risk, citizen)
├── components/  ← الواجهة المشتركة (Sidebar, Topbar, Layouts, Logo, UI)
├── i18n/        ← عربي/إنجليزي + RTL/LTR
├── pages/
│   ├── ministry/  Analyze · Reports · Risk
│   └── citizen/   CitizenHome · CitizenReport · CitizenRisk
├── types/       ← عقود البيانات
└── lib/         ← أدوات + ألوان الخطورة
```

---

## اللغة والثيم

العربية افتراضية (RTL) وزر التبديل للإنجليزية في الأعلى. اللون الأساسي والخطوط في `tailwind.config.js` و`src/index.css` — تغيّرهم من مكان واحد.

© 2026 رقيب — واجهة جاهزة لربط موديلزكم وإطلاق الخدمة.

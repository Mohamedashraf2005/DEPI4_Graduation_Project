import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import {
  ArrowLeft,
  Building2,
  CheckCircle2,
  ChevronLeft,
  ClipboardList,
  HelpCircle,
  Home,
  Languages,
  Plug,
  ScanLine,
  Search,
  Send,
  TrendingUp,
  UploadCloud,
  Users,
} from "lucide-react";
import { Logo } from "@/components/Logo";
import { LanguageToggle } from "@/components/Topbar";
import { useI18n } from "@/i18n/I18nContext";

function SectionTitle({ icon, children }: { icon: ReactNode; children: ReactNode }) {
  return (
    <div className="mb-4 mt-12 flex items-center gap-3 first:mt-0">
      <span className="grid h-10 w-10 place-items-center rounded-2xl gradient-primary text-white shadow-lift">{icon}</span>
      <h2 className="text-xl font-bold text-ink">{children}</h2>
    </div>
  );
}

function Step({ n, title, body, children }: { n: number; title: string; body: string; children: ReactNode }) {
  return (
    <div className="card grid gap-5 p-5 md:grid-cols-2 md:items-center">
      <div className="order-2 md:order-1">
        <div className="flex items-center gap-2.5">
          <span className="grid h-7 w-7 place-items-center rounded-full bg-primary/10 text-sm font-bold text-primary-700">{n}</span>
          <h3 className="text-base font-bold text-ink">{title}</h3>
        </div>
        <p className="mt-2 text-sm leading-relaxed text-ink-soft">{body}</p>
      </div>
      <div className="order-1 rounded-2xl bg-panel/60 p-5 md:order-2">{children}</div>
    </div>
  );
}

export function Guide() {
  const { t, lang } = useI18n();
  const ar = lang === "ar";

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b border-line glass">
        <div className="mx-auto flex h-16 w-full max-w-4xl items-center justify-between gap-2 px-4 sm:px-6">
          <Link to="/"><Logo size={36} /></Link>
          <div className="flex items-center gap-2">
            <Link to="/" className="inline-flex items-center gap-1.5 rounded-xl border border-line bg-white px-3 py-2 text-xs font-semibold text-ink-soft transition hover:bg-panel hover:text-ink">
              <Home className="h-4 w-4" />
              <span className="hidden sm:inline">{t("nav.home")}</span>
            </Link>
            <LanguageToggle />
          </div>
        </div>
      </header>

      <section className="bg-topo">
        <div className="mx-auto w-full max-w-4xl px-6 py-12 text-center">
          <span className="chip border border-primary/20 bg-primary/10 text-primary-700">
            <HelpCircle className="h-3.5 w-3.5" />
            {t("onb.help")}
          </span>
          <h1 className="mt-4 text-3xl font-extrabold tracking-tight text-ink sm:text-4xl">{t("guide.title")}</h1>
          <p className="mx-auto mt-3 max-w-xl text-sm leading-relaxed text-ink-soft">{t("guide.subtitle")}</p>
        </div>
      </section>

      <main className="mx-auto w-full max-w-4xl px-6 pb-20">
        {/* Authority portal */}
        <SectionTitle icon={<Building2 className="h-5 w-5" />}>{t("landing.enterMinistry")}</SectionTitle>

        <div className="space-y-4">
          <Step
            n={1}
            title={t("analyze.title")}
            body={ar ? "ادخل «الكشف الذكي»، ارفع صورة أو فيديو من الطريق، واضغط زر «تشغيل الكشف». نتيجة الموديل (الخطر المكتشف) تظهر في المكان المخصّص بجوارها." : "Open Smart detection, upload a road image or video, and press Run detection. The model's result (the detected hazard) appears in its slot."}
          >
            <div className="mx-auto max-w-xs">
              <div className="flex flex-col items-center rounded-xl border-2 border-dashed border-line bg-white px-4 py-5 text-center">
                <span className="grid h-10 w-10 place-items-center rounded-xl bg-primary/10 text-primary"><UploadCloud className="h-5 w-5" /></span>
                <div className="mt-2 text-xs font-medium text-ink-soft">{t("analyze.drop")}</div>
              </div>
              <span className="btn-primary mt-2 w-full justify-center"><ScanLine className="h-4 w-4" />{t("analyze.run")}</span>
            </div>
          </Step>

          <Step
            n={2}
            title={t("reports.title")}
            body={ar ? "كل بلاغ وارد (من المواطن أو من الكشف) يظهر كصف في القائمة، وعليه «حالة» ملوّنة. اضغط زر «مراجعة» لفتح البلاغ." : "Every incoming report shows as a row with a colored status. Press Review to open it."}
          >
            <div className="card flex w-full items-center gap-3 px-4 py-3">
              <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-primary/10 text-primary-700"><ClipboardList className="h-5 w-5" /></span>
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-semibold text-ink">{ar ? "حفرة في الطريق الدائري" : "Pothole on the ring road"}</div>
                <div className="mono text-[11px] text-ink-faint">RQ-482193</div>
              </div>
              <span className="chip shrink-0" style={{ backgroundColor: "#E0A00814", color: "#E0A008" }}>
                <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: "#E0A008" }} />
                {t("status.under_review")}
              </span>
              <span className="hidden shrink-0 items-center gap-1 text-xs font-semibold text-primary-700 sm:flex">
                {t("reports.openReview")}
                <ChevronLeft className="h-4 w-4 ltr:rotate-180" />
              </span>
            </div>
          </Step>

          <Step
            n={3}
            title={t("review.title")}
            body={ar ? "داخل البلاغ تشوف كل تفاصيله. اختر الحالة من القائمة، اكتب ردّك للمواطن، واضغط «إرسال الرد للمواطن» — يوصله ويتابعه برقم بلاغه." : "Inside the report you see all details. Pick a status, write your reply, and press Send reply — the citizen receives it and tracks it by number."}
          >
            <div className="w-full space-y-2">
              <select className="input cursor-default" defaultValue="resolved">
                <option value="resolved">{t("status.resolved")}</option>
              </select>
              <textarea className="input resize-none" rows={2} readOnly value={ar ? "تم إصلاح الحفرة. شكرًا لمساهمتك." : "The pothole has been repaired. Thank you."} />
              <span className="btn-primary w-full justify-center"><Send className="h-4 w-4" />{t("review.send")}</span>
            </div>
          </Step>
        </div>

        {/* Citizen portal */}
        <SectionTitle icon={<Users className="h-5 w-5" />}>{t("landing.enterCitizen")}</SectionTitle>

        <div className="space-y-4">
          <Step
            n={1}
            title={t("citizen.report.title")}
            body={ar ? "ارفع صورة أو فيديو للخطر، اكتب الموقع (اختياري وصف)، واضغط «إرسال البلاغ». هتاخد رقم بلاغ تتابع بيه." : "Upload a photo or video of the hazard, enter the location, and press Submit report. You get a report number to track it."}
          >
            <div className="mx-auto max-w-xs space-y-2">
              <div className="flex flex-col items-center rounded-xl border-2 border-dashed border-line bg-white px-4 py-4 text-center">
                <span className="grid h-9 w-9 place-items-center rounded-xl bg-primary/10 text-primary"><UploadCloud className="h-5 w-5" /></span>
                <div className="mt-2 text-xs font-medium text-ink-soft">{t("citizen.report.upload")}</div>
              </div>
              <div className="input text-xs text-ink-faint">{t("citizen.report.locationPh")}</div>
              <span className="btn-primary w-full justify-center"><CheckCircle2 className="h-4 w-4" />{t("citizen.report.submit")}</span>
            </div>
          </Step>

          <Step
            n={2}
            title={t("track.title")}
            body={ar ? "اكتب رقم البلاغ واضغط «متابعة». تشوف مرحلة بلاغك (استلام → مراجعة → رد) ورد الجهة المختصة." : "Enter the report number and press Track. You see its stage (received → review → reply) and the authority's response."}
          >
            <div className="w-full">
              <div className="flex gap-2">
                <span className="input flex-1 text-ink-faint">RQ-482193</span>
                <span className="btn-primary px-4"><Search className="h-4 w-4" /></span>
              </div>
              <div className="mt-4 flex items-center">
                {[t("track.stage1"), t("track.stage2"), t("track.stage3")].map((s, i, a) => (
                  <div key={i} className="flex flex-1 items-center last:flex-none">
                    <div className="flex flex-col items-center gap-1.5">
                      <span className="grid h-7 w-7 place-items-center rounded-full gradient-primary text-white">
                        <CheckCircle2 className="h-3.5 w-3.5" />
                      </span>
                      <span className="text-[10px] font-semibold text-ink">{s}</span>
                    </div>
                    {i < a.length - 1 && <span className="mx-1 mb-5 h-0.5 flex-1 rounded bg-primary" />}
                  </div>
                ))}
              </div>
            </div>
          </Step>

          <Step
            n={3}
            title={t("citizen.risk.title")}
            body={ar ? "تكتب اسم الطريق فتعرف مدى خطورته قبل ما تسلكه. الخدمة دي شغّالة بنموذج التنبؤ وتتفعّل بعد ربطه." : "Enter a road name to learn its risk before you take it. This service runs on the prediction model and activates once connected."}
          >
            <div className="grid place-items-center rounded-xl border-2 border-dashed border-line bg-white px-4 py-6 text-center">
              <span className="grid h-12 w-12 place-items-center rounded-full border-[5px] border-dashed border-line text-ink-faint"><Plug className="h-5 w-5" /></span>
              <div className="mt-2 text-xs font-semibold text-ink">{t("citizen.risk.emptyTitle")}</div>
            </div>
          </Step>
        </div>

        {/* General buttons */}
        <SectionTitle icon={<HelpCircle className="h-5 w-5" />}>{ar ? "أزرار موجودة في كل صفحة" : "Buttons on every page"}</SectionTitle>
        <div className="grid gap-3 sm:grid-cols-3">
          {[
            { btn: <span className="inline-flex items-center gap-1.5 rounded-xl border border-line bg-white px-3 py-2 text-xs font-semibold text-ink-soft"><Home className="h-4 w-4" />{t("nav.home")}</span>, d: ar ? "يرجّعك للصفحة الرئيسية من أي مكان." : "Returns you home from anywhere." },
            { btn: <span className="inline-flex items-center gap-1.5 rounded-xl border border-line bg-white px-3 py-2 text-xs font-semibold text-ink-soft"><Languages className="h-4 w-4" />{t("lang.toggle")}</span>, d: ar ? "يبدّل اللغة بين العربية والإنجليزية." : "Switches Arabic / English." },
            { btn: <span className="inline-flex items-center gap-1.5 rounded-xl border border-line bg-white px-3 py-2 text-xs font-semibold text-ink-soft"><HelpCircle className="h-4 w-4" />{t("onb.help")}</span>, d: ar ? "يفتح هذا الدليل في أي وقت." : "Opens this guide anytime." },
          ].map((g, i) => (
            <div key={i} className="card flex flex-col items-center gap-3 p-5 text-center">
              {g.btn}
              <p className="text-xs leading-relaxed text-ink-soft">{g.d}</p>
            </div>
          ))}
        </div>

        {/* Developer / go-live */}
        <SectionTitle icon={<Plug className="h-5 w-5" />}>{ar ? "التشغيل المباشر (للمطوّر)" : "Go live (for the developer)"}</SectionTitle>
        <div className="card p-5">
          <p className="text-sm leading-relaxed text-ink-soft">
            {ar ? "أول ما الموديلز تجهز، حُط روابط الـ API في ملف الإعداد وغيّر سطرًا واحدًا — وكل الشاشات تشتغل ببيانات حقيقية فورًا." : "Once the models are ready, set the API URLs in the config file and flip one line — every screen goes live instantly."}
          </p>
          <pre className="mono mt-3 overflow-x-auto rounded-xl border border-line bg-ink/[0.03] p-4 text-xs leading-relaxed text-ink-soft">{`VITE_USE_MOCK=false
VITE_API_BASE_URL=https://api.your-domain.com
VITE_MODEL_POTHOLE_URL=/models/pothole/infer
VITE_MODEL_SIGN_DEFECT_URL=/models/sign-defect/infer
VITE_MODEL_ACCIDENT_URL=/models/accident/infer
VITE_MODEL_RISK_URL=/models/risk/predict`}</pre>
        </div>

        {/* CTA */}
        <div className="mt-12 grid gap-3 sm:grid-cols-2">
          <Link to="/app" className="btn-primary justify-center py-3.5">
            <Building2 className="h-[18px] w-[18px]" />
            {t("landing.enterMinistry")}
            <ArrowLeft className="h-4 w-4 ltr:rotate-180" />
          </Link>
          <Link to="/citizen" className="btn-ghost justify-center py-3.5">
            <Users className="h-[18px] w-[18px]" />
            {t("landing.enterCitizen")}
          </Link>
        </div>
      </main>
    </div>
  );
}

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
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
  Sparkles,
  UploadCloud,
  Users,
  X,
} from "lucide-react";
import { useI18n } from "@/i18n/I18nContext";

const KEY = "raqib_seen_v3";
function readSeen() {
  try {
    return localStorage.getItem(KEY) === "1";
  } catch {
    return false;
  }
}
function writeSeen() {
  try {
    localStorage.setItem(KEY, "1");
  } catch {
    /* ignore */
  }
}

const Ctx = createContext<{ open: () => void } | null>(null);
export function useGuide() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useGuide must be used within GuideProvider");
  return c;
}

/** Pinned button that opens the guide as an in-page frame. */
export function HelpButton() {
  const { t } = useI18n();
  const { open } = useGuide();
  return (
    <button
      onClick={open}
      className="inline-flex items-center gap-1.5 rounded-xl border border-line bg-white px-3 py-2 text-xs font-semibold text-ink-soft transition hover:bg-panel hover:text-ink"
    >
      <HelpCircle className="h-4 w-4" />
      <span className="hidden sm:inline">{t("onb.help")}</span>
    </button>
  );
}

function SectionTitle({ icon, children }: { icon: ReactNode; children: ReactNode }) {
  return (
    <div className="mb-4 mt-10 flex items-center gap-3">
      <span className="grid h-10 w-10 place-items-center rounded-2xl gradient-primary text-white shadow-lift">{icon}</span>
      <h3 className="text-lg font-bold text-ink">{children}</h3>
    </div>
  );
}

function Step({ n, title, body, children }: { n: number; title: string; body: string; children: ReactNode }) {
  return (
    <div className="card grid gap-5 p-5 md:grid-cols-2 md:items-center">
      <div className="order-2 md:order-1">
        <div className="flex items-center gap-2.5">
          <span className="grid h-7 w-7 place-items-center rounded-full bg-primary/10 text-sm font-bold text-primary-700">{n}</span>
          <h4 className="text-base font-bold text-ink">{title}</h4>
        </div>
        <p className="mt-2 text-sm leading-relaxed text-ink-soft">{body}</p>
      </div>
      <div className="order-1 rounded-2xl bg-panel/60 p-5 md:order-2">{children}</div>
    </div>
  );
}

function GuideOverlay({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { t, lang } = useI18n();
  const ar = lang === "ar";

  const points = [
    { Icon: ScanLine, t: ar ? "كشف ذكي" : "Smart detection", d: ar ? "يكتشف الخطر تلقائيًا" : "Spots hazards automatically" },
    { Icon: Send, t: ar ? "توجيه فوري" : "Instant routing", d: ar ? "يصل البلاغ للجهة فورًا" : "Report reaches the authority" },
    { Icon: Users, t: ar ? "مشاركة المواطن" : "Citizen participation", d: ar ? "بلّغ وتابِع حتى الحل" : "Report and track to resolution" },
  ];

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          className="fixed inset-0 z-[70] overflow-y-auto bg-ink/45 p-3 backdrop-blur-sm sm:p-6"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.97, y: 16 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ type: "spring", stiffness: 260, damping: 28 }}
            onClick={(e) => e.stopPropagation()}
            className="mx-auto my-2 w-full max-w-3xl overflow-hidden rounded-3xl border border-line bg-bg shadow-soft"
          >
            <div className="sticky top-0 z-10 flex items-center justify-between border-b border-line glass px-5 py-4">
              <div className="flex items-center gap-2">
                <span className="grid h-8 w-8 place-items-center rounded-lg bg-primary/10 text-primary"><HelpCircle className="h-4 w-4" /></span>
                <h2 className="text-base font-bold text-ink">{t("guide.title")}</h2>
              </div>
              <button onClick={onClose} className="grid h-9 w-9 place-items-center rounded-lg text-ink-soft transition hover:bg-panel hover:text-ink" aria-label="close">
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="px-5 py-6 sm:px-7">
              {/* Who is Raqib */}
              <div className="card overflow-hidden">
                <div className="gradient-primary p-6 text-white">
                  <div className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5" />
                    <h3 className="text-xl font-bold">{t("guide.about.t")}</h3>
                  </div>
                  <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-white/90">{t("guide.about.d")}</p>
                </div>
                <div className="grid gap-3 p-5 sm:grid-cols-3">
                  {points.map((p) => (
                    <div key={p.t} className="flex items-center gap-3 rounded-xl bg-panel px-3 py-3">
                      <span className="grid h-9 w-9 place-items-center rounded-lg bg-white text-primary-700"><p.Icon className="h-4 w-4" /></span>
                      <div className="leading-tight">
                        <div className="text-sm font-bold text-ink">{p.t}</div>
                        <div className="text-[11px] text-ink-faint">{p.d}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Authority */}
              <SectionTitle icon={<Building2 className="h-5 w-5" />}>{t("landing.enterMinistry")}</SectionTitle>
              <div className="space-y-4">
                <Step n={1} title={t("analyze.title")} body={ar ? "ادخل «الكشف الذكي»، ارفع صورة أو فيديو من الطريق، واضغط «تشغيل الكشف» — يظهر الخطر المكتشف في مكانه." : "Open Smart detection, upload a road image or video, press Run detection — the detected hazard appears in its slot."}>
                  <div className="mx-auto max-w-xs">
                    <div className="flex flex-col items-center rounded-xl border-2 border-dashed border-line bg-white px-4 py-5 text-center">
                      <span className="grid h-10 w-10 place-items-center rounded-xl bg-primary/10 text-primary"><UploadCloud className="h-5 w-5" /></span>
                      <div className="mt-2 text-xs font-medium text-ink-soft">{t("analyze.drop")}</div>
                    </div>
                    <span className="btn-primary mt-2 w-full justify-center"><ScanLine className="h-4 w-4" />{t("analyze.run")}</span>
                  </div>
                </Step>
                <Step n={2} title={t("reports.title")} body={ar ? "كل بلاغ وارد يظهر كصف بحالته الملوّنة. اضغط «مراجعة» لفتحه." : "Every incoming report shows as a row with a colored status. Press Review to open it."}>
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
                      {t("reports.openReview")}<ChevronLeft className="h-4 w-4 ltr:rotate-180" />
                    </span>
                  </div>
                </Step>
                <Step n={3} title={t("review.title")} body={ar ? "اختر الحالة، اكتب ردّك للمواطن، واضغط «إرسال الرد للمواطن» — يوصله ويتابعه برقم بلاغه." : "Pick a status, write your reply, press Send reply — the citizen receives and tracks it."}>
                  <div className="w-full space-y-2">
                    <div className="input flex items-center justify-between text-ink"><span>{t("status.resolved")}</span><ChevronLeft className="h-4 w-4 rotate-[-90deg] text-ink-faint" /></div>
                    <div className="input text-ink">{ar ? "تم إصلاح الحفرة. شكرًا لمساهمتك." : "The pothole has been repaired. Thank you."}</div>
                    <span className="btn-primary w-full justify-center"><Send className="h-4 w-4" />{t("review.send")}</span>
                  </div>
                </Step>
              </div>

              {/* Citizen */}
              <SectionTitle icon={<Users className="h-5 w-5" />}>{t("landing.enterCitizen")}</SectionTitle>
              <div className="space-y-4">
                <Step n={1} title={t("citizen.report.title")} body={ar ? "ارفع صورة أو فيديو للخطر، اكتب الموقع، واضغط «إرسال البلاغ». هتاخد رقم بلاغ تتابع بيه." : "Upload a photo or video, enter the location, press Submit report. You get a number to track it."}>
                  <div className="mx-auto max-w-xs space-y-2">
                    <div className="flex flex-col items-center rounded-xl border-2 border-dashed border-line bg-white px-4 py-4 text-center">
                      <span className="grid h-9 w-9 place-items-center rounded-xl bg-primary/10 text-primary"><UploadCloud className="h-5 w-5" /></span>
                      <div className="mt-2 text-xs font-medium text-ink-soft">{t("citizen.report.upload")}</div>
                    </div>
                    <div className="input text-xs text-ink-faint">{t("citizen.report.locationPh")}</div>
                    <span className="btn-primary w-full justify-center"><CheckCircle2 className="h-4 w-4" />{t("citizen.report.submit")}</span>
                  </div>
                </Step>
                <Step n={2} title={t("track.title")} body={ar ? "اكتب رقم البلاغ واضغط «متابعة». تشوف مرحلته (استلام ← مراجعة ← رد) ورد الجهة المختصة." : "Enter the number and press Track. See its stage (received → review → reply) and the authority's response."}>
                  <div className="w-full">
                    <div className="flex gap-2">
                      <span className="input flex-1 text-ink-faint">RQ-482193</span>
                      <span className="btn-primary px-4"><Search className="h-4 w-4" /></span>
                    </div>
                    <div className="mt-4 flex items-center">
                      {[t("track.stage1"), t("track.stage2"), t("track.stage3")].map((s, i, a) => (
                        <div key={i} className="flex flex-1 items-center last:flex-none">
                          <div className="flex flex-col items-center gap-1.5">
                            <span className="grid h-7 w-7 place-items-center rounded-full gradient-primary text-white"><CheckCircle2 className="h-3.5 w-3.5" /></span>
                            <span className="text-[10px] font-semibold text-ink">{s}</span>
                          </div>
                          {i < a.length - 1 && <span className="mx-1 mb-5 h-0.5 flex-1 rounded bg-primary" />}
                        </div>
                      ))}
                    </div>
                  </div>
                </Step>
                <Step n={3} title={t("citizen.risk.title")} body={ar ? "تكتب اسم الطريق فتعرف مدى خطورته قبل ما تسلكه. الخدمة دي قيد الإعداد وتتاح قريبًا." : "Enter a road name to learn its risk before you take it. This service is being prepared and arrives soon."}>
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
                  { btn: <span className="inline-flex items-center gap-1.5 rounded-xl border border-line bg-white px-3 py-2 text-xs font-semibold text-ink-soft"><Home className="h-4 w-4" />{t("nav.home")}</span>, d: ar ? "يرجّعك للصفحة الرئيسية." : "Returns you home." },
                  { btn: <span className="inline-flex items-center gap-1.5 rounded-xl border border-line bg-white px-3 py-2 text-xs font-semibold text-ink-soft"><Languages className="h-4 w-4" />{t("lang.toggle")}</span>, d: ar ? "يبدّل اللغة عربي/إنجليزي." : "Switches Arabic / English." },
                  { btn: <span className="inline-flex items-center gap-1.5 rounded-xl border border-line bg-white px-3 py-2 text-xs font-semibold text-ink-soft"><HelpCircle className="h-4 w-4" />{t("onb.help")}</span>, d: ar ? "يفتح هذا الدليل أي وقت." : "Opens this guide anytime." },
                ].map((g, i) => (
                  <div key={i} className="card flex flex-col items-center gap-3 p-5 text-center">
                    {g.btn}
                    <p className="text-xs leading-relaxed text-ink-soft">{g.d}</p>
                  </div>
                ))}
              </div>

              {/* CTA */}
              <div className="mt-10 grid gap-3 sm:grid-cols-2">
                <Link to="/app" onClick={onClose} className="btn-primary justify-center py-3.5">
                  <Building2 className="h-[18px] w-[18px]" />{t("landing.enterMinistry")}
                  <ArrowLeft className="h-4 w-4 ltr:rotate-180" />
                </Link>
                <Link to="/citizen" onClick={onClose} className="btn-ghost justify-center py-3.5">
                  <Users className="h-[18px] w-[18px]" />{t("landing.enterCitizen")}
                </Link>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export function GuideProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!readSeen()) setOpen(true);
  }, []);
  const close = () => {
    writeSeen();
    setOpen(false);
  };
  return (
    <Ctx.Provider value={{ open: () => setOpen(true) }}>
      {children}
      <GuideOverlay open={open} onClose={close} />
    </Ctx.Provider>
  );
}

import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft, Building2, FileCheck2, ScanLine, Send, ShieldCheck, Sparkles, TrendingUp, Users } from "lucide-react";
import { Logo } from "@/components/Logo";
import { LanguageToggle } from "@/components/Topbar";
import { HelpButton } from "@/components/Onboarding";
import { useI18n } from "@/i18n/I18nContext";

function ScannerVisual() {
  return (
    <div className="card relative overflow-hidden rounded-3xl p-3 shadow-soft">
      <div className="relative aspect-[4/3] overflow-hidden rounded-2xl bg-gradient-to-b from-[#d2e8e3] via-[#bcd9d2] to-[#566b66]">
        <svg viewBox="0 0 400 300" className="absolute inset-0 h-full w-full" aria-hidden>
          <path d="M0 300 L165 120 L235 120 L400 300 Z" fill="#3c4a47" opacity="0.5" />
          <line x1="200" y1="300" x2="200" y2="125" stroke="#eef3ef" strokeWidth="3" strokeDasharray="16 18" opacity="0.75" />
        </svg>
        <div className="absolute inset-5">
          <span className="absolute h-6 w-6 border-t-2 border-white/70 ltr:left-0 ltr:border-l-2 rtl:right-0 rtl:border-r-2" />
          <span className="absolute top-0 h-6 w-6 border-t-2 border-white/70 ltr:right-0 ltr:border-r-2 rtl:left-0 rtl:border-l-2" />
          <span className="absolute bottom-0 h-6 w-6 border-b-2 border-white/70 ltr:left-0 ltr:border-l-2 rtl:right-0 rtl:border-r-2" />
          <span className="absolute bottom-0 h-6 w-6 border-b-2 border-white/70 ltr:right-0 ltr:border-r-2 rtl:left-0 rtl:border-l-2" />
        </div>
        <div className="absolute inset-x-5 top-5 h-16 scanline animate-scan rounded-xl" />
        <div className="absolute bottom-5 flex items-center gap-1.5 rounded-full bg-black/40 px-3 py-1.5 text-[11px] font-semibold text-white backdrop-blur ltr:left-5 rtl:right-5">
          <span className="h-1.5 w-1.5 rounded-full bg-accent" />
          Raqib vision
        </div>
      </div>
    </div>
  );
}

export function Landing() {
  const { t, lang } = useI18n();
  const ar = lang === "ar";
  const features = [
    { Icon: ScanLine, t: t("landing.f1t"), d: t("landing.f1d") },
    { Icon: FileCheck2, t: t("landing.f2t"), d: t("landing.f2d") },
    { Icon: TrendingUp, t: t("landing.f3t"), d: t("landing.f3d") },
  ];
  const points = [
    { Icon: ScanLine, t: ar ? "كشف ذكي" : "Smart detection" },
    { Icon: Send, t: ar ? "توجيه فوري للجهة" : "Instant routing" },
    { Icon: Users, t: ar ? "مشاركة المواطن" : "Citizen participation" },
  ];

  return (
    <div className="min-h-screen">
      <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-5">
        <Logo size={40} />
        <div className="flex items-center gap-2">
          <HelpButton />
          <LanguageToggle />
          <span className="hidden rounded-full border border-line bg-white/70 px-3 py-2 text-xs font-semibold text-ink-soft sm:inline">
            {t("brand.gov")}
          </span>
        </div>
      </header>

      <section className="bg-topo">
        <div className="mx-auto grid w-full max-w-6xl items-center gap-12 px-6 py-12 lg:grid-cols-2 lg:py-20">
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
            <span className="chip border border-primary/20 bg-primary/10 text-primary-700">
              <ShieldCheck className="h-3.5 w-3.5" />
              {t("landing.badge")}
            </span>
            <h1 className="mt-5 text-4xl font-extrabold leading-[1.15] tracking-tight text-ink sm:text-[52px]">
              {t("landing.title1")}
              <br />
              <span className="text-gradient">{t("landing.title2")}</span>
            </h1>
            <p className="mt-5 max-w-xl text-base leading-relaxed text-ink-soft">{t("landing.subtitle")}</p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link to="/app" className="btn-primary px-5 py-3 text-[15px]">
                <Building2 className="h-[18px] w-[18px]" />
                {t("landing.enterMinistry")}
                <ArrowLeft className="h-4 w-4 ltr:rotate-180" />
              </Link>
              <Link to="/citizen" className="btn-ghost px-5 py-3 text-[15px]">
                <Users className="h-[18px] w-[18px]" />
                {t("landing.enterCitizen")}
              </Link>
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.6, delay: 0.15 }} className="px-4 lg:px-0">
            <ScannerVisual />
          </motion.div>
        </div>
      </section>

      <section className="border-y border-line bg-white/60">
        <div className="mx-auto w-full max-w-4xl px-6 py-16 text-center">
          <motion.span initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="chip bg-primary/10 text-primary-700">
            <Sparkles className="h-3.5 w-3.5" />
            {t("guide.about.t")}
          </motion.span>
          <motion.h2
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.05 }}
            className="mx-auto mt-4 max-w-3xl text-2xl font-bold leading-relaxed tracking-tight text-ink sm:text-[26px]"
          >
            {t("guide.about.d")}
          </motion.h2>
          <div className="mx-auto mt-8 flex max-w-2xl flex-wrap items-center justify-center gap-3">
            {points.map((p) => (
              <span key={p.t} className="inline-flex items-center gap-2 rounded-full border border-line bg-white px-4 py-2 text-sm font-semibold text-ink-soft">
                <span className="grid h-6 w-6 place-items-center rounded-md bg-primary/10 text-primary"><p.Icon className="h-3.5 w-3.5" /></span>
                {p.t}
              </span>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto w-full max-w-6xl px-6 py-16">
        <div className="grid gap-5 md:grid-cols-3">
          {features.map((f, i) => (
            <motion.div
              key={f.t}
              initial={{ opacity: 0, y: 14 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.45, delay: i * 0.08 }}
              className="card p-6"
            >
              <span className="grid h-11 w-11 place-items-center rounded-xl bg-primary/10 text-primary">
                <f.Icon className="h-5 w-5" />
              </span>
              <h3 className="mt-4 text-base font-bold text-ink">{f.t}</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-soft">{f.d}</p>
            </motion.div>
          ))}
        </div>

        <div className="mt-12 grid gap-5 md:grid-cols-2">
          <PortalCard to="/app" icon={<Building2 className="h-6 w-6" />} title={t("landing.enterMinistry")} desc={t("brand.gov")} />
          <PortalCard to="/citizen" icon={<Users className="h-6 w-6" />} title={t("landing.enterCitizen")} desc={t("citizen.home.subtitle")} />
        </div>

        <p className="mt-10 text-center text-xs text-ink-faint">{t("landing.footer")}</p>
      </section>
    </div>
  );
}

function PortalCard({ to, icon, title, desc }: { to: string; icon: React.ReactNode; title: string; desc: string }) {
  return (
    <Link to={to} className="group card flex items-center justify-between gap-4 p-6 transition-all hover:-translate-y-0.5 hover:shadow-soft">
      <div className="flex items-center gap-4">
        <span className="grid h-12 w-12 place-items-center rounded-2xl bg-primary/10 text-primary transition group-hover:gradient-primary group-hover:text-white">
          {icon}
        </span>
        <div>
          <div className="text-base font-bold text-ink">{title}</div>
          <div className="line-clamp-1 text-sm text-ink-soft">{desc}</div>
        </div>
      </div>
      <ArrowLeft className="h-5 w-5 text-ink-faint transition group-hover:text-primary ltr:rotate-180" />
    </Link>
  );
}

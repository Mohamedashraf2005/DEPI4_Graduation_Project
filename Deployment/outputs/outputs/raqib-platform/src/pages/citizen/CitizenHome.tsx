import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft, Camera, MapPin, Search, ShieldCheck, TrendingUp } from "lucide-react";
import { useI18n } from "@/i18n/I18nContext";

export function CitizenHome() {
  const { t } = useI18n();
  const steps = [
    { Icon: Camera, t: t("citizen.home.s1t"), d: t("citizen.home.s1d") },
    { Icon: ShieldCheck, t: t("citizen.home.s2t"), d: t("citizen.home.s2d") },
    { Icon: MapPin, t: t("citizen.home.s3t"), d: t("citizen.home.s3d") },
  ];

  return (
    <div>
      <section className="bg-topo">
        <div className="mx-auto w-full max-w-4xl px-6 py-16 text-center sm:py-24">
          <motion.span initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="chip border border-primary/20 bg-primary/10 text-primary-700">
            <ShieldCheck className="h-3.5 w-3.5" />
            {t("citizen.brand")}
          </motion.span>
          <motion.h1
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
            className="mx-auto mt-5 max-w-2xl text-4xl font-extrabold leading-tight tracking-tight text-ink sm:text-5xl"
          >
            {t("citizen.home.title")}
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="mx-auto mt-5 max-w-xl text-base leading-relaxed text-ink-soft"
          >
            {t("citizen.home.subtitle")}
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className="mt-8 flex flex-wrap items-center justify-center gap-3"
          >
            <Link to="/citizen/report" className="btn-primary px-6 py-3.5 text-[15px]">
              <Camera className="h-5 w-5" />
              {t("citizen.home.cta")}
              <ArrowLeft className="h-4 w-4 ltr:rotate-180" />
            </Link>
            <Link to="/citizen/risk" className="btn-ghost px-6 py-3.5 text-[15px]">
              <TrendingUp className="h-[18px] w-[18px]" />
              {t("citizen.home.risk")}
            </Link>
          </motion.div>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.25 }} className="mt-4">
            <Link to="/citizen/track" className="inline-flex items-center gap-1.5 text-sm font-semibold text-primary-700 hover:underline">
              <Search className="h-4 w-4" />
              {t("citizen.home.track")}
            </Link>
          </motion.div>
        </div>
      </section>

      <section className="mx-auto w-full max-w-5xl px-6 pb-20">
        <div className="grid gap-5 md:grid-cols-3">
          {steps.map((s, i) => (
            <motion.div
              key={s.t}
              initial={{ opacity: 0, y: 14 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.08 }}
              className="card relative p-6 text-center"
            >
              <span className="mono absolute top-2 text-5xl font-bold text-primary/[0.07] ltr:right-4 rtl:left-4">{i + 1}</span>
              <span className="mx-auto grid h-14 w-14 place-items-center rounded-2xl gradient-primary text-white shadow-lift">
                <s.Icon className="h-6 w-6" />
              </span>
              <h3 className="mt-4 text-base font-bold text-ink">{s.t}</h3>
              <p className="mx-auto mt-2 max-w-xs text-sm leading-relaxed text-ink-soft">{s.d}</p>
            </motion.div>
          ))}
        </div>
      </section>
    </div>
  );
}

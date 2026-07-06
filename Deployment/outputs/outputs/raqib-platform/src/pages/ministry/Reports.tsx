import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ChevronLeft, ClipboardList, Inbox } from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { CASE_STATUS, useCases } from "@/api/store";
import { timeAgo } from "@/lib/utils";
import { useI18n } from "@/i18n/I18nContext";

export function Reports() {
  const { t, lang } = useI18n();
  const navigate = useNavigate();
  const cases = useCases();

  const counts = useMemo(() => {
    const pending = cases.filter((c) => c.status === "new" || c.status === "under_review").length;
    const replied = cases.filter((c) => c.status === "resolved" || c.status === "rejected").length;
    return { total: cases.length, pending, replied };
  }, [cases]);

  return (
    <div>
      <PageHeader icon={<ClipboardList className="h-5 w-5" />} title={t("reports.title")} subtitle={t("reports.subtitle")} />

      <div className="mb-5 grid grid-cols-3 gap-3">
        {[
          { k: "reports.count.total", v: counts.total, c: "#0E9F8E" },
          { k: "reports.count.pending", v: counts.pending, c: "#E0A008" },
          { k: "reports.count.replied", v: counts.replied, c: "#1A9E54" },
        ].map((s) => (
          <div key={s.k} className="card p-4">
            <div className="flex items-center gap-2 text-xs font-semibold text-ink-soft">
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: s.c }} />
              {t(s.k)}
            </div>
            <div className="mono mt-2 text-2xl font-semibold text-ink">{s.v}</div>
          </div>
        ))}
      </div>

      {cases.length === 0 ? (
        <div className="card grid place-items-center py-20 text-center">
          <span className="grid h-16 w-16 place-items-center rounded-2xl border-2 border-dashed border-line text-ink-faint">
            <Inbox className="h-7 w-7" />
          </span>
          <div className="mt-4 font-semibold text-ink">{t("reports.emptyTitle")}</div>
          <p className="mx-auto mt-2 max-w-sm text-sm leading-relaxed text-ink-faint">{t("reports.emptyBody")}</p>
        </div>
      ) : (
        <div className="card divide-y divide-line">
          {cases.map((c, i) => {
            const sc = CASE_STATUS[c.status];
            return (
              <motion.button
                key={c.ref}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.04 }}
                onClick={() => navigate(`/app/reports/${c.ref}`)}
                className="flex w-full items-center gap-3 px-5 py-4 text-start transition hover:bg-panel/60"
              >
                <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-primary/10 text-primary-700">
                  <ClipboardList className="h-5 w-5" />
                </span>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-semibold text-ink">{c.description || c.location || c.ref}</div>
                  <div className="mt-0.5 flex items-center gap-2 text-xs text-ink-soft">
                    <span className="mono text-ink-faint">{c.ref}</span>
                    <span>· {c.location || "—"}</span>
                    <span className="text-ink-faint">· {timeAgo(c.createdAt, lang)}</span>
                  </div>
                </div>
                <span className="chip shrink-0" style={{ backgroundColor: `${sc.color}14`, color: sc.color }}>
                  <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: sc.color }} />
                  {t(sc.key)}
                </span>
                <span className="hidden shrink-0 items-center gap-1 text-xs font-semibold text-primary-700 sm:flex">
                  {t("reports.openReview")}
                  <ChevronLeft className="h-4 w-4 ltr:rotate-180" />
                </span>
              </motion.button>
            );
          })}
        </div>
      )}
    </div>
  );
}

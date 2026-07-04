import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { motion } from "framer-motion";
import { CheckCircle2, ChevronRight, MapPin, Paperclip, Send } from "lucide-react";
import { CASE_STATUS, type CaseStatus, replyCase, useCases } from "@/api/store";
import { formatDateTime } from "@/lib/utils";
import { useI18n } from "@/i18n/I18nContext";

const options: CaseStatus[] = ["under_review", "resolved", "rejected"];

export function CaseReview() {
  const { ref = "" } = useParams();
  const { t, lang } = useI18n();
  const cases = useCases();
  const c = cases.find((x) => x.ref === ref);
  const [status, setStatus] = useState<CaseStatus>("under_review");
  const [msg, setMsg] = useState("");
  const [justSent, setJustSent] = useState(false);

  if (!c) {
    return (
      <div className="py-24 text-center">
        <p className="text-ink-soft">{t("review.notFound")}</p>
        <Link to="/app/reports" className="btn-ghost mt-4 inline-flex">{t("review.back")}</Link>
      </div>
    );
  }

  const replied = !!c.reply;

  function send() {
    if (!msg.trim()) return;
    replyCase(ref, status, msg.trim());
    setJustSent(true);
  }

  const sc = CASE_STATUS[c.status];

  return (
    <div>
      <Link to="/app/reports" className="mb-4 inline-flex items-center gap-1.5 text-sm font-medium text-ink-soft transition hover:text-ink">
        <ChevronRight className="h-4 w-4 ltr:rotate-180" />
        {t("review.back")}
      </Link>

      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="mono text-sm font-bold tracking-wider text-primary-700">{c.ref}</span>
            <span className="chip ring-1 ring-inset" style={{ backgroundColor: `${sc.color}14`, color: sc.color }}>
              <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: sc.color }} />
              {t(sc.key)}
            </span>
          </div>
          <h1 className="mt-1.5 text-xl font-bold text-ink">{t("review.title")}</h1>
        </div>
        <span className="rounded-full bg-panel px-3 py-1 text-xs text-ink-soft">{t("reports.fromCitizen")}</span>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        {/* details */}
        <div className="card p-5">
          <h2 className="mb-4 text-sm font-bold text-ink">{t("review.details")}</h2>
          <dl className="space-y-3 text-sm">
            <div>
              <dt className="text-xs font-semibold text-ink-faint">{t("citizen.report.desc")}</dt>
              <dd className="mt-1 leading-relaxed text-ink">{c.description || "—"}</dd>
            </div>
            <div className="flex items-center gap-2 border-t border-line pt-3">
              <MapPin className="h-4 w-4 text-primary" />
              <span className="text-ink">{c.location || "—"}</span>
            </div>
            <div className="flex items-center gap-2">
              <Paperclip className="h-4 w-4 text-ink-faint" />
              <span className="text-ink-soft">{c.mediaName ?? t("review.noMedia")}</span>
            </div>
            <div className="border-t border-line pt-3 text-xs text-ink-faint">
              {formatDateTime(c.createdAt, lang)}
            </div>
          </dl>
        </div>

        {/* reply */}
        <div className="card p-5">
          <h2 className="mb-4 text-sm font-bold text-ink">{t("review.reply")}</h2>

          {justSent || replied ? (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <div className="flex items-center gap-2 rounded-xl bg-primary/10 p-4 text-primary-700">
                <CheckCircle2 className="h-5 w-5" />
                <span className="text-sm font-semibold">{t("review.sent")}</span>
              </div>
              <div className="mt-4 rounded-xl border border-line p-4">
                <div className="text-xs font-semibold text-ink-faint">{t("track.reply")}</div>
                <p className="mt-1.5 text-sm leading-relaxed text-ink">{msg || c.reply}</p>
              </div>
              <button onClick={() => { setJustSent(false); setMsg(""); }} className="btn-ghost mt-4 w-full">
                {t("review.editReply")}
              </button>
            </motion.div>
          ) : (
            <>
              <label className="label">{t("review.setStatus")}</label>
              <select value={status} onChange={(e) => setStatus(e.target.value as CaseStatus)} className="input cursor-pointer">
                {options.map((s) => (
                  <option key={s} value={s}>{t(CASE_STATUS[s].key)}</option>
                ))}
              </select>

              <label className="label mt-4">{t("review.reply")}</label>
              <textarea value={msg} onChange={(e) => setMsg(e.target.value)} rows={5} placeholder={t("review.replyPh")} className="input resize-none" />

              <button onClick={send} disabled={!msg.trim()} className="btn-primary mt-4 w-full py-3">
                <Send className="h-4 w-4" />
                {t("review.send")}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

import { useState } from "react";
import { motion } from "framer-motion";
import { CheckCircle2, MapPin, MessageSquare, Search } from "lucide-react";
import { CASE_STATUS, useCases } from "@/api/store";
import { formatDateTime } from "@/lib/utils";
import { useI18n } from "@/i18n/I18nContext";

export function CitizenTrack() {
  const { t, lang } = useI18n();
  const cases = useCases();
  const [q, setQ] = useState("");
  const [lookup, setLookup] = useState<string | null>(null);

  const found = lookup ? cases.find((c) => c.ref === lookup.trim()) : null;

  const stages = (status: string, replied: boolean) => [
    { label: t("track.stage1"), done: true },
    { label: t("track.stage2"), done: status !== "new" },
    { label: t("track.stage3"), done: replied },
  ];

  return (
    <div className="mx-auto w-full max-w-xl px-6 py-12">
      <h1 className="text-2xl font-bold text-ink">{t("track.title")}</h1>
      <p className="mt-2 text-sm text-ink-soft">{t("track.subtitle")}</p>

      <div className="mt-5 flex gap-2">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute top-1/2 h-4 w-4 -translate-y-1/2 text-ink-faint ltr:left-3.5 rtl:right-3.5" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && setLookup(q)}
            placeholder={t("track.ph")}
            className="input ltr:pl-10 rtl:pr-10"
          />
        </div>
        <button onClick={() => setLookup(q)} className="btn-primary px-5">{t("track.btn")}</button>
      </div>

      {lookup && !found && (
        <div className="mt-5 card p-6 text-center text-sm text-ink-soft">{t("track.notFound")}</div>
      )}

      {found && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mt-5 card p-6">
          <div className="flex items-center justify-between">
            <span className="mono text-sm font-bold tracking-wider text-primary-700">{found.ref}</span>
            <span className="chip" style={{ backgroundColor: `${CASE_STATUS[found.status].color}14`, color: CASE_STATUS[found.status].color }}>
              {t(CASE_STATUS[found.status].key)}
            </span>
          </div>
          <p className="mt-3 text-sm text-ink">{found.description || "—"}</p>
          <div className="mt-1 flex items-center gap-1 text-xs text-ink-soft">
            <MapPin className="h-3.5 w-3.5" />
            {found.location || "—"} · {formatDateTime(found.createdAt, lang)}
          </div>

          {/* progress */}
          <div className="mt-6 flex items-center">
            {stages(found.status, !!found.reply).map((s, i, arr) => (
              <div key={i} className="flex flex-1 items-center last:flex-none">
                <div className="flex flex-col items-center gap-1.5">
                  <span className={"grid h-8 w-8 place-items-center rounded-full text-xs font-bold transition " + (s.done ? "gradient-primary text-white" : "bg-panel text-ink-faint")}>
                    {s.done ? <CheckCircle2 className="h-4 w-4" /> : i + 1}
                  </span>
                  <span className={"text-[10px] " + (s.done ? "font-semibold text-ink" : "text-ink-faint")}>{s.label}</span>
                </div>
                {i < arr.length - 1 && <span className={"mx-1 mb-5 h-0.5 flex-1 rounded " + (arr[i + 1].done ? "bg-primary" : "bg-line")} />}
              </div>
            ))}
          </div>

          {/* authority reply */}
          <div className="mt-6 rounded-xl border border-line p-4">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-ink-faint">
              <MessageSquare className="h-3.5 w-3.5" />
              {t("track.reply")}
            </div>
            {found.reply ? (
              <p className="mt-2 text-sm leading-relaxed text-ink">{found.reply}</p>
            ) : (
              <p className="mt-2 text-sm text-ink-faint">{t("track.awaitingReply")}</p>
            )}
          </div>
        </motion.div>
      )}

      {/* this session's reports */}
      <div className="mt-8">
        <h2 className="mb-3 text-sm font-bold text-ink">{t("track.mine")}</h2>
        {cases.length === 0 ? (
          <p className="text-sm text-ink-faint">{t("track.none")}</p>
        ) : (
          <div className="space-y-2.5">
            {cases.map((c) => (
              <button
                key={c.ref}
                onClick={() => { setQ(c.ref); setLookup(c.ref); }}
                className="card flex w-full items-center gap-3 p-4 text-start transition hover:shadow-soft"
              >
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-semibold text-ink">{c.description || c.location || c.ref}</div>
                  <div className="mono text-[11px] text-ink-faint">{c.ref}</div>
                </div>
                <span className="chip shrink-0" style={{ backgroundColor: `${CASE_STATUS[c.status].color}14`, color: CASE_STATUS[c.status].color }}>
                  {t(CASE_STATUS[c.status].key)}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>

      <p className="mt-6 text-center text-xs text-ink-faint">{t("track.note")}</p>
    </div>
  );
}

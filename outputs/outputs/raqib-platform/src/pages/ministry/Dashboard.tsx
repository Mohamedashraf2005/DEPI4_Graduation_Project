import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Area,
  AreaChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
} from "recharts";
import { Activity, ChevronLeft, Clock, Download, Info, Plus, ShieldAlert, TriangleAlert } from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { KPIStat } from "@/components/ui/Stat";
import { HazardChip, SeverityBadge, SourceChip, StatusPill } from "@/components/ui/Badge";
import { SaudiMap } from "@/components/SaudiMap";
import { listReports } from "@/api/reports";
import { bySeverity, byType, kpis, trend7 } from "@/mock/data";
import { timeAgo } from "@/lib/utils";
import { useI18n } from "@/i18n/I18nContext";
import type { HazardReport } from "@/types";

export function Dashboard() {
  const { t, tl, lang, dir } = useI18n();
  const navigate = useNavigate();
  const [reports, setReports] = useState<HazardReport[]>([]);

  useEffect(() => {
    listReports().then(setReports);
  }, []);

  const active = reports.filter((r) => !["resolved"].includes(r.status));

  return (
    <div>
      <PageHeader
        icon={<Activity className="h-5 w-5" />}
        title={t("dash.title")}
        subtitle={t("dash.subtitle")}
        actions={
          <>
            <button className="btn-ghost">
              <Download className="h-4 w-4" />
              <span className="hidden sm:inline">{t("common.export")}</span>
            </button>
            <button onClick={() => navigate("/app/analyze")} className="btn-primary">
              <Plus className="h-4 w-4" />
              {t("dash.newCapture")}
            </button>
          </>
        }
      />

      {/* KPIs */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KPIStat index={0} icon={ShieldAlert} accent="#0E9F8E" label={t("dash.kpi.active")} value={String(kpis.active.value)} delta={kpis.active.delta} trend={kpis.active.trend} />
        <KPIStat index={1} icon={Clock} accent="#E0A008" label={t("dash.kpi.pending")} value={String(kpis.pending.value)} delta={kpis.pending.delta} upIsGood={false} trend={kpis.pending.trend} />
        <KPIStat index={2} icon={Activity} accent="#12B5C4" label={t("dash.kpi.response")} value={`${kpis.responseMins.value}m`} delta={kpis.responseMins.delta} upIsGood={false} trend={kpis.responseMins.trend} />
        <KPIStat index={3} icon={TriangleAlert} accent="#F07316" label={t("dash.kpi.risk")} value={String(kpis.highRisk.value)} delta={kpis.highRisk.delta} upIsGood={false} trend={kpis.highRisk.trend} />
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-3">
        {/* left: feed + trend */}
        <div className="space-y-5 lg:col-span-2">
          {/* live feed */}
          <div className="card overflow-hidden">
            <div className="flex items-center justify-between border-b border-line px-5 py-4">
              <div className="flex items-center gap-2.5">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-sev-low/60" />
                  <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-sev-low" />
                </span>
                <h2 className="text-sm font-bold text-ink">{t("dash.feed.title")}</h2>
              </div>
              <button onClick={() => navigate("/app/reports")} className="text-xs font-semibold text-primary-700 hover:underline">
                {t("common.viewAll")}
              </button>
            </div>

            <div className="flex items-start gap-2 bg-panel/60 px-5 py-2.5 text-[11px] text-ink-soft">
              <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-faint" />
              {t("dash.feed.note")}
            </div>

            <div className="divide-y divide-line">
              {reports.slice(0, 6).map((r, i) => (
                <motion.button
                  key={r.id}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                  onClick={() => navigate(`/app/reports/${r.id}`)}
                  className="flex w-full items-center gap-3 px-5 py-3.5 text-start transition hover:bg-panel/60"
                >
                  <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-panel">
                    <HazardChip type={r.type} />
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-sm font-semibold text-ink">{tl(r.title)}</span>
                      <span className="mono shrink-0 text-[10px] text-ink-faint">{r.id}</span>
                    </div>
                    <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-ink-soft">
                      <span>{tl(r.location.city)} · {tl(r.location.road)}</span>
                      <SourceChip source={r.source} />
                      <span className="text-ink-faint">· {timeAgo(r.createdAt, lang)}</span>
                    </div>
                  </div>
                  <div className="hidden shrink-0 flex-col items-end gap-1.5 sm:flex">
                    <SeverityBadge level={r.severity} />
                    <StatusPill status={r.status} />
                  </div>
                  <ChevronLeft className="h-4 w-4 shrink-0 text-ink-faint ltr:rotate-180" />
                </motion.button>
              ))}
            </div>
          </div>

          {/* trend */}
          <div className="card p-5">
            <h2 className="mb-4 text-sm font-bold text-ink">{t("dash.chart.trend")}</h2>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={trend7} margin={{ top: 6, right: 6, left: 6, bottom: 0 }}>
                <defs>
                  <linearGradient id="gA" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#0E9F8E" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#0E9F8E" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gB" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#12B5C4" stopOpacity={0.25} />
                    <stop offset="100%" stopColor="#12B5C4" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey={lang === "ar" ? "day" : "dayEn"}
                  reversed={dir === "rtl"}
                  axisLine={false}
                  tickLine={false}
                  dy={8}
                />
                <Tooltip
                  cursor={{ stroke: "#0E9F8E", strokeOpacity: 0.2 }}
                  contentStyle={{ direction: dir }}
                />
                <Area type="monotone" dataKey="reports" stroke="#0E9F8E" strokeWidth={2.5} fill="url(#gA)" />
                <Area type="monotone" dataKey="dispatched" stroke="#12B5C4" strokeWidth={2} fill="url(#gB)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* right: map + severity + types */}
        <div className="space-y-5">
          <div className="card p-5">
            <h2 className="text-sm font-bold text-ink">{t("dash.map.title")}</h2>
            <p className="mb-3 text-[11px] text-ink-faint">{t("dash.map.legend")}</p>
            <SaudiMap reports={active} onSelect={(r) => navigate(`/app/reports/${r.id}`)} />
          </div>

          <div className="card p-5">
            <h2 className="mb-2 text-sm font-bold text-ink">{t("dash.chart.severity")}</h2>
            <div className="flex items-center gap-4">
              <ResponsiveContainer width={120} height={120}>
                <PieChart>
                  <Pie data={bySeverity} dataKey="value" innerRadius={34} outerRadius={56} paddingAngle={3} stroke="none">
                    {bySeverity.map((s) => (
                      <Cell key={s.level} fill={s.color} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
              <div className="flex-1 space-y-1.5">
                {bySeverity.map((s) => (
                  <div key={s.level} className="flex items-center justify-between text-xs">
                    <span className="flex items-center gap-2 text-ink-soft">
                      <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: s.color }} />
                      {lang === "ar" ? s.ar : s.en}
                    </span>
                    <span className="mono font-semibold text-ink">{s.value}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="card p-5">
            <h2 className="mb-3 text-sm font-bold text-ink">{t("dash.chart.types")}</h2>
            <div className="space-y-3">
              {byType.map((tp) => (
                <div key={tp.key}>
                  <div className="mb-1 flex items-center justify-between text-xs">
                    <span className="font-medium text-ink-soft">{lang === "ar" ? tp.ar : tp.en}</span>
                    <span className="mono font-semibold text-ink">{tp.value}</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-ink/[0.06]">
                    <div className="h-full rounded-full" style={{ width: `${tp.value}%`, backgroundColor: tp.color }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

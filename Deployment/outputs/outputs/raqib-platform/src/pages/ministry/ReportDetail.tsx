import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Camera,
  CheckCircle2,
  ChevronRight,
  Download,
  FileText,
  Forward,
  MapPin,
  Play,
  ScanLine,
  Send,
  ShieldCheck,
} from "lucide-react";
import { HazardChip, HazardIcon, SeverityBadge, SourceChip, StatusPill } from "@/components/ui/Badge";
import { Meter } from "@/components/ui/Gauge";
import { dispatchReport, getReport } from "@/api/reports";
import { formatDateTime, hazardMeta, severityMeta } from "@/lib/utils";
import { useI18n } from "@/i18n/I18nContext";
import type { HazardReport, ReportStatus, TimelineEvent } from "@/types";

const timelineIcon: Record<TimelineEvent["kind"], typeof Camera> = {
  capture: Camera,
  analysis: ScanLine,
  report: FileText,
  dispatch: Send,
  ack: CheckCircle2,
  resolve: ShieldCheck,
};

function MediaFrame({ report }: { report: HazardReport }) {
  const { t, tl } = useI18n();
  return (
    <div className="relative aspect-video overflow-hidden rounded-2xl bg-gradient-to-b from-[#cfe7e2] via-[#aeccc5] to-[#46524f]">
      <svg viewBox="0 0 640 360" className="absolute inset-0 h-full w-full">
        <path d="M0 360 L270 150 L370 150 L640 360 Z" fill="#3a4744" opacity="0.5" />
        <line x1="320" y1="360" x2="320" y2="156" stroke="#eef3ef" strokeWidth="4" strokeDasharray="18 20" opacity="0.7" />
      </svg>

      <div className="absolute inset-4 rounded-xl border border-white/25" />
      <div className="absolute inset-x-4 top-4 h-20 scanline animate-scan rounded-xl" />

      {/* model bounding boxes */}
      {report.detections.map((d, i) => {
        const c = severityMeta[report.severity].color;
        return (
          <div
            key={i}
            className="absolute rounded-md border-2"
            style={{
              left: `${d.x * 100}%`,
              top: `${d.y * 100}%`,
              width: `${d.w * 100}%`,
              height: `${d.h * 100}%`,
              borderColor: c,
              boxShadow: `0 0 0 3px ${c}22`,
            }}
          >
            <span
              className="absolute -top-6 whitespace-nowrap rounded-md px-2 py-0.5 text-[11px] font-bold text-white ltr:left-0 rtl:right-0"
              style={{ backgroundColor: c }}
            >
              {tl(d.label)} · {Math.round(d.confidence * 100)}%
            </span>
          </div>
        );
      })}

      {/* top chips */}
      <div className="absolute top-6 flex items-center gap-2 ltr:left-6 rtl:right-6">
        <span className="flex items-center gap-1.5 rounded-full bg-black/45 px-2.5 py-1 text-[11px] font-semibold text-white backdrop-blur">
          <span className="h-1.5 w-1.5 rounded-full bg-sev-crit" />
          {report.mediaKind === "video" ? "REC" : t("reports.detail.media")}
        </span>
        <span className="rounded-full bg-black/45 px-2.5 py-1 text-[11px] font-semibold text-white backdrop-blur">
          {report.source === "field_camera" ? "CAM 04" : report.source === "citizen" ? "MOBILE" : "UPLOAD"}
        </span>
      </div>

      {report.mediaKind === "video" && (
        <button className="absolute inset-0 grid place-items-center">
          <span className="grid h-16 w-16 place-items-center rounded-full bg-white/90 text-primary-700 shadow-lift transition hover:scale-105">
            <Play className="h-7 w-7 ltr:ml-1 rtl:mr-1" />
          </span>
        </button>
      )}
    </div>
  );
}

export function ReportDetail() {
  const { id = "" } = useParams();
  const { t, tl, lang } = useI18n();
  const navigate = useNavigate();
  const [report, setReport] = useState<HazardReport | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [status, setStatus] = useState<ReportStatus>("new");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getReport(id).then((r) => {
      setReport(r ?? null);
      if (r) setStatus(r.status);
      setLoaded(true);
    });
  }, [id]);

  if (loaded && !report) {
    return (
      <div className="py-24 text-center">
        <p className="text-ink-soft">{t("reports.detail.notFound")}</p>
        <Link to="/app/reports" className="btn-ghost mt-4 inline-flex">{t("reports.detail.back")}</Link>
      </div>
    );
  }
  if (!report) {
    return <div className="h-64 animate-pulse rounded-2xl bg-panel" />;
  }

  const canDispatch = ["new", "analyzing", "pending_dispatch"].includes(status);

  async function onDispatch() {
    setBusy(true);
    await dispatchReport(report!.id);
    setStatus("dispatched");
    setBusy(false);
  }

  const hz = hazardMeta[report.type];

  return (
    <div>
      <Link
        to="/app/reports"
        className="mb-4 inline-flex items-center gap-1.5 text-sm font-medium text-ink-soft transition hover:text-ink"
      >
        <ChevronRight className="h-4 w-4 ltr:rotate-180" />
        {t("reports.detail.back")}
      </Link>

      {/* header */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-5 flex flex-wrap items-start justify-between gap-4"
      >
        <div>
          <div className="flex items-center gap-2">
            <span className="mono text-xs text-ink-faint">{report.id}</span>
            <span className="text-ink-faint">·</span>
            <span className="text-xs text-ink-soft">{formatDateTime(report.createdAt, lang)}</span>
          </div>
          <h1 className="mt-1.5 text-2xl font-bold tracking-tight text-ink">{tl(report.title)}</h1>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <HazardChip type={report.type} />
            <SeverityBadge level={report.severity} />
            <StatusPill status={status} />
            <SourceChip source={report.source} />
          </div>
        </div>
      </motion.div>

      <div className="grid gap-5 lg:grid-cols-3">
        {/* left */}
        <div className="space-y-5 lg:col-span-2">
          <div className="card p-3">
            <MediaFrame report={report} />
          </div>

          {/* AI description */}
          <div className="card p-5">
            <div className="mb-3 flex items-center gap-2">
              <span className="grid h-7 w-7 place-items-center rounded-lg bg-primary/10 text-primary">
                <ScanLine className="h-4 w-4" />
              </span>
              <h2 className="text-sm font-bold text-ink">{t("reports.detail.aiDesc")}</h2>
            </div>
            <p className="text-[15px] leading-relaxed text-ink-soft">{tl(report.description)}</p>

            <div className="mt-5 grid gap-4 sm:grid-cols-2">
              <div className="panel p-4">
                <div className="text-xs font-semibold text-ink-faint">{t("reports.detail.hazard")}</div>
                <div className="mt-2 flex items-center gap-2">
                  <span className="grid h-8 w-8 place-items-center rounded-lg" style={{ backgroundColor: `${hz.color}1a`, color: hz.color }}>
                    <HazardIcon type={report.type} className="h-4 w-4" />
                  </span>
                  <span className="font-semibold text-ink">{tl(hz.label)}</span>
                </div>
              </div>
              <div className="panel p-4">
                <div className="flex items-center justify-between text-xs font-semibold text-ink-faint">
                  <span>{t("common.confidence")}</span>
                  <span className="mono text-ink">{Math.round(report.confidence * 100)}%</span>
                </div>
                <div className="mt-3">
                  <Meter value={report.confidence * 100} color={severityMeta[report.severity].color} />
                </div>
              </div>
            </div>
          </div>

          {/* detections */}
          <div className="card p-5">
            <h2 className="mb-3 text-sm font-bold text-ink">{t("reports.detail.detections")}</h2>
            <div className="space-y-3">
              {report.detections.map((d, i) => (
                <div key={i} className="flex items-center gap-3">
                  <span className="mono w-10 shrink-0 text-xs text-ink-faint">#{i + 1}</span>
                  <span className="w-40 shrink-0 text-sm font-medium text-ink">{tl(d.label)}</span>
                  <div className="flex-1">
                    <Meter value={d.confidence * 100} delay={i * 0.1} />
                  </div>
                  <span className="mono w-12 shrink-0 text-end text-xs font-semibold text-ink">
                    {Math.round(d.confidence * 100)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* right */}
        <div className="space-y-5">
          {/* routed authority + actions */}
          <div className="card p-5">
            <div className="text-xs font-semibold text-ink-faint">{t("reports.detail.routed")}</div>
            <div className="mt-3 flex items-start gap-3">
              <span className="grid h-11 w-11 shrink-0 place-items-center rounded-xl gradient-primary text-white">
                <ShieldCheck className="h-5 w-5" />
              </span>
              <div>
                <div className="font-bold text-ink">{tl(report.authority.name)}</div>
                <div className="text-xs text-ink-soft">{tl(report.authority.region)}</div>
                <div className="mono mt-1 text-[10px] text-ink-faint">{report.authority.id}</div>
              </div>
            </div>

            <div className="mt-5 space-y-2">
              {canDispatch ? (
                <button onClick={onDispatch} disabled={busy} className="btn-primary w-full">
                  <Send className="h-4 w-4" />
                  {busy ? "..." : t("reports.detail.dispatch")}
                </button>
              ) : (
                <div className="flex items-center justify-center gap-2 rounded-xl bg-primary/10 py-2.5 text-sm font-semibold text-primary-700">
                  <CheckCircle2 className="h-4 w-4" />
                  {t("reports.detail.dispatched")}
                </div>
              )}
              <div className="grid grid-cols-2 gap-2">
                <button className="btn-ghost">
                  <Forward className="h-4 w-4" />
                  {t("reports.detail.forward")}
                </button>
                <button className="btn-ghost">
                  <Download className="h-4 w-4" />
                  {t("common.export")}
                </button>
              </div>
            </div>
          </div>

          {/* location */}
          <div className="card p-5">
            <div className="flex items-center gap-2 text-sm font-bold text-ink">
              <MapPin className="h-4 w-4 text-primary" />
              {t("common.location")}
            </div>
            <div className="mt-2 text-sm text-ink-soft">
              {tl(report.location.city)} · {tl(report.location.road)}
            </div>
            <div className="relative mt-3 aspect-[16/10] overflow-hidden rounded-xl bg-panel bg-grid">
              <div
                className="absolute -translate-x-1/2 -translate-y-1/2"
                style={{ left: `${report.location.x}%`, top: `${report.location.y}%` }}
              >
                <span className="relative flex h-3.5 w-3.5">
                  <span
                    className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60"
                    style={{ backgroundColor: severityMeta[report.severity].color }}
                  />
                  <span
                    className="relative inline-flex h-3.5 w-3.5 rounded-full ring-2 ring-white"
                    style={{ backgroundColor: severityMeta[report.severity].color }}
                  />
                </span>
              </div>
            </div>
          </div>

          {/* timeline */}
          <div className="card p-5">
            <h2 className="mb-4 text-sm font-bold text-ink">{t("reports.detail.timeline")}</h2>
            <ol className="relative space-y-4 ltr:pl-1 rtl:pr-1">
              {report.timeline.map((ev, i) => {
                const Icon = timelineIcon[ev.kind];
                const last = i === report.timeline.length - 1;
                return (
                  <li key={i} className="flex gap-3">
                    <div className="flex flex-col items-center">
                      <span className="grid h-7 w-7 place-items-center rounded-full bg-primary/10 text-primary">
                        <Icon className="h-3.5 w-3.5" />
                      </span>
                      {!last && <span className="mt-1 w-px flex-1 bg-line" />}
                    </div>
                    <div className="pb-1">
                      <div className="text-sm font-medium text-ink">{tl(ev.label)}</div>
                      <div className="mono text-[11px] text-ink-faint">{formatDateTime(ev.at, lang)}</div>
                    </div>
                  </li>
                );
              })}
              {status === "dispatched" && !report.timeline.some((e) => e.kind === "dispatch") && (
                <li className="flex gap-3">
                  <div className="flex flex-col items-center">
                    <span className="grid h-7 w-7 place-items-center rounded-full bg-primary text-white">
                      <Send className="h-3.5 w-3.5" />
                    </span>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-ink">{t("reports.detail.dispatched")}</div>
                    <div className="mono text-[11px] text-ink-faint">{formatDateTime(new Date().toISOString(), lang)}</div>
                  </div>
                </li>
              )}
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
}

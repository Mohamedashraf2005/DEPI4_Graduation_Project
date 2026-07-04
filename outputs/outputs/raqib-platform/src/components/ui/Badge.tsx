import { Car, Construction, Signpost, TriangleAlert, type LucideIcon } from "lucide-react";
import { useI18n } from "@/i18n/I18nContext";
import { cn, hazardMeta, severityMeta, sourceLabel, statusMeta } from "@/lib/utils";
import type { HazardType, ModelStatus, ReportStatus, Severity } from "@/types";

export const hazardIcon: Record<HazardType, LucideIcon> = {
  sign_defect: Signpost,
  pothole: Construction,
  accident: Car,
  other: TriangleAlert,
};

export function HazardIcon({ type, className }: { type: HazardType; className?: string }) {
  const Icon = hazardIcon[type];
  return <Icon className={className} />;
}

export function SeverityBadge({ level }: { level: Severity }) {
  const { tl } = useI18n();
  const m = severityMeta[level];
  return (
    <span className={cn("chip", m.pill)}>
      <span className={cn("h-1.5 w-1.5 rounded-full", m.dot)} />
      {tl(m.label)}
    </span>
  );
}

export function StatusPill({ status }: { status: ReportStatus }) {
  const { tl } = useI18n();
  const m = statusMeta[status];
  return (
    <span className={cn("chip", m.pill)}>
      <span className={cn("h-1.5 w-1.5 rounded-full", m.dot)} />
      {tl(m.label)}
    </span>
  );
}

export function HazardChip({ type }: { type: HazardType }) {
  const { tl } = useI18n();
  const m = hazardMeta[type];
  return (
    <span
      className="chip ring-1 ring-inset"
      style={{
        backgroundColor: `${m.color}14`,
        color: m.color,
        // @ts-expect-error css var for ring color
        "--tw-ring-color": `${m.color}33`,
      }}
    >
      <HazardIcon type={type} className="h-3.5 w-3.5" />
      {tl(m.label)}
    </span>
  );
}

export function SourceChip({ source }: { source: string }) {
  const { tl } = useI18n();
  return (
    <span className="chip bg-ink/[0.04] text-ink-soft ring-1 ring-inset ring-ink/10">
      {tl(sourceLabel(source))}
    </span>
  );
}

export function ModelStatusBadge({ status }: { status: ModelStatus }) {
  const { t } = useI18n();
  const map = {
    online: { label: t("common.online"), cls: "bg-sev-low/10 text-sev-low ring-sev-low/20", dot: "bg-sev-low" },
    degraded: { label: t("common.degraded"), cls: "bg-sev-med/10 text-sev-med ring-sev-med/25", dot: "bg-sev-med" },
    not_connected: {
      label: t("common.notConnected"),
      cls: "bg-ink/[0.05] text-ink-soft ring-ink/10",
      dot: "bg-ink-faint",
    },
  }[status];
  return (
    <span className={cn("chip ring-1 ring-inset", map.cls)}>
      <span className={cn("h-1.5 w-1.5 rounded-full", map.dot, status === "online" && "animate-pulse")} />
      {map.label}
    </span>
  );
}

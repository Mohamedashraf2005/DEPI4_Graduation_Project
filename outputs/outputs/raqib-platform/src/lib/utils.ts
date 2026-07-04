import type { HazardType, Lang, Localized, ReportStatus, Severity } from "@/types";

/** Tiny classNames joiner (no extra deps). */
export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

/** Resolve a Localized value for the active language. */
export function loc(value: Localized, lang: Lang): string {
  return value[lang];
}

export function clamp(n: number, min = 0, max = 100): number {
  return Math.min(max, Math.max(min, n));
}

// ── Time formatting ───────────────────────────────────────────

export function formatDateTime(iso: string, lang: Lang): string {
  const d = new Date(iso);
  return d.toLocaleString(lang === "ar" ? "ar-SA" : "en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function timeAgo(iso: string, lang: Lang): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.round(diff / 60000);
  if (mins < 1) return lang === "ar" ? "الآن" : "now";
  if (mins < 60) return lang === "ar" ? `قبل ${mins} د` : `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return lang === "ar" ? `قبل ${hrs} س` : `${hrs}h ago`;
  const days = Math.round(hrs / 24);
  return lang === "ar" ? `قبل ${days} ي` : `${days}d ago`;
}

// ── Severity metadata ─────────────────────────────────────────

export interface Meta {
  label: Localized;
  /** Solid accent color (hex). */
  color: string;
  /** Tailwind text class. */
  text: string;
  /** Tailwind soft background + text classes for pills. */
  pill: string;
  dot: string;
}

export const severityMeta: Record<Severity, Meta> = {
  low: {
    label: { ar: "منخفضة", en: "Low" },
    color: "#1A9E54",
    text: "text-sev-low",
    pill: "bg-sev-low/10 text-sev-low ring-1 ring-inset ring-sev-low/20",
    dot: "bg-sev-low",
  },
  medium: {
    label: { ar: "متوسطة", en: "Medium" },
    color: "#E0A008",
    text: "text-sev-med",
    pill: "bg-sev-med/10 text-sev-med ring-1 ring-inset ring-sev-med/25",
    dot: "bg-sev-med",
  },
  high: {
    label: { ar: "عالية", en: "High" },
    color: "#F07316",
    text: "text-sev-high",
    pill: "bg-sev-high/10 text-sev-high ring-1 ring-inset ring-sev-high/25",
    dot: "bg-sev-high",
  },
  critical: {
    label: { ar: "حرجة", en: "Critical" },
    color: "#DC2A28",
    text: "text-sev-crit",
    pill: "bg-sev-crit/10 text-sev-crit ring-1 ring-inset ring-sev-crit/25",
    dot: "bg-sev-crit",
  },
};

export const statusMeta: Record<ReportStatus, Meta> = {
  new: {
    label: { ar: "جديد", en: "New" },
    color: "#2F6FED",
    text: "text-info",
    pill: "bg-info/10 text-info ring-1 ring-inset ring-info/20",
    dot: "bg-info",
  },
  analyzing: {
    label: { ar: "قيد التحليل", en: "Analyzing" },
    color: "#12B5C4",
    text: "text-accent",
    pill: "bg-accent/10 text-accent ring-1 ring-inset ring-accent/20",
    dot: "bg-accent",
  },
  pending_dispatch: {
    label: { ar: "بانتظار الإرسال", en: "Pending dispatch" },
    color: "#E0A008",
    text: "text-sev-med",
    pill: "bg-sev-med/10 text-sev-med ring-1 ring-inset ring-sev-med/25",
    dot: "bg-sev-med",
  },
  dispatched: {
    label: { ar: "أُرسل للجهة", en: "Dispatched" },
    color: "#0E9F8E",
    text: "text-primary",
    pill: "bg-primary/10 text-primary-700 ring-1 ring-inset ring-primary/20",
    dot: "bg-primary",
  },
  acknowledged: {
    label: { ar: "تم الاستلام", en: "Acknowledged" },
    color: "#0B8273",
    text: "text-primary-700",
    pill: "bg-primary-700/10 text-primary-700 ring-1 ring-inset ring-primary-700/20",
    dot: "bg-primary-700",
  },
  resolved: {
    label: { ar: "تمت المعالجة", en: "Resolved" },
    color: "#54666D",
    text: "text-ink-soft",
    pill: "bg-ink/5 text-ink-soft ring-1 ring-inset ring-ink/10",
    dot: "bg-ink-soft",
  },
};

export const hazardMeta: Record<
  HazardType,
  { label: Localized; color: string }
> = {
  sign_defect: { label: { ar: "عيب إشارة مرور", en: "Sign defect" }, color: "#2F6FED" },
  pothole: { label: { ar: "حفرة / تلف سطح", en: "Pothole / damage" }, color: "#F07316" },
  accident: { label: { ar: "حادث مروري", en: "Accident" }, color: "#DC2A28" },
  other: { label: { ar: "أخرى", en: "Other" }, color: "#54666D" },
};

export function sourceLabel(source: string): Localized {
  switch (source) {
    case "manual":
      return { ar: "رفع يدوي", en: "Manual upload" };
    case "field_camera":
      return { ar: "كاميرا ميدانية", en: "Field camera" };
    case "citizen":
      return { ar: "بلاغ مواطن", en: "Citizen report" };
    default:
      return { ar: "غير معروف", en: "Unknown" };
  }
}

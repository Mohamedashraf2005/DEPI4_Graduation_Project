import { useState } from "react";
import { useI18n } from "@/i18n/I18nContext";
import { severityMeta } from "@/lib/utils";
import type { HazardReport } from "@/types";

const KSA_PATH =
  "M14,37 L22,28 L31,21 L46,21 L58,25 L67,31 L72,38 L70,46 L75,55 L72,64 L62,71 L48,77 L35,74 L27,65 L22,55 L16,47 Z";

const CITIES: { x: number; y: number; ar: string; en: string }[] = [
  { x: 53, y: 45, ar: "الرياض", en: "Riyadh" },
  { x: 22, y: 50, ar: "جدة", en: "Jeddah" },
  { x: 28, y: 40, ar: "المدينة", en: "Madinah" },
  { x: 67, y: 39, ar: "الدمام", en: "Dammam" },
  { x: 31, y: 65, ar: "أبها", en: "Abha" },
  { x: 25, y: 30, ar: "تبوك", en: "Tabuk" },
];

export function SaudiMap({
  reports,
  onSelect,
}: {
  reports: HazardReport[];
  onSelect?: (r: HazardReport) => void;
}) {
  const { lang, tl } = useI18n();
  const [hover, setHover] = useState<HazardReport | null>(null);

  return (
    <div className="relative aspect-[5/4] w-full overflow-hidden rounded-xl bg-gradient-to-b from-panel to-[#e6efec] bg-grid">
      <svg viewBox="0 0 90 90" className="absolute inset-0 h-full w-full" preserveAspectRatio="xMidYMid meet">
        <defs>
          <linearGradient id="land" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#0E9F8E" stopOpacity="0.14" />
            <stop offset="1" stopColor="#12B5C4" stopOpacity="0.08" />
          </linearGradient>
        </defs>
        {/* road network hints */}
        <g stroke="#0B6458" strokeOpacity="0.18" strokeWidth="0.5" fill="none">
          <path d="M53,45 L22,50 M53,45 L28,40 M53,45 L67,39 M53,45 L31,65 M28,40 L25,30" />
        </g>
        <path d={KSA_PATH} fill="url(#land)" stroke="#0B6458" strokeOpacity="0.45" strokeWidth="0.7" />
        {CITIES.map((c) => (
          <g key={c.en}>
            <circle cx={c.x} cy={c.y} r="0.9" fill="#0B6458" fillOpacity="0.6" />
            <text
              x={c.x}
              y={c.y - 1.8}
              textAnchor="middle"
              fontSize="2.4"
              fill="#54666D"
              style={{ fontWeight: 600 }}
            >
              {lang === "ar" ? c.ar : c.en}
            </text>
          </g>
        ))}
      </svg>

      {/* hazard markers as positioned buttons for hover + pulse */}
      {reports.map((r) => {
        const m = severityMeta[r.severity];
        const sz = r.severity === "critical" ? 16 : r.severity === "high" ? 13 : 10;
        return (
          <button
            key={r.id}
            onMouseEnter={() => setHover(r)}
            onMouseLeave={() => setHover((h) => (h?.id === r.id ? null : h))}
            onClick={() => onSelect?.(r)}
            className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-white transition-transform hover:scale-125"
            style={{
              left: `${r.location.x}%`,
              top: `${r.location.y}%`,
              width: sz,
              height: sz,
              backgroundColor: m.color,
              boxShadow: `0 0 0 4px ${m.color}22`,
            }}
            aria-label={tl(r.title)}
          >
            {r.severity === "critical" && (
              <span
                className="absolute inset-0 animate-ping rounded-full"
                style={{ backgroundColor: `${m.color}66` }}
              />
            )}
          </button>
        );
      })}

      {hover && (
        <div
          className="pointer-events-none absolute z-10 w-44 -translate-x-1/2 -translate-y-full rounded-xl border border-line bg-white/95 p-2.5 text-right shadow-soft"
          style={{ left: `${hover.location.x}%`, top: `calc(${hover.location.y}% - 14px)` }}
        >
          <div className="flex items-center justify-between gap-2">
            <span className="mono text-[10px] text-ink-faint">{hover.id}</span>
            <span className="text-xs font-bold" style={{ color: severityMeta[hover.severity].color }}>
              {tl(severityMeta[hover.severity].label)}
            </span>
          </div>
          <div className="mt-1 text-xs font-semibold text-ink">{tl(hover.title)}</div>
          <div className="text-[11px] text-ink-soft">{tl(hover.location.city)} · {tl(hover.location.road)}</div>
        </div>
      )}
    </div>
  );
}

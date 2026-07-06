import { ArrowDownRight, ArrowUpRight, type LucideIcon } from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export function Sparkline({
  data,
  color = "#0E9F8E",
  className,
}: {
  data: number[];
  color?: string;
  className?: string;
}) {
  const w = 96;
  const h = 30;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const span = max - min || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / span) * (h - 4) - 2;
    return [x, y] as const;
  });
  const d = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
  const area = `${d} L${w} ${h} L0 ${h} Z`;
  const id = `sp-${color.replace("#", "")}`;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className={className} aria-hidden>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity="0.28" />
          <stop offset="1" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${id})`} />
      <path d={d} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export interface KPIStatProps {
  icon: LucideIcon;
  label: string;
  value: string;
  delta?: number; // percent
  upIsGood?: boolean;
  trend?: number[];
  accent?: string;
  index?: number;
}

export function KPIStat({
  icon: Icon,
  label,
  value,
  delta,
  upIsGood = true,
  trend,
  accent = "#0E9F8E",
  index = 0,
}: KPIStatProps) {
  const up = (delta ?? 0) >= 0;
  const good = up === upIsGood;
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: index * 0.06, ease: [0.22, 0.61, 0.36, 1] }}
      className="card group relative overflow-hidden p-5"
    >
      <div
        className="pointer-events-none absolute -inset-px opacity-0 transition-opacity duration-300 group-hover:opacity-100"
        style={{ background: `radial-gradient(380px 120px at 100% 0%, ${accent}14, transparent 70%)` }}
      />
      <div className="flex items-start justify-between">
        <div
          className="grid h-10 w-10 place-items-center rounded-xl"
          style={{ backgroundColor: `${accent}14`, color: accent }}
        >
          <Icon className="h-5 w-5" />
        </div>
        {typeof delta === "number" && (
          <span
            className={cn(
              "chip text-[11px]",
              good ? "bg-sev-low/10 text-sev-low" : "bg-sev-crit/10 text-sev-crit"
            )}
          >
            {up ? <ArrowUpRight className="h-3 w-3" /> : <ArrowDownRight className="h-3 w-3" />}
            {Math.abs(delta)}%
          </span>
        )}
      </div>
      <div className="mt-4 mono text-[28px] font-semibold leading-none text-ink tnum">{value}</div>
      <div className="mt-2 flex items-end justify-between gap-2">
        <div className="text-[13px] font-medium text-ink-soft">{label}</div>
        {trend && <Sparkline data={trend} color={accent} className="opacity-90" />}
      </div>
    </motion.div>
  );
}

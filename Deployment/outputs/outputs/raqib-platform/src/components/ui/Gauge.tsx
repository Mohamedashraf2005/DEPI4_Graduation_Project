import { motion } from "framer-motion";

/** Circular progress gauge (0..100). */
export function Gauge({
  value,
  size = 132,
  stroke = 12,
  color = "#0E9F8E",
  label,
  sub,
}: {
  value: number;
  size?: number;
  stroke?: number;
  color?: string;
  label?: string;
  sub?: string;
}) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, value));
  const offset = c - (pct / 100) * c;
  return (
    <div className="relative grid place-items-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#E2EAE7" strokeWidth={stroke} />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={c}
          initial={{ strokeDashoffset: c }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1, ease: [0.22, 0.61, 0.36, 1] }}
        />
      </svg>
      <div className="absolute inset-0 grid place-content-center text-center">
        <div className="mono text-[26px] font-bold leading-none tnum" style={{ color }}>
          {Math.round(pct)}
        </div>
        {label && <div className="mt-1 text-xs font-semibold text-ink">{label}</div>}
        {sub && <div className="text-[10px] text-ink-faint">{sub}</div>}
      </div>
    </div>
  );
}

/** Thin horizontal meter used for feature-importance / confidence bars. */
export function Meter({
  value,
  color = "#0E9F8E",
  delay = 0,
}: {
  value: number;
  color?: string;
  delay?: number;
}) {
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-ink/[0.07]">
      <motion.div
        className="h-full rounded-full"
        style={{ backgroundColor: color }}
        initial={{ width: 0 }}
        animate={{ width: `${Math.max(0, Math.min(100, value))}%` }}
        transition={{ duration: 0.8, delay, ease: [0.22, 0.61, 0.36, 1] }}
      />
    </div>
  );
}

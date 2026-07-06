import type { ReactNode } from "react";
import { motion } from "framer-motion";

export function PageHeader({
  title,
  subtitle,
  icon,
  actions,
}: {
  title: string;
  subtitle?: string;
  icon?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="mb-6 flex flex-wrap items-end justify-between gap-4"
    >
      <div className="flex items-center gap-3">
        {icon && (
          <div className="grid h-11 w-11 place-items-center rounded-2xl gradient-primary text-white shadow-lift">
            {icon}
          </div>
        )}
        <div>
          <h1 className="text-[22px] font-bold tracking-tight text-ink">{title}</h1>
          {subtitle && <p className="mt-0.5 max-w-2xl text-sm text-ink-soft">{subtitle}</p>}
        </div>
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </motion.div>
  );
}

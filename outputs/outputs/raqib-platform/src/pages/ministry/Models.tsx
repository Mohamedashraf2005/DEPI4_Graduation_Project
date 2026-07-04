import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Boxes,
  Car,
  Construction,
  Cpu,
  FlaskConical,
  type LucideIcon,
  Signpost,
  Timer,
  TrendingUp,
  Zap,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { ModelStatusBadge } from "@/components/ui/Badge";
import { ENDPOINTS } from "@/api/endpoints";
import { USE_MOCK } from "@/api/client";
import { models } from "@/mock/data";
import { useI18n } from "@/i18n/I18nContext";
import type { ModelInfo, ModelKey } from "@/types";

const modelIcon: Record<ModelKey, LucideIcon> = {
  sign_defect: Signpost,
  pothole: Construction,
  accident: Car,
  risk: TrendingUp,
};

function endpointValue(key: ModelKey): string {
  if (key === "risk") return ENDPOINTS.risk;
  return ENDPOINTS.infer[key];
}

function ModelCard({ m, index }: { m: ModelInfo; index: number }) {
  const { t, tl } = useI18n();
  const navigate = useNavigate();
  const Icon = modelIcon[m.key];

  const metrics = [
    { icon: Zap, label: t("models.accuracy"), value: `${m.accuracy}`, sub: tl(m.metricLabel) },
    { icon: Timer, label: t("models.latency"), value: `${m.latencyMs}ms`, sub: tl(m.throughput) },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.07 }}
      className="card flex flex-col p-5"
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <span className="grid h-12 w-12 place-items-center rounded-2xl gradient-primary text-white shadow-lift">
            <Icon className="h-6 w-6" />
          </span>
          <div>
            <h3 className="font-bold leading-tight text-ink">{tl(m.name)}</h3>
            <span className="mt-1 inline-flex items-center gap-1 rounded-md bg-panel px-2 py-0.5 text-[10px] font-semibold text-ink-soft">
              {m.family === "cv" ? <Boxes className="h-3 w-3" /> : <Cpu className="h-3 w-3" />}
              {m.family === "cv" ? t("models.family.cv") : t("models.family.ml")}
            </span>
          </div>
        </div>
        <ModelStatusBadge status={m.status} />
      </div>

      <p className="mt-3 text-sm leading-relaxed text-ink-soft">{tl(m.task)}</p>

      {/* metrics */}
      <div className="mt-4 grid grid-cols-2 gap-2">
        {metrics.map((mt) => (
          <div key={mt.label} className="panel p-3">
            <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wide text-ink-faint">
              <mt.icon className="h-3 w-3" />
              {mt.label}
            </div>
            <div className="mono mt-1 text-lg font-bold text-ink">{mt.value}</div>
            <div className="truncate text-[10px] text-ink-faint">{mt.sub}</div>
          </div>
        ))}
      </div>

      {/* endpoint slot */}
      <div className="mono mt-4 rounded-xl border border-line bg-ink/[0.025] p-3 text-xs">
        <div className="flex items-center justify-between">
          <span className="text-[10px] uppercase tracking-wide text-ink-faint">{t("models.endpoint")}</span>
          <span className="rounded bg-sev-med/10 px-1.5 py-0.5 text-[10px] font-bold text-sev-med">
            {USE_MOCK ? "MOCK" : "LIVE"}
          </span>
        </div>
        <div className="mt-2 break-all leading-relaxed">
          <span className="text-primary-700">{m.endpointEnv}</span>
          <span className="text-ink-faint"> = </span>
          <span className="text-ink-soft">"{endpointValue(m.key)}"</span>
        </div>
        <div className="mt-1.5 text-[10px] text-ink-faint">
          {t("models.connectHint")} <code className="rounded bg-panel px-1">.env</code>
        </div>
      </div>

      <div className="mt-4 flex items-center gap-2">
        <button
          onClick={() => navigate(m.family === "cv" ? "/app/analyze" : "/app/risk")}
          className="btn-soft flex-1"
        >
          <FlaskConical className="h-4 w-4" />
          {m.family === "cv" ? t("models.test") : t("risk.predict")}
        </button>
      </div>
    </motion.div>
  );
}

export function Models() {
  const { t } = useI18n();
  return (
    <div>
      <PageHeader icon={<Boxes className="h-5 w-5" />} title={t("models.title")} subtitle={t("models.subtitle")} />
      <div className="grid gap-5 md:grid-cols-2">
        {models.map((m, i) => (
          <ModelCard key={m.key} m={m} index={i} />
        ))}
      </div>
    </div>
  );
}

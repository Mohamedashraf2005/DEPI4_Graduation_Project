import { Clock3, SlidersHorizontal, TrendingUp } from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { useI18n } from "@/i18n/I18nContext";

export function Risk() {
  const { t } = useI18n();
  return (
    <div>
      <PageHeader icon={<TrendingUp className="h-5 w-5" />} title={t("risk.title")} subtitle={t("risk.subtitle")} />

      <div className="grid gap-5 lg:grid-cols-2">
        <div className="card p-5">
          <div className="mb-4 flex items-center gap-2">
            <span className="grid h-8 w-8 place-items-center rounded-lg bg-primary/10 text-primary">
              <SlidersHorizontal className="h-4 w-4" />
            </span>
            <h2 className="text-sm font-bold text-ink">{t("risk.featuresTBD")}</h2>
          </div>
          <div className="space-y-3">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="flex items-center gap-3 rounded-xl border border-dashed border-line bg-panel/40 px-3 py-3">
                <span className="h-2 w-2 rounded-full bg-ink-faint/40" />
                <span className="h-2.5 flex-1 rounded-full bg-ink/[0.06]" />
              </div>
            ))}
          </div>
        </div>

        <div className="card grid place-items-center p-5">
          <div className="py-10 text-center">
            <span className="mx-auto grid h-24 w-24 place-items-center rounded-full border-[6px] border-dashed border-line text-ink-faint">
              <Clock3 className="h-7 w-7" />
            </span>
            <div className="mt-5 font-semibold text-ink">{t("risk.emptyTitle")}</div>
            <p className="mx-auto mt-2 max-w-xs text-sm leading-relaxed text-ink-faint">{t("risk.emptyBody")}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

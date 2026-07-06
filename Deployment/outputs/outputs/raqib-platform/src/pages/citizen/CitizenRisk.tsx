import { MapPin, Plug, Search } from "lucide-react";
import { useI18n } from "@/i18n/I18nContext";

export function CitizenRisk() {
  const { t } = useI18n();
  return (
    <div className="mx-auto w-full max-w-xl px-6 py-12">
      <h1 className="text-2xl font-bold text-ink">{t("citizen.risk.title")}</h1>
      <p className="mt-2 text-sm text-ink-soft">{t("citizen.risk.subtitle")}</p>

      <div className="mt-6 card p-5">
        {/* disabled road input — activates once the model is live */}
        <div className="flex gap-2 opacity-60">
          <div className="relative flex-1">
            <MapPin className="pointer-events-none absolute top-1/2 h-4 w-4 -translate-y-1/2 text-ink-faint ltr:left-3.5 rtl:right-3.5" />
            <input disabled placeholder={t("citizen.report.locationPh")} className="input ltr:pl-10 rtl:pr-10" />
          </div>
          <button disabled className="btn-primary px-5"><Search className="h-4 w-4" /></button>
        </div>

        <div className="mt-6 grid place-items-center py-10 text-center">
          <span className="grid h-20 w-20 place-items-center rounded-full border-[6px] border-dashed border-line text-ink-faint">
            <Plug className="h-7 w-7" />
          </span>
          <div className="mt-4 font-semibold text-ink">{t("citizen.risk.emptyTitle")}</div>
          <p className="mx-auto mt-2 max-w-sm text-sm leading-relaxed text-ink-faint">{t("citizen.risk.emptyBody")}</p>
        </div>
      </div>
    </div>
  );
}

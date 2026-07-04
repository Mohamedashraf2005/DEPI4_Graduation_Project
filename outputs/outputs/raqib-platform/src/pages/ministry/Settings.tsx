import { useState } from "react";
import { Bell, Globe, Server, ShieldCheck, SlidersHorizontal } from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { authorities } from "@/mock/reports";
import { models } from "@/mock/data";
import { ENDPOINTS } from "@/api/endpoints";
import { API_BASE, USE_MOCK } from "@/api/client";
import { useI18n } from "@/i18n/I18nContext";
import type { Lang } from "@/types";

function Toggle({ on, onChange }: { on: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!on)}
      className={"relative h-6 w-11 rounded-full transition " + (on ? "bg-primary" : "bg-ink/15")}
    >
      <span
        className={
          "absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-all " +
          (on ? "ltr:left-[22px] rtl:right-[22px]" : "ltr:left-0.5 rtl:right-0.5")
        }
      />
    </button>
  );
}

function Card({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="card p-5">
      <div className="mb-4 flex items-center gap-2">
        <span className="grid h-8 w-8 place-items-center rounded-lg bg-primary/10 text-primary">{icon}</span>
        <h2 className="text-sm font-bold text-ink">{title}</h2>
      </div>
      {children}
    </div>
  );
}

export function Settings() {
  const { t, tl, lang, setLang } = useI18n();
  const [notif, setNotif] = useState({ critical: true, dispatch: true, daily: false });
  const [auths, setAuths] = useState<Record<string, boolean>>(
    Object.fromEntries(Object.values(authorities).map((a) => [a.id, true]))
  );

  return (
    <div>
      <PageHeader icon={<SlidersHorizontal className="h-5 w-5" />} title={t("nav.settings")} subtitle={t("brand.gov")} />

      <div className="grid gap-5 lg:grid-cols-2">
        {/* Language */}
        <Card title={lang === "ar" ? "اللغة والاتجاه" : "Language & direction"} icon={<Globe className="h-4 w-4" />}>
          <div className="grid grid-cols-2 gap-2">
            {(["ar", "en"] as Lang[]).map((l) => (
              <button
                key={l}
                onClick={() => setLang(l)}
                className={
                  "rounded-xl border p-3 text-center text-sm font-semibold transition " +
                  (lang === l ? "border-primary/40 bg-primary/[0.06] text-primary-700" : "border-line text-ink-soft hover:bg-panel")
                }
              >
                {l === "ar" ? "العربية · RTL" : "English · LTR"}
              </button>
            ))}
          </div>
        </Card>

        {/* Notifications */}
        <Card title={lang === "ar" ? "التنبيهات" : "Notifications"} icon={<Bell className="h-4 w-4" />}>
          <div className="space-y-1">
            {[
              { k: "critical", ar: "تنبيه فوري للبلاغات الحرجة", en: "Instant alert for critical reports" },
              { k: "dispatch", ar: "إشعار عند إرسال بلاغ للجهة", en: "Notify on dispatch to authority" },
              { k: "daily", ar: "ملخّص يومي", en: "Daily summary digest" },
            ].map((row) => (
              <div key={row.k} className="flex items-center justify-between py-2">
                <span className="text-sm text-ink-soft">{lang === "ar" ? row.ar : row.en}</span>
                <Toggle
                  on={notif[row.k as keyof typeof notif]}
                  onChange={(v) => setNotif((p) => ({ ...p, [row.k]: v }))}
                />
              </div>
            ))}
          </div>
        </Card>

        {/* Authorities routing */}
        <Card title={lang === "ar" ? "الجهات المختصة والتوجيه" : "Authorities & routing"} icon={<ShieldCheck className="h-4 w-4" />}>
          <div className="space-y-1">
            {Object.values(authorities).map((a) => (
              <div key={a.id} className="flex items-center justify-between py-2">
                <div>
                  <div className="text-sm font-medium text-ink">{tl(a.name)}</div>
                  <div className="text-xs text-ink-faint">{tl(a.region)}</div>
                </div>
                <Toggle on={auths[a.id]} onChange={(v) => setAuths((p) => ({ ...p, [a.id]: v }))} />
              </div>
            ))}
          </div>
        </Card>

        {/* Integration status */}
        <Card title={lang === "ar" ? "حالة الربط (API)" : "Integration (API)"} icon={<Server className="h-4 w-4" />}>
          <div className="mono space-y-2 rounded-xl border border-line bg-ink/[0.025] p-3 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-ink-faint">VITE_API_BASE_URL</span>
              <span className="text-ink-soft">"{API_BASE || "—"}"</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-ink-faint">VITE_USE_MOCK</span>
              <span className={USE_MOCK ? "text-sev-med" : "text-sev-low"}>{String(USE_MOCK)}</span>
            </div>
            <div className="border-t border-line pt-2">
              {models.map((m) => (
                <div key={m.key} className="flex items-center justify-between py-0.5">
                  <span className="text-primary-700">{m.endpointEnv}</span>
                  <span className="h-1.5 w-1.5 rounded-full bg-ink/20" />
                </div>
              ))}
              <div className="flex items-center justify-between py-0.5">
                <span className="text-primary-700">VITE_RAG_QUERY_URL</span>
                <span className="text-ink-soft">"{ENDPOINTS.rag}"</span>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

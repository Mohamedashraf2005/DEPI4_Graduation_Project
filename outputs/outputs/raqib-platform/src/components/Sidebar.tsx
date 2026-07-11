import { NavLink } from "react-router-dom";
import { ClipboardList, type LucideIcon, ScanLine, TrendingUp, Users } from "lucide-react";
import { Logo } from "./Logo";
import { useI18n } from "@/i18n/I18nContext";
import { cn } from "@/lib/utils";

const items: { to: string; icon: LucideIcon; key: string; end?: boolean }[] = [
  { to: "/app", icon: ScanLine, key: "nav.analyze", end: true },
  { to: "/app/reports", icon: ClipboardList, key: "nav.reports" },
  { to: "/app/risk", icon: TrendingUp, key: "nav.risk" },
  { to: "/app/heatmap", icon: TrendingUp, key: "nav.heatmap" },
  { to: "/app/chatbot", icon: TrendingUp, key: "nav.chatbot" },
  { to: "/app/roadriskanalyzer", icon: TrendingUp, key: "nav.roadriskanalyzer" },
];
//raqib-platform\src\pages\ministry\chatbot.ts
export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const { t } = useI18n();
  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-line bg-surface/80 ltr:border-r rtl:border-l">
      <div className="px-5 py-5">
        <NavLink to="/" onClick={onNavigate}>
          <Logo size={38} />
        </NavLink>
      </div>

      <nav className="flex-1 px-3 py-2">
        <div className="space-y-1">
          {items.map((it) => (
            <NavLink
              key={it.to}
              to={it.to}
              end={it.end}
              onClick={onNavigate}
              className={({ isActive }) =>
                cn(
                  "group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-all",
                  isActive ? "bg-primary/10 text-primary-700" : "text-ink-soft hover:bg-panel hover:text-ink"
                )
              }
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <span className="absolute inset-y-1.5 w-1 rounded-full bg-primary ltr:left-0 rtl:right-0" />
                  )}
                  <it.icon className={cn("h-[18px] w-[18px]", isActive && "text-primary")} />
                  <span>{t(it.key)}</span>
                </>
              )}
            </NavLink>
          ))}
        </div>
      </nav>

      <div className="border-t border-line p-3">
        <NavLink
          to="/citizen"
          onClick={onNavigate}
          className="flex items-center gap-3 rounded-xl bg-panel px-3 py-2.5 text-sm font-medium text-ink-soft transition hover:bg-primary/10 hover:text-primary-700"
        >
          <Users className="h-[18px] w-[18px]" />
          {t("nav.citizen")}
        </NavLink>
      </div>
    </aside>
  );
}

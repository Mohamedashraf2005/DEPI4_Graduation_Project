import { Link, NavLink, Outlet } from "react-router-dom";
import { Building2, Home } from "lucide-react";
import { Logo } from "./Logo";
import { LanguageToggle } from "./Topbar";
import { HelpButton } from "./Onboarding";
import { useI18n } from "@/i18n/I18nContext";
import { cn } from "@/lib/utils";

export function CitizenLayout() {
  const { t } = useI18n();
  const links = [
    { to: "/citizen/report", key: "citizen.report.title" },
    { to: "/citizen/track", key: "track.title" },
    { to: "/citizen/risk", key: "citizen.risk.title" },
  ];
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-30 border-b border-line glass">
        <div className="mx-auto flex h-16 w-full max-w-6xl items-center justify-between gap-2 px-4 sm:px-6">
          <Link to="/citizen" className="flex items-center gap-2.5">
            <Logo size={36} />
            <span className="hidden rounded-full bg-primary/10 px-2.5 py-1 text-xs font-semibold text-primary-700 sm:inline">
              {t("citizen.brand")}
            </span>
          </Link>

          <nav className="hidden items-center gap-1 lg:flex">
            {links.map((l) => (
              <NavLink
                key={l.to}
                to={l.to}
                className={({ isActive }) =>
                  cn(
                    "rounded-lg px-3 py-2 text-sm font-medium transition",
                    isActive ? "bg-primary/10 text-primary-700" : "text-ink-soft hover:text-ink"
                  )
                }
              >
                {t(l.key)}
              </NavLink>
            ))}
          </nav>

          <div className="flex items-center gap-2">
            <HelpButton />
            <Link
              to="/"
              className="inline-flex items-center gap-1.5 rounded-xl border border-line bg-white px-3 py-2 text-xs font-semibold text-ink-soft transition hover:bg-panel hover:text-ink"
            >
              <Home className="h-4 w-4" />
              <span className="hidden sm:inline">{t("nav.home")}</span>
            </Link>
            <LanguageToggle />
            <Link
              to="/app"
              className="hidden items-center gap-1.5 rounded-xl border border-line bg-white px-3 py-2 text-xs font-semibold text-ink-soft transition hover:text-ink xl:inline-flex"
            >
              <Building2 className="h-4 w-4" />
              {t("citizen.toMinistry")}
            </Link>
          </div>
        </div>
      </header>

      <main className="flex-1">
        <Outlet />
      </main>

      <footer className="border-t border-line py-6">
        <div className="mx-auto flex w-full max-w-6xl flex-col items-center justify-between gap-2 px-6 text-xs text-ink-faint sm:flex-row">
          <span>© 2026 {t("brand.name")} · {t("brand.tagline")}</span>
          <span>{t("brand.gov")}</span>
        </div>
      </footer>
    </div>
  );
}

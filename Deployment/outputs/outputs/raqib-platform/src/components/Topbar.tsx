import { Link } from "react-router-dom";
import { Home, Languages, Menu } from "lucide-react";
import { HelpButton } from "./Onboarding";
import { useI18n } from "@/i18n/I18nContext";

export function LanguageToggle() {
  const { t, toggleLang } = useI18n();
  return (
    <button
      onClick={toggleLang}
      className="inline-flex items-center gap-1.5 rounded-xl border border-line bg-white px-3 py-2 text-xs font-semibold text-ink-soft transition hover:bg-panel hover:text-ink"
    >
      <Languages className="h-4 w-4" />
      {t("lang.toggle")}
    </button>
  );
}

export function Topbar({ onMenu }: { onMenu: () => void }) {
  const { t } = useI18n();
  return (
    <header className="sticky top-0 z-30 border-b border-line glass">
      <div className="flex h-16 items-center gap-2 px-4 sm:px-6">
        <button
          onClick={onMenu}
          className="grid h-10 w-10 place-items-center rounded-xl border border-line bg-white text-ink-soft lg:hidden"
          aria-label="menu"
        >
          <Menu className="h-5 w-5" />
        </button>
        <div className="hidden text-sm font-semibold text-ink sm:block">{t("nav.ministry")}</div>
        <div className="flex-1" />
        <HelpButton />
        <Link
          to="/"
          className="inline-flex items-center gap-1.5 rounded-xl border border-line bg-white px-3 py-2 text-xs font-semibold text-ink-soft transition hover:bg-panel hover:text-ink"
        >
          <Home className="h-4 w-4" />
          <span className="hidden sm:inline">{t("nav.home")}</span>
        </Link>
        <LanguageToggle />
      </div>
    </header>
  );
}

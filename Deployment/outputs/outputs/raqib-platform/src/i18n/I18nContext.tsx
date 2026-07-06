import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { Lang, Localized } from "@/types";
import { ar } from "./ar";
import { en } from "./en";

type Dict = Record<string, string>;
const dictionaries: Record<Lang, Dict> = { ar, en };

interface I18nValue {
  lang: Lang;
  dir: "rtl" | "ltr";
  t: (key: string) => string;
  /** Resolve a {ar,en} object for the active language. */
  tl: (value: Localized) => string;
  setLang: (l: Lang) => void;
  toggleLang: () => void;
}

const I18nContext = createContext<I18nValue | null>(null);

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>("ar");
  const dir = lang === "ar" ? "rtl" : "ltr";

  useEffect(() => {
    const root = document.documentElement;
    root.lang = lang;
    root.dir = dir;
  }, [lang, dir]);

  const value = useMemo<I18nValue>(
    () => ({
      lang,
      dir,
      t: (key) => dictionaries[lang][key] ?? dictionaries.en[key] ?? key,
      tl: (v) => v[lang],
      setLang,
      toggleLang: () => setLang((p) => (p === "ar" ? "en" : "ar")),
    }),
    [lang, dir]
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nValue {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("useI18n must be used within I18nProvider");
  return ctx;
}

import { useI18n } from "@/i18n/I18nContext";
import { cn } from "@/lib/utils";

export function LogoMark({ size = 40, className }: { size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      className={cn("shrink-0", className)}
      aria-hidden
    >
      <defs>
        <linearGradient id="rq-mark" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
          <stop stopColor="#16B8A4" />
          <stop offset="1" stopColor="#0B5F55" />
        </linearGradient>
      </defs>
      <rect width="48" height="48" rx="13" fill="url(#rq-mark)" />
      {/* perspective road */}
      <path d="M15 41 L21.5 16 L26.5 16 L33 41 Z" fill="white" fillOpacity="0.18" />
      <line
        x1="24"
        y1="39"
        x2="24"
        y2="18"
        stroke="white"
        strokeWidth="2.2"
        strokeDasharray="2.4 4.2"
        strokeLinecap="round"
      />
      {/* scan horizon */}
      <circle cx="24" cy="14.5" r="3.4" fill="white" />
      <circle cx="24" cy="14.5" r="6.4" stroke="white" strokeOpacity="0.5" strokeWidth="1.4" />
    </svg>
  );
}

export function Logo({
  size = 40,
  showText = true,
  onDark = false,
}: {
  size?: number;
  showText?: boolean;
  onDark?: boolean;
}) {
  const { t } = useI18n();
  return (
    <div className="flex items-center gap-2.5">
      <LogoMark size={size} />
      {showText && (
        <div className="leading-none">
          <div
            className={cn(
              "text-[19px] font-bold tracking-tight",
              onDark ? "text-white" : "text-ink"
            )}
          >
            {t("brand.name")}
          </div>
          <div
            className={cn(
              "mt-1 text-[10px] font-medium",
              onDark ? "text-white/60" : "text-ink-faint"
            )}
          >
            {t("brand.tagline")}
          </div>
        </div>
      )}
    </div>
  );
}

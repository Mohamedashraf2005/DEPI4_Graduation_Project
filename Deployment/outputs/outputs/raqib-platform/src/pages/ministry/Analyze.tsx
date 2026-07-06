import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { ImageIcon, Loader2, ScanLine, Sparkles, UploadCloud, X } from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { HazardIcon } from "@/components/ui/Badge";
import { type AnalyzeResult, runInference } from "@/api/models";
import { hazardMeta, severityMeta } from "@/lib/utils";
import { useI18n } from "@/i18n/I18nContext";
import type { HazardType } from "@/types";

const detects: { key: HazardType; label: string }[] = [
  { key: "sign_defect", label: "analyze.cv1" },
  { key: "pothole", label: "analyze.cv2" },
  { key: "accident", label: "analyze.cv3" },
];

export function Analyze() {
  const { t, tl } = useI18n();
  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<AnalyzeResult | null>(null);

  function pick(f?: File) {
    if (!f) return;
    setFile(f);
    setResult(null);
    setPreview(f.type.startsWith("image") ? URL.createObjectURL(f) : null);
  }

  async function run() {
    if (!file) return;
    setRunning(true);
    setResult(null);
    try {
      const r = await runInference("pothole", file);
      setResult(r);
    } catch {
      setResult(null);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div>
      <PageHeader icon={<ScanLine className="h-5 w-5" />} title={t("analyze.title")} subtitle={t("analyze.subtitle")} />

      <div className="mb-5 card p-4">
        <div className="mb-3 text-xs font-semibold text-ink-faint">{t("analyze.detects")}</div>
        <div className="grid gap-2 sm:grid-cols-3">
          {detects.map((d) => (
            <div key={d.key} className="flex items-center gap-2.5 rounded-xl bg-panel px-3 py-2.5">
              <span className="grid h-8 w-8 place-items-center rounded-lg" style={{ backgroundColor: `${hazardMeta[d.key].color}18`, color: hazardMeta[d.key].color }}>
                <HazardIcon type={d.key} className="h-4 w-4" />
              </span>
              <span className="text-sm font-medium text-ink">{t(d.label)}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <div className="card p-5">
          <input ref={fileRef} type="file" accept="image/*,video/*" className="hidden" onChange={(e) => pick(e.target.files?.[0])} />
          {!preview && !file ? (
            <button
              onClick={() => fileRef.current?.click()}
              className="group flex w-full flex-col items-center justify-center rounded-2xl border-2 border-dashed border-line bg-panel/50 px-6 py-16 text-center transition hover:border-primary/50 hover:bg-primary/[0.03]"
            >
              <span className="grid h-14 w-14 place-items-center rounded-2xl bg-primary/10 text-primary transition group-hover:scale-105">
                <UploadCloud className="h-7 w-7" />
              </span>
              <div className="mt-4 font-semibold text-ink">{t("analyze.drop")}</div>
              <div className="mt-1 text-xs text-ink-faint">{t("analyze.hint")}</div>
            </button>
          ) : (
            <div className="relative overflow-hidden rounded-2xl border border-line bg-ink/[0.02]">
              <div className="relative aspect-video">
                {preview ? (
                  <img src={preview} alt="" className="absolute inset-0 h-full w-full object-cover" />
                ) : (
                  <div className="absolute inset-0 grid place-items-center bg-gradient-to-b from-[#cfe7e2] to-[#46524f] text-white/80">
                    <ImageIcon className="h-10 w-10" />
                  </div>
                )}
                {running && <div className="absolute inset-x-0 top-0 h-24 scanline animate-scan" />}
                {result &&
                  result.detections.map((d, i) => {
                    const c = severityMeta[result.severity].color;
                    return (
                      <div key={i} className="absolute rounded-md border-2" style={{ left: `${d.x * 100}%`, top: `${d.y * 100}%`, width: `${d.w * 100}%`, height: `${d.h * 100}%`, borderColor: c, boxShadow: `0 0 0 3px ${c}22` }}>
                        <span className="absolute -top-6 whitespace-nowrap rounded px-1.5 py-0.5 text-[10px] font-bold text-white ltr:left-0 rtl:right-0" style={{ backgroundColor: c }}>
                          {tl(d.label)} · {Math.round(d.confidence * 100)}%
                        </span>
                      </div>
                    );
                  })}
              </div>
              <button onClick={() => { setFile(null); setPreview(null); setResult(null); }} className="absolute top-3 grid h-8 w-8 place-items-center rounded-lg bg-white/90 text-ink-soft shadow-sm ltr:right-3 rtl:left-3">
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          <button onClick={run} disabled={!file || running} className="btn-primary mt-4 w-full py-3">
            {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <ScanLine className="h-4 w-4" />}
            {t("analyze.run")}
          </button>
        </div>

        <div className="card grid place-items-center p-5">
          {result ? (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="w-full">
              <div className="flex items-center gap-3 rounded-xl bg-panel p-4">
                <span className="grid h-11 w-11 place-items-center rounded-xl" style={{ backgroundColor: `${hazardMeta[result.type].color}18`, color: hazardMeta[result.type].color }}>
                  <HazardIcon type={result.type} className="h-5 w-5" />
                </span>
                <div>
                  <div className="font-bold text-ink">{tl(hazardMeta[result.type].label)}</div>
                  <div className="text-xs text-ink-soft">{Math.round(result.confidence * 100)}%</div>
                </div>
              </div>
              <p className="mt-4 text-sm leading-relaxed text-ink-soft">{tl(result.description)}</p>
            </motion.div>
          ) : (
            <div className="py-12 text-center">
              <span className="mx-auto grid h-16 w-16 place-items-center rounded-2xl bg-panel text-ink-faint">
                <Sparkles className="h-7 w-7" />
              </span>
              <div className="mt-4 font-semibold text-ink">{t("analyze.emptyTitle")}</div>
              <p className="mx-auto mt-2 max-w-xs text-sm leading-relaxed text-ink-faint">{t("analyze.emptyBody")}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { CheckCircle2, Loader2, MapPin, Search, UploadCloud, X } from "lucide-react";
import { submitCitizen } from "@/api/citizen";
import { useI18n } from "@/i18n/I18nContext";

type Stage = "form" | "submitting" | "success";

export function CitizenReport() {
  const { t } = useI18n();
  const fileRef = useRef<HTMLInputElement>(null);
  const [stage, setStage] = useState<Stage>("form");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [location, setLocation] = useState("");
  const [desc, setDesc] = useState("");
  const [reference, setReference] = useState("");

  function pick(f?: File) {
    if (!f) return;
    setFile(f);
    setPreview(f.type.startsWith("image") ? URL.createObjectURL(f) : null);
  }

  async function submit() {
    setStage("submitting");
    const res = await submitCitizen({ description: desc, location, file: file ?? undefined });
    setReference(res.ref);
    setStage("success");
  }

  function reset() {
    setStage("form");
    setFile(null);
    setPreview(null);
    setLocation("");
    setDesc("");
    setReference("");
  }

  return (
    <div className="mx-auto w-full max-w-xl px-6 py-12">
      {stage !== "success" ? (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="card p-6">
          <h1 className="text-xl font-bold text-ink">{t("citizen.report.title")}</h1>

          <input ref={fileRef} type="file" accept="image/*,video/*" className="hidden" onChange={(e) => pick(e.target.files?.[0])} />

          <div className="mt-5">
            <label className="label">{t("citizen.report.upload")}</label>
            {!preview && !file ? (
              <button
                onClick={() => fileRef.current?.click()}
                className="group flex w-full flex-col items-center justify-center rounded-2xl border-2 border-dashed border-line bg-panel/50 px-6 py-12 text-center transition hover:border-primary/50 hover:bg-primary/[0.03]"
              >
                <span className="grid h-12 w-12 place-items-center rounded-2xl bg-primary/10 text-primary"><UploadCloud className="h-6 w-6" /></span>
                <div className="mt-3 text-sm font-semibold text-ink">{t("analyze.drop")}</div>
              </button>
            ) : (
              <div className="relative overflow-hidden rounded-2xl border border-line">
                <div className="aspect-video">
                  {preview ? (
                    <img src={preview} alt="" className="h-full w-full object-cover" />
                  ) : (
                    <div className="grid h-full place-items-center bg-gradient-to-b from-[#cfe7e2] to-[#46524f] text-white"><UploadCloud className="h-8 w-8" /></div>
                  )}
                </div>
                <button onClick={() => { setFile(null); setPreview(null); }} className="absolute top-2 grid h-8 w-8 place-items-center rounded-lg bg-white/90 text-ink-soft ltr:right-2 rtl:left-2"><X className="h-4 w-4" /></button>
              </div>
            )}
          </div>

          <div className="mt-4">
            <label className="label">{t("citizen.report.location")}</label>
            <div className="relative">
              <MapPin className="pointer-events-none absolute top-1/2 h-4 w-4 -translate-y-1/2 text-ink-faint ltr:left-3.5 rtl:right-3.5" />
              <input value={location} onChange={(e) => setLocation(e.target.value)} placeholder={t("citizen.report.locationPh")} className="input ltr:pl-10 rtl:pr-10" />
            </div>
          </div>

          <div className="mt-4">
            <label className="label">{t("citizen.report.desc")}</label>
            <textarea value={desc} onChange={(e) => setDesc(e.target.value)} rows={3} placeholder={t("citizen.report.descPh")} className="input resize-none" />
          </div>

          <p className="mt-3 text-xs text-ink-faint">{t("citizen.report.note")}</p>

          <button onClick={submit} disabled={stage === "submitting" || (!file && !location)} className="btn-primary mt-5 w-full py-3.5">
            {stage === "submitting" ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />}
            {t("citizen.report.submit")}
          </button>
        </motion.div>
      ) : (
        <motion.div initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} className="card p-8 text-center">
          <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", stiffness: 260, damping: 18 }} className="mx-auto grid h-16 w-16 place-items-center rounded-full bg-sev-low/15 text-sev-low">
            <CheckCircle2 className="h-9 w-9" />
          </motion.span>
          <h1 className="mt-4 text-xl font-bold text-ink">{t("citizen.report.successTitle")}</h1>
          <p className="mx-auto mt-2 max-w-sm text-sm text-ink-soft">{t("citizen.report.successBody")}</p>
          <div className="mx-auto mt-5 inline-flex flex-col items-center rounded-2xl border border-line bg-panel px-6 py-4">
            <span className="text-xs text-ink-faint">{t("citizen.report.ref")}</span>
            <span className="mono mt-1 text-lg font-bold tracking-wider text-primary-700">{reference}</span>
          </div>
          <div className="mt-6 flex justify-center gap-2">
            <Link to="/citizen/track" className="btn-primary">
              <Search className="h-4 w-4" />
              {t("citizen.report.track")}
            </Link>
            <button onClick={reset} className="btn-ghost">{t("citizen.report.another")}</button>
          </div>
        </motion.div>
      )}
    </div>
  );
}

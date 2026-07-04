import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { ImageIcon, Loader2, ScanLine, Sparkles, UploadCloud, X, AlertTriangle, TrafficCone, Construction, CheckCircle } from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { HazardIcon } from "@/components/ui/Badge";
import { hazardMeta } from "@/lib/utils";
import { useI18n } from "@/i18n/I18nContext";
import type { HazardType } from "@/types";

const detects: { key: HazardType; label: string }[] = [
  { key: "sign_defect", label: "analyze.cv1" },
  { key: "pothole", label: "analyze.cv2" },
  { key: "accident", label: "analyze.cv3" },
];

export function Analyze() {
  const { t, lang } = useI18n();
  const isArabic = lang === "ar";
  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  
  // وضع الفحص المختار
  const [selectedMode, setSelectedMode] = useState<"all" | HazardType>("all");
  
  // تجميع النتائج من الخوادم السحابية
  const [results, setResults] = useState<{ accidents?: any; traffic?: any; potholes?: any } | null>(null);

  // دالة ضغط الصور لتسريع الرفع إلى Hugging Face
  const compressImage = (imageFile: File): Promise<File> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.src = URL.createObjectURL(imageFile);
      img.onload = () => {
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        if (!ctx) return resolve(imageFile);

        const MAX_WIDTH = 640;
        let width = img.width;
        let height = img.height;

        if (width > height) {
          if (width > MAX_WIDTH) {
            height *= MAX_WIDTH / width;
            width = MAX_WIDTH;
          }
        } else {
          if (height > MAX_WIDTH) {
            width *= MAX_WIDTH / height;
            height = MAX_WIDTH;
          }
        }

        canvas.width = width;
        canvas.height = height;
        ctx.drawImage(img, 0, 0, width, height);

        canvas.toBlob(
          (blob) => {
            if (blob) {
              const compressedFile = new File([blob], imageFile.name, {
                type: "image/jpeg",
                lastModified: Date.now(),
              });
              resolve(compressedFile);
            } else {
              resolve(imageFile);
            }
          },
          "image/jpeg",
          0.8
        );
      };
      img.onerror = (err) => reject(err);
    });
  };

  async function pick(f?: File) {
    if (!f) return;
    setFile(f);
    setResults(null);
    setApiError(null);
    setPreview(f.type.startsWith("image") ? URL.createObjectURL(f) : null);
  }

  function clear() {
    setFile(null);
    setPreview(null);
    setResults(null);
    setApiError(null);
  }

  async function run() {
    if (!file) return;
    setRunning(true);
    setResults(null);
    setApiError(null);

    let fileToSend = file;
    if (file.type.startsWith("image")) {
      try {
        fileToSend = await compressImage(file);
      } catch (e) {
        console.warn("Compression failed, uploading original image instead.", e);
      }
    }

    const formData = new FormData();
    formData.append("file", fileToSend);

    const fetchAccident = selectedMode === "all" || selectedMode === "accident";
    const fetchTraffic = selectedMode === "all" || selectedMode === "sign_defect";
    const fetchPothole = selectedMode === "all" || selectedMode === "pothole";

    const promises = [];

    if (fetchAccident) {
      const url = import.meta.env.VITE_MODEL_ACCIDENT_URL || "https://mohamedachrvf-raqib-accident-api.hf.space/predict";
      promises.push(fetch(url, { method: "POST", body: formData }).then(r => r.json()));
    } else {
      promises.push(Promise.resolve(null));
    }

    if (fetchTraffic) {
      const url = import.meta.env.VITE_MODEL_SIGN_DEFECT_URL || "https://mohamedachrvf-raqib-traffic-sign-api.hf.space/predict";
      promises.push(fetch(url, { method: "POST", body: formData }).then(r => r.json()));
    } else {
      promises.push(Promise.resolve(null));
    }

    if (fetchPothole) {
      const url = import.meta.env.VITE_MODEL_POTHOLE_URL || "https://mohamedachrvf-raqib-pot-hole-api.hf.space/predict";
      promises.push(fetch(url, { method: "POST", body: formData }).then(r => r.json()));
    } else {
      promises.push(Promise.resolve(null));
    }

    try {
      const [resAccident, resTraffic, resPothole] = await Promise.all(promises);
      setResults({
        accidents: resAccident,
        traffic: resTraffic,
        potholes: resPothole,
      });
    } catch (error) {
      console.error("Cloud connection failed", error);
      setApiError(isArabic ? "فشل الاتصال بذكاء رقيب السحابي. يرجى التحقق من تفعيل خوادم Hugging Face." : "Failed to reach Raqib Cloud AI. Make sure Hugging Face Spaces are active.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8" dir={isArabic ? "rtl" : "ltr"}>
      <PageHeader title={t("nav.analyze")} subtitle={t("analyze.desc")} icon={ScanLine} />

      {apiError && (
        <div className="mb-5 flex items-center gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm font-semibold text-red-700">
          <AlertTriangle className="h-5 w-5 shrink-0" />
          <span>{apiError}</span>
        </div>
      )}

      <div className="mb-5 card p-4">
        <div className="mb-3 text-sm font-bold text-ink">{isArabic ? "حدد نوع الفحص المطلوب:" : "Select Analysis Focus:"}</div>
        <div className="grid gap-2 sm:grid-cols-4">
          <button
            onClick={() => setSelectedMode("all")}
            className={`flex items-center justify-center gap-2 rounded-xl border px-3 py-2.5 transition-all text-sm font-semibold ${
              selectedMode === "all" ? "border-primary bg-primary/10 text-primary-700" : "border-line bg-panel text-ink-soft hover:bg-panel/80"
            }`}
          >
            <CheckCircle className="h-4 w-4" />
            {t("common.all")}
          </button>
          {detects.map((d) => (
            <button
              key={d.key}
              onClick={() => setSelectedMode(d.key)}
              className={`flex items-center gap-2.5 rounded-xl border px-3 py-2 transition-all text-sm font-semibold ${
                selectedMode === d.key ? "border-primary bg-primary/10 text-primary-700" : "border-line bg-panel text-ink hover:bg-panel/80"
              }`}
            >
              <span className="grid h-8 w-8 place-items-center rounded-lg" style={{ backgroundColor: `${hazardMeta[d.key].color}18`, color: hazardMeta[d.key].color }}>
                <HazardIcon type={d.key} className="h-4 w-4" />
              </span>
              <span>{t(d.label)}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <div className="card p-5 flex flex-col justify-between">
          <div>
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
                    <img src={preview} alt="preview" className="absolute inset-0 h-full w-full object-cover" />
                  ) : (
                    <div className="absolute inset-0 grid place-items-center bg-gradient-to-b from-[#cfe7e2] to-[#46524f] text-white/80">
                      <ImageIcon className="h-10 w-10" />
                    </div>
                  )}
                  {running && <div className="absolute inset-x-0 top-0 h-24 scanline animate-scan" />}
                </div>
                <button onClick={clear} className="absolute top-3 grid h-8 w-8 place-items-center rounded-lg bg-white/90 text-ink-soft shadow-sm ltr:right-3 rtl:left-3 transition hover:bg-white hover:text-red-500">
                  <X className="h-4 w-4" />
                </button>
              </div>
            )}
          </div>

          <button onClick={run} disabled={!file || running} className="btn-primary mt-4 w-full py-3">
            {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <ScanLine className="h-4 w-4" />}
            {running ? (isArabic ? "جاري كشف وتحليل الأضرار..." : "Analyzing road health...") : t("analyze.run")}
          </button>
        </div>

        <div className="card flex flex-col p-5 min-h-[350px]">
          {results ? (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="w-full space-y-4">
              <h3 className="text-base font-bold text-ink border-b border-line pb-3">
                {isArabic ? "نتائج الكشف الذكي الموحدة:" : "Unified Intelligent Detections:"}
              </h3>

              {/* 1. كارت رصد الحوادث (8001) */}
              {results.accidents && (
                <div className="rounded-xl border border-red-100 bg-red-50/50 p-4">
                  <h4 className="mb-2.5 flex items-center gap-2 font-bold text-red-700 text-sm">
                    <AlertTriangle className="h-4 w-4" />
                    {isArabic ? "رصد الحوادث والطرق المحصورة" : "Accident and Collision Detection"}
                  </h4>
                  {results.accidents.detections?.length > 0 ? (
                    <div className="space-y-2">
                      {results.accidents.detections.map((d: any, i: number) => (
                        <div key={i} className="flex items-center justify-between rounded-lg border border-red-100 bg-white p-3 shadow-sm text-sm">
                          <span className="font-bold text-ink capitalize">
                            {d.class_name.toLowerCase() === "accident" 
                              ? (isArabic ? "حادث مروري" : "Traffic Accident") 
                              : d.class_name.toLowerCase() === "normal"
                                ? (isArabic ? " لا يوجد حادث مروري" : "Normal / No Accident")
                                : d.class_name}
                          </span>
                          <span className="font-bold text-red-600 font-mono">{Math.round(d.confidence * 100)}%</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-xs font-medium text-red-600/70">
                      {isArabic ? "لم يتم الكشف عن أية حوادث مرورية في هذه الصورة." : "No traffic accidents detected."}
                    </div>
                  )}
                </div>
              )}

              {/* 2. كارت رصد تلف العلامات (8002) */}
              {results.traffic && (
                <div className="rounded-xl border border-blue-100 bg-blue-50/50 p-4">
                  <h4 className="mb-2.5 flex items-center gap-2 font-bold text-blue-700 text-sm">
                    <TrafficCone className="h-4 w-4" />
                    {isArabic ? "حالة وعيوب العلامات الإرشادية" : "Traffic Signs & Infrastructure Defects"}
                  </h4>
                  {results.traffic.detections?.length > 0 ? (
                    <div className="space-y-2">
                      {results.traffic.detections.map((d: any, i: number) => (
                        <div key={i} className="flex items-center justify-between rounded-lg border border-blue-100 bg-white p-3 shadow-sm text-sm">
                          <span className="font-bold text-ink capitalize">
                            {d.class_name === "damaged" ? (isArabic ? "لوحة تالفة" : "Damaged Sign") : d.class_name}
                          </span>
                          <span className="font-bold text-blue-600 font-mono">{Math.round(d.confidence * 100)}%</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-xs font-medium text-blue-600/70">
                      {isArabic ? "العلامات الإرشادية تبدو بحالة جيدة." : "No traffic sign defects found."}
                    </div>
                  )}
                </div>
              )}

              {/* 3. كارت رصد عيوب رصف الطرق والحفر (8003) */}
              {results.potholes && (
                <div className="rounded-xl border border-amber-100 bg-amber-50/50 p-4">
                  <h4 className="mb-2.5 flex items-center gap-2 font-bold text-amber-800 text-sm">
                    <Construction className="h-4 w-4" />
                    {isArabic ? "الحفر وعيوب رصف الأسفلت" : "Potholes & Asphalt Degradation"}
                  </h4>
                  {results.potholes.detections?.length > 0 ? (
                    <div className="space-y-2">
                      {results.potholes.detections.map((d: any, i: number) => (
                        <div key={i} className="flex items-center justify-between rounded-lg border border-amber-100 bg-white p-3 shadow-sm text-sm">
                          <span className="font-bold text-ink capitalize">
                            {d.class_name === "pothole" ? (isArabic ? "حفرة أسفلتية عميقة" : "Asphalt Pothole") : d.class_name}
                          </span>
                          <span className="font-bold text-amber-700 font-mono">{Math.round(d.confidence * 100)}%</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-xs font-medium text-amber-700/70">
                      {isArabic ? "لم يتم رصد حفر أسفلتية أو تشققات وعرة." : "No road cracks or potholes detected."}
                    </div>
                  )}
                </div>
              )}
            </motion.div>
          ) : (
            <div className="py-12 text-center my-auto">
              <span className="mx-auto grid h-16 w-16 place-items-center rounded-2xl bg-panel text-ink-faint">
                <Sparkles className="h-7 w-7 animate-pulse" />
              </span>
              <div className="mt-4 font-semibold text-ink">{t("analyze.emptyTitle")}</div>
              <p className="mx-auto mt-2 max-w-xs text-sm leading-relaxed text-ink-soft">
                {isArabic ? "قم برفع صورة الطريق وتفعيل الفحص لبدء رصد المخاطر الذكي." : "Upload road media and launch detection to scan for dynamic risks."}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
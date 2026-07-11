import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Car,
  Clock,
  Map,
  CloudRain,
  Sun,
  Navigation,
  GitMerge,
  Calendar,
  Layers,
  Cpu,
  Loader2,
  AlertTriangle,
  Gauge,
  Info,
  TrendingUp,
  AlertOctagon
} from "lucide-react";

type RiskFormData = {
  Road_surface_type: string;
  Road_surface_conditions: string;
  Light_conditions: string;
  Weather_conditions: string;
  Road_allignment: string;
  Lanes_or_Medians: string;
  Types_of_Junction: string;
  Number_of_vehicles_involved: number;
  Hour: number;
  Day_of_week: string;
};

// ============================================================
// قواميس الترجمة والتحويل (Mapping) لضمان توافق الـ API الإنجليزي
// ============================================================
const TRANSLATIONS = {
  Road_surface_type: {
    "Asphalt roads": "طرق أسفلتية ناعمة",
    "Asphalt roads with some distress": "طرق أسفلتية بها عيوب/تلفيات",
    "Earth roads": "طرق ترابية",
    "Gravel roads": "طرق حصوية",
    "Other": "أخرى"
  },
  Road_surface_conditions: {
    "Dry": "جاف",
    "Flood over 3cm. deep": "غمر مائي (أعمق من 3 سم)",
    "Snow": "ثلوج متراكمة",
    "Wet or damp": "رطب أو مبتل"
  },
  Weather_conditions: {
    "Normal": "مستقر / طبيعي",
    "Cloudy": "غائم",
    "Fog or mist": "ضباب أو شبورة مائية",
    "Raining": "ممطر",
    "Raining and Windy": "أمطار مصحوبة برياح",
    "Snow": "تساقط ثلوج",
    "Windy": "عاصف / رياح شديدة",
    "Other": "أخرى"
  },
  Light_conditions: {
    "Daylight": "ضوء النهار",
    "Darkness - lights lit": "ظلام - أعمدة الإنارة مضاءة",
    "Darkness - lights unlit": "ظلام - أعمدة الإنارة مطفأة",
    "Darkness - no lighting": "ظلام دامس - لا توجد إنارة"
  },
  Road_allignment: {
    "Tangent road with flat terrain": "طريق مستقيم وأرض مسطحة",
    "Tangent road with mild grade and flat terrain": "طريق مستقيم بمنحدر خفيف",
    "Tangent road with rolling terrain": "طريق مستقيم بتضاريس مموجة",
    "Tangent road with mountainous terrain and": "طريق مستقيم بتضاريس جبلية",
    "Gentle horizontal curve": "منحنى أفقي خفيف",
    "Sharp reverse curve": "منحنى عكسي حاد",
    "Steep grade upward with mountainous terrain": "منحدر صاعد حاد (منطقة جبلية)",
    "Steep grade downward with mountainous terrain": "منحدر هابط حاد (منطقة جبلية)",
    "Escarpments": "منحدرات جبلية وعرة"
  },
  Types_of_Junction: {
    "No junction": "طريق مفتوح (لا توجد تقاطعات)",
    "Crossing": "مفترق طرق / عبور",
    "T Shape": "تقاطع على شكل حرف T",
    "Y Shape": "تقاطع على شكل حرف Y",
    "X Shape": "تقاطع رباعي (شكل X)",
    "O Shape": "دوار مروري (صينية)",
    "Other": "أخرى"
  },
  Lanes_or_Medians: {
    "Two-way (divided with solid lines road marking)": "اتجاهين (مفصول بخطوط متصلة)",
    "Two-way (divided with broken lines road marking)": "اتجاهين (مفصول بخطوط متقطعة)",
    "Double carriageway (median)": "طريق مزدوج بحاجز/جزيرة وسطية",
    "Undivided Two way": "اتجاهين غير مفصلين",
    "One way": "اتجاه واحد",
    "other": "أخرى"
  },
  Day_of_week: {
    "Saturday": "السبت",
    "Sunday": "الأحد",
    "Monday": "الإثنين",
    "Tuesday": "الثلاثاء",
    "Wednesday": "الأربعاء",
    "Thursday": "الخميس",
    "Friday": "الجمعة"
  }
};

const SEVERITY_TRANSLATIONS: Record<string, string> = {
  "Slight Injury": "إصابة طفيفة",
  "Serious Injury": "إصابة جسيمة",
  "Fatal injury": "إصابة مميتة (وفاة)",
};

const FACTOR_TRANSLATIONS: Record<string, string> = {
  surface_cond: "حالة سطح الطريق",
  light: "حالة الإضاءة",
  surface_type: "نوع سطح الطريق",
  weather: "الظروف الجوية",
  align: "استقامة وتضاريس الطريق",
  lanes: "تخطيط الحارات والجزر",
  junction: "تصميم التقاطع",
  vehicles: "كثافة المركبات",
  hour: "توقيت الساعة",
};

// القيم الابتدائية بالإنجليزي لتسهيل معالجتها خلف الكواليس
const initialFormState: RiskFormData = {
  Road_surface_type: "Asphalt roads",
  Road_surface_conditions: "Dry",
  Light_conditions: "Daylight",
  Weather_conditions: "Normal",
  Road_allignment: "Tangent road with flat terrain",
  Lanes_or_Medians: "Two-way (divided with solid lines road marking)",
  Types_of_Junction: "No junction",
  Number_of_vehicles_involved: 2,
  Hour: 12,
  Day_of_week: "Monday",
};

export function RoadRiskAnalyzer() {
  const [formData, setFormData] = useState<RiskFormData>(initialFormState);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any | null>(null);

  const handleSelectChange = (field: keyof RiskFormData, value: string | number) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const runAnalysis = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const apiUrl = import.meta.env.VITE_MODEL_RISK_URL || "http://localhost:8000/predict";
      
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!response.ok) throw new Error("فشل الاتصال بخادم النموذج الذكي.");
      
      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "حدث خطأ غير متوقع أثناء التحليل.");
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (score: number) => {
    if (score >= 0.7) return { text: "text-red-500", bg: "bg-red-500/10", border: "border-red-500/20", bar: "bg-red-500" };
    if (score >= 0.4) return { text: "text-amber-500", bg: "bg-amber-500/10", border: "border-amber-500/20", bar: "bg-amber-500" };
    return { text: "text-emerald-500", bg: "bg-emerald-500/10", border: "border-emerald-500/20", bar: "bg-emerald-500" };
  };

  return (
    <div className="card p-6 w-full max-w-5xl mx-auto" dir="rtl">
      <div className="mb-6 border-b border-line pb-4 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-ink flex items-center gap-2">
            <Cpu className="w-5 h-5 text-primary" />
            تحليل مخاطر الطريق الذكي 
          </h2>
          <p className="text-xs text-ink-soft mt-1">
            قم بتهيئة المعطيات والظروف المحيطة بالطريق لمحاكاة احتمالات خطورة الحوادث.
          </p>
        </div>
      </div>

      {/* Grid Inputs */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        
        <FormSelect
          label="نوع سطح الطريق"
          icon={<Map className="w-3.5 h-3.5" />}
          value={formData.Road_surface_type}
          mapping={TRANSLATIONS.Road_surface_type}
          onChange={(val) => handleSelectChange("Road_surface_type", val)}
        />
        <FormSelect
          label="حالة سطح الطريق"
          icon={<Layers className="w-3.5 h-3.5" />}
          value={formData.Road_surface_conditions}
          mapping={TRANSLATIONS.Road_surface_conditions}
          onChange={(val) => handleSelectChange("Road_surface_conditions", val)}
        />
        <FormSelect
          label="الطقس / الأحوال الجوية"
          icon={<CloudRain className="w-3.5 h-3.5" />}
          value={formData.Weather_conditions}
          mapping={TRANSLATIONS.Weather_conditions}
          onChange={(val) => handleSelectChange("Weather_conditions", val)}
        />

        <FormSelect
          label="الإضاءة / حالة الضوء"
          icon={<Sun className="w-3.5 h-3.5" />}
          value={formData.Light_conditions}
          mapping={TRANSLATIONS.Light_conditions}
          onChange={(val) => handleSelectChange("Light_conditions", val)}
        />
        <FormSelect
          label="استقامة وتضاريس الطريق"
          icon={<Navigation className="w-3.5 h-3.5" />}
          value={formData.Road_allignment}
          mapping={TRANSLATIONS.Road_allignment}
          onChange={(val) => handleSelectChange("Road_allignment", val)}
        />
        <FormSelect
          label="نوع التقاطعات"
          icon={<GitMerge className="w-3.5 h-3.5" />}
          value={formData.Types_of_Junction}
          mapping={TRANSLATIONS.Types_of_Junction}
          onChange={(val) => handleSelectChange("Types_of_Junction", val)}
        />

        <FormSelect
          label="تخطيط الحارات / الجزر الوسطية"
          icon={<Map className="w-3.5 h-3.5" />}
          value={formData.Lanes_or_Medians}
          mapping={TRANSLATIONS.Lanes_or_Medians}
          onChange={(val) => handleSelectChange("Lanes_or_Medians", val)}
        />
        <FormSelect
          label="اليوم"
          icon={<Calendar className="w-3.5 h-3.5" />}
          value={formData.Day_of_week}
          mapping={TRANSLATIONS.Day_of_week}
          onChange={(val) => handleSelectChange("Day_of_week", val)}
        />
        
        {/* Numeric Inputs */}
        <div className="flex gap-4">
           <div className="flex-1 text-right">
              <label className="mb-1.5 flex items-center gap-1.5 text-[11px] font-bold tracking-wider text-ink-soft uppercase">
                <Car className="w-3.5 h-3.5" /> عدد المركبات
              </label>
              <input
                type="number"
                min="1"
                max="10"
                value={formData.Number_of_vehicles_involved}
                onChange={(e) => handleSelectChange("Number_of_vehicles_involved", parseInt(e.target.value) || 1)}
                className="w-full rounded-xl border border-line bg-panel py-2 px-3 text-sm text-ink outline-none transition focus:border-primary focus:ring-1 focus:ring-primary/50 text-right"
              />
           </div>
           <div className="flex-1 text-right">
              <label className="mb-1.5 flex items-center gap-1.5 text-[11px] font-bold tracking-wider text-ink-soft uppercase">
                <Clock className="w-3.5 h-3.5" /> الساعة (0-23)
              </label>
              <input
                type="number"
                min="0"
                max="23"
                value={formData.Hour}
                onChange={(e) => handleSelectChange("Hour", parseInt(e.target.value) || 0)}
                className="w-full rounded-xl border border-line bg-panel py-2 px-3 text-sm text-ink outline-none transition focus:border-primary focus:ring-1 focus:ring-primary/50 text-right"
              />
           </div>
        </div>
      </div>

      <button
        onClick={runAnalysis}
        disabled={loading}
        className="btn-primary w-full mt-6 py-3.5 text-sm font-bold tracking-wide rounded-xl shadow-lg shadow-primary/20 transition-all active:scale-[0.98] disabled:opacity-70 disabled:pointer-events-none flex justify-center items-center gap-2"
      >
        {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Cpu className="w-5 h-5" />}
        {loading ? "جاري تشغيل التحليل العميق..." : "تشغيل التحليل العميق للمخاطر"}
      </button>

      {/* Results Display */}
      <AnimatePresence>
        {error && (
          <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, height: 0 }} className="mt-4 p-4 rounded-xl border border-red-200 bg-red-50 text-red-700 text-sm font-semibold flex items-center gap-2">
            <AlertTriangle className="w-4 h-4" /> {error}
          </motion.div>
        )}

        {result && (
          <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} className="mt-6 p-6 rounded-2xl border border-line bg-panel/30 space-y-6">
            <div className="flex items-center justify-between pb-3 border-b border-line">
              <h3 className="text-base font-bold text-ink flex items-center gap-2">
                <Gauge className="w-5 h-5 text-primary" /> لوحة نتائج فحص الطريق الشاملة
              </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* Risk Engine Card */}
              {result.road_risk && (
                <div className="rounded-xl border border-line bg-panel p-4 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-xs font-bold text-ink-soft flex items-center gap-1">
                        <AlertOctagon className="w-4 h-4 text-primary" /> مؤشر خطورة الطريق الهندسي
                      </span>
                      {(() => {
                        const style = getRiskColor(result.road_risk.risk_score);
                        return (
                          <span className={`text-[11px] font-bold px-2.5 py-0.5 rounded-full border ${style.bg} ${style.text} ${style.border}`}>
                            {result.road_risk.risk_level === "High" ? "مرتفع الخطورة" : 
                             result.road_risk.risk_level === "Medium" ? "خطورة متوسطة" : "آمن / منخفض"}
                          </span>
                        );
                      })()}
                    </div>
                    <div className="flex items-baseline gap-1 my-2">
                      <span className={`text-3xl font-black font-mono ${getRiskColor(result.road_risk.risk_score).text}`}>
                        {(result.road_risk.risk_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-line h-2 rounded-full overflow-hidden mt-3">
                      <div 
                        className={`h-full rounded-full transition-all duration-500 ${getRiskColor(result.road_risk.risk_score).bar}`}
                        style={{ width: `${result.road_risk.risk_score * 100}%` }}
                      />
                    </div>
                  </div>

                  {result.road_risk.top_factors && (
                    <div className="mt-5 pt-3 border-t border-line/60">
                      <span className="text-[10px] font-bold text-ink-faint block mb-2">العوامل الأكثر تأثيراً على الخطر:</span>
                      <div className="flex flex-wrap gap-1.5">
                        {result.road_risk.top_factors.map((factor: string) => (
                          <span key={factor} className="text-[10px] bg-panel-hover border border-line text-ink-soft px-2 py-1 rounded-md font-semibold">
                            ⚠️ {FACTOR_TRANSLATIONS[factor] || factor}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* ML Model Card */}
              {result.ml_model && (
                <div className="rounded-xl border border-line bg-panel p-4 flex flex-col justify-between">
                  <div>
                    <span className="text-xs font-bold text-ink-soft flex items-center gap-1 mb-3">
                      <TrendingUp className="w-4 h-4 text-blue-500" /> توقعات الذكاء الاصطناعي (ML)
                    </span>
                    <div className="my-2">
                      <div className="text-xs text-ink-faint">خطورة الحادث المحتملة:</div>
                      <div className="text-xl font-bold text-ink mt-0.5">
                        {SEVERITY_TRANSLATIONS[result.ml_model.predicted_severity] || result.ml_model.predicted_severity}
                      </div>
                    </div>
                  </div>

                  {result.ml_model.probabilities && (
                    <div className="mt-4 pt-3 border-t border-line/60 space-y-2">
                      <span className="text-[10px] font-bold text-ink-faint block">توزيع احتمالات الثقة للنموذج:</span>
                      {Object.entries(result.ml_model.probabilities).map(([severity, prob]: [string, any]) => {
                        const percent = (prob * 100).toFixed(1);
                        return (
                          <div key={severity} className="space-y-1">
                            <div className="flex justify-between text-[10px] font-semibold text-ink-soft">
                              <span>{SEVERITY_TRANSLATIONS[severity] || severity}</span>
                              <span className="font-mono">{percent}%</span>
                            </div>
                            <div className="w-full bg-line h-1.5 rounded-full overflow-hidden">
                              <div className="h-full bg-blue-500/80 rounded-full" style={{ width: `${percent}%` }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// المكون المنسدل الذكي الداخلي لدعم عرض القيم بالعربية ومحاذاتها لليمين
function FormSelect({
  label,
  icon,
  value,
  mapping,
  onChange,
}: {
  label: string;
  icon: React.ReactNode;
  value: string;
  mapping: Record<string, string>;
  onChange: (val: string) => void;
}) {
  return (
    <div className="flex flex-col text-right">
      <label className="mb-1.5 flex items-center gap-1.5 text-[11px] font-bold tracking-wider text-ink-soft uppercase">
        {icon} {label}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full appearance-none rounded-xl border border-line bg-panel py-2 px-3 text-sm text-ink outline-none transition focus:border-primary focus:ring-1 focus:ring-primary/50 cursor-pointer text-right pr-3 pl-8"
      >
        {Object.entries(mapping).map(([engKey, arbValue]) => (
          <option key={engKey} value={engKey}>
            {arbValue}
          </option>
        ))}
      </select>
    </div>
  );
}
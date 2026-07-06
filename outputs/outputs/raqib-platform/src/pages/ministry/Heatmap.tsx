import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  Loader2,
  MapPin,
  AlertTriangle,
  Construction,
  TrafficCone,
  Download,
  RotateCcw,
  Filter,
  Eye,
  EyeOff,
  Calendar,
  TrendingUp,
  Users,
  MapIcon,
  ChevronDown,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { useI18n } from "@/i18n/I18nContext";
import type { HazardType } from "@/types";

// Lazy-load Leaflet and Leaflet.heat to avoid issues if not in package.json yet
// (they're optional dependencies for heatmap visualization)
let L: any;
let HeatmapLayer: any;
let leafletLoadPromise: Promise<{ L: any; HeatmapLayer: any }> | null = null;

async function loadLeaflet(): Promise<{ L: any; HeatmapLayer: any }> {
  if (L && HeatmapLayer) return { L, HeatmapLayer };
  if (leafletLoadPromise) return leafletLoadPromise;
  
  leafletLoadPromise = new Promise((resolve) => {
    // Load Leaflet CSS
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css";
    document.head.appendChild(link);

    // Load Leaflet JS
    const script1 = document.createElement("script");
    script1.src = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js";
    script1.onload = () => {
      L = (window as any).L;

      // Load Leaflet.heat
      const script2 = document.createElement("script");
      script2.src = "https://cdnjs.cloudflare.com/ajax/libs/leaflet.heat/0.2.0/leaflet-heat.js";
      script2.onload = () => {
        HeatmapLayer = (window as any).L.heatLayer;
        resolve({ L, HeatmapLayer });
      };
      document.head.appendChild(script2);
    };
    document.head.appendChild(script1);
  });
  
  return leafletLoadPromise;
}
// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ReportRecord {
  id: string;
  citizenName?: string;
  phone?: string;
  description?: string;
  capturedAt?: string;
  location: {
    mode: string;
    lat?: number;
    lng?: number;
    address?: string;
    governorate?: string;
  };
  analysisMode: "all" | HazardType;
  severity: "none" | "low" | "medium" | "high";
  results: { accidents?: any; traffic?: any; potholes?: any };
  createdAt: string;
}

interface HeatmapData {
  accidents: Array<[number, number, number]>; // [lat, lng, weight]
  potholes: Array<[number, number, number]>;
  traffic: Array<[number, number, number]>;
}

type HazardKey = "accidents" | "potholes" | "traffic";

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

function extractConfidence(results: any, type: "accident" | "pothole" | "traffic"): number {
  let maxConf = 0;

  if (type === "accident" && results?.accidents?.detections) {
    results.accidents.detections.forEach((d: any) => {
      if (d.class_name?.toLowerCase() !== "normal" && d.confidence > maxConf) {
        maxConf = d.confidence;
      }
    });
  } else if (type === "pothole" && results?.potholes?.detections) {
    results.potholes.detections.forEach((d: any) => {
      if (d.class_name?.toLowerCase() === "pothole" && d.confidence > maxConf) {
        maxConf = d.confidence;
      }
    });
  } else if (type === "traffic" && results?.traffic?.detections) {
    results.traffic.detections.forEach((d: any) => {
      if (d.class_name?.toLowerCase() === "damaged" && d.confidence > maxConf) {
        maxConf = d.confidence;
      }
    });
  }

  return maxConf;
}

function buildHeatmapData(
  reports: ReportRecord[],
  filters: {
    startDate?: string;
    endDate?: string;
    governorate?: string;
    minSeverity?: "low" | "medium" | "high";
  }
): HeatmapData {
  const data: HeatmapData = { accidents: [], potholes: [], traffic: [] };

  reports.forEach((r) => {
    // Apply filters
    if (filters.startDate && new Date(r.createdAt) < new Date(filters.startDate)) return;
    if (filters.endDate && new Date(r.createdAt) > new Date(filters.endDate)) return;
    if (filters.governorate && r.location.governorate !== filters.governorate) return;

    const severityRank = { low: 1, medium: 2, high: 3 };
    const minRank = severityRank[filters.minSeverity || "low"];
    if (severityRank[r.severity as keyof typeof severityRank] < minRank) return;

    const lat = r.location.lat;
    const lng = r.location.lng;
    if (!lat || !lng) return;

    // Extract confidence for each hazard type and add to heatmap
    const accidentConf = extractConfidence(r.results, "accident");
    if (accidentConf > 0) data.accidents.push([lat, lng, accidentConf]);

    const potholeConf = extractConfidence(r.results, "pothole");
    if (potholeConf > 0) data.potholes.push([lat, lng, potholeConf]);

    const trafficConf = extractConfidence(r.results, "traffic");
    if (trafficConf > 0) data.traffic.push([lat, lng, trafficConf]);
  });

  return data;
}

const GOVERNORATES = [
  { value: "cairo", labelAr: "القاهرة", labelEn: "Cairo" },
  { value: "giza", labelAr: "الجيزة", labelEn: "Giza" },
  { value: "alexandria", labelAr: "الإسكندرية", labelEn: "Alexandria" },
  { value: "menoufia", labelAr: "المنوفية", labelEn: "Menoufia" },
  { value: "qalyubia", labelAr: "القليوبية", labelEn: "Qalyubia" },
];

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export function Heatmap() {
  const { t, lang } = useI18n();
  const isArabic = lang === "ar";
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const heatmapLayersRef = useRef<{
    accidents?: any;
    potholes?: any;
    traffic?: any;
  }>({});

  const [reports, setReports] = useState<ReportRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mapReady, setMapReady] = useState(false);
  const [showLayers, setShowLayers] = useState({
    accidents: true,
    potholes: true,
    traffic: true,
  });

  const [filters, setFilters] = useState({
    startDate: "",
    endDate: "",
    governorate: "",
    minSeverity: "low" as "low" | "medium" | "high",
  });

  const [selectedReport, setSelectedReport] = useState<ReportRecord | null>(null);
  const [filteredReports, setFilteredReports] = useState<ReportRecord[]>([]);
  const [clusterView, setClusterView] = useState(false);

  // Load reports on mount
  useEffect(() => {
    async function fetchReports() {
      setLoading(true);
      setError(null);
      try {
        // Try to fetch from backend API first
        const apiUrl = import.meta.env.VITE_REPORTS_API_URL as string | undefined;
        if (apiUrl) {
          const res = await fetch(`${apiUrl}`);
          if (res.ok) {
            const data = await res.json();
            setReports(Array.isArray(data) ? data : []);
            setLoading(false);
            return;
          }
        }

        // Fallback: try to load from localStorage
        const stored = localStorage.getItem("raqib_reports_db_v1");
        if (stored) {
          try {
            setReports(JSON.parse(stored));
            setLoading(false);
            return;
          } catch {
            // ignored
          }
        }

        setError(
          isArabic
            ? "لم يتم العثور على أي بلاغات. تأكد من تشغيل خادم البلاغات أو قم بتصدير البلاغات محليًا أولاً."
            : "No reports found. Ensure the reports backend is running or export reports locally first."
        );
        setLoading(false);
      } catch (err) {
        setError(
          isArabic
            ? "فشل في تحميل البلاغات: " + String(err)
            : "Failed to load reports: " + String(err)
        );
        setLoading(false);
      }
    }

    fetchReports();
  }, [isArabic]);

  // Apply filters and rebuild heatmap
  useEffect(() => {
    const severityRank = { low: 1, medium: 2, high: 3 };
    const minRank = severityRank[filters.minSeverity];

    const filtered = reports.filter((r) => {
      if (filters.startDate && new Date(r.createdAt) < new Date(filters.startDate)) return false;
      if (filters.endDate && new Date(r.createdAt) > new Date(filters.endDate)) return false;
      if (filters.governorate && r.location.governorate !== filters.governorate) return false;
      if (severityRank[r.severity as keyof typeof severityRank] < minRank) return false;
      return true;
    });

    setFilteredReports(filtered);

// Rebuild heatmap layers
    if (mapReady && mapInstanceRef.current && HeatmapLayer) {
      const data = buildHeatmapData(filtered, filters);

      // Remove old layers
      Object.values(heatmapLayersRef.current).forEach((layer: any) => {
        if (layer && mapInstanceRef.current) {
          mapInstanceRef.current.removeLayer(layer);
        }
      });
      heatmapLayersRef.current = {};

      // Add new layers
      if (showLayers.accidents && data.accidents.length > 0) {
        const layer = HeatmapLayer(data.accidents, {
          radius: 25,
          blur: 15,
          max: 1,
          gradient: { 0.2: "#FF6B6B", 0.5: "#FF8E8E", 1: "#FF0000" },
        });
        layer.addTo(mapInstanceRef.current);
        heatmapLayersRef.current.accidents = layer;
      }

      if (showLayers.potholes && data.potholes.length > 0) {
        const layer = HeatmapLayer(data.potholes, {
          radius: 25,
          blur: 15,
          max: 1,
          gradient: { 0.2: "#FFC107", 0.5: "#FFD700", 1: "#FF8C00" },
        });
        layer.addTo(mapInstanceRef.current);
        heatmapLayersRef.current.potholes = layer;
      }

      if (showLayers.traffic && data.traffic.length > 0) {
        const layer = HeatmapLayer(data.traffic, {
          radius: 25,
          blur: 15,
          max: 1,
          gradient: { 0.2: "#2196F3", 0.5: "#64B5F6", 1: "#0D47A1" },
        });
        layer.addTo(mapInstanceRef.current);
        heatmapLayersRef.current.traffic = layer;
      }
    }
}, [reports, filters, showLayers, mapReady]);

  // Initialize map on first render
  useEffect(() => {
    if (loading || !mapContainerRef.current || mapInstanceRef.current) return;

    (async () => {
      try {
        const { L } = await loadLeaflet();

        // Default center: Cairo
        const map = L.map(mapContainerRef.current).setView([30.0444, 31.2357], 10);

        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          attribution: "© OpenStreetMap contributors",
          maxZoom: 19,
        }).addTo(map);

        mapInstanceRef.current = map;
        
        setTimeout(() => map.invalidateSize(), 200);
        
        setMapReady(true);
      } catch (err) {
        setError(isArabic ? "فشل تحميل خريطة Leaflet" : "Failed to load Leaflet map");
      }
    })();

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
        setMapReady(false);
      }
    };
  }, [isArabic, loading]);

  function handleExportFiltered() {
    const blob = new Blob([JSON.stringify(filteredReports, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `raqib-heatmap-filtered-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function resetFilters() {
    setFilters({
      startDate: "",
      endDate: "",
      governorate: "",
      minSeverity: "low",
    });
  }

  if (loading) {
    return (
      <div className="mx-auto max-w-7xl px-4 py-8" dir={isArabic ? "rtl" : "ltr"}>
        <PageHeader
          title={isArabic ? "خريطة الأضرار" : "Damage Heatmap"}
          subtitle={isArabic ? "تصور جغرافي لمناطق الأخطار والأضرار في الطرق" : "Geographic hotspots of road hazards and damages"}
          icon={MapPin}
        />
        <div className="card p-8 flex items-center justify-center gap-3">
          <Loader2 className="h-5 w-5 animate-spin text-primary" />
          <span className="font-semibold text-ink">{isArabic ? "جاري تحميل البلاغات..." : "Loading reports..."}</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mx-auto max-w-7xl px-4 py-8" dir={isArabic ? "rtl" : "ltr"}>
        <PageHeader
          title={isArabic ? "خريطة الأضرار" : "Damage Heatmap"}
          subtitle={isArabic ? "تصور جغرافي لمناطق الأخطار والأضرار في الطرق" : "Geographic hotspots of road hazards and damages"}
          icon={MapPin}
        />
        <div className="card p-8 border border-red-200 bg-red-50">
          <div className="flex items-center gap-3 text-red-700 font-semibold mb-3">
            <AlertTriangle className="h-5 w-5" />
            {isArabic ? "خطأ في التحميل" : "Error"}
          </div>
          <p className="text-sm text-red-600">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-8" dir={isArabic ? "rtl" : "ltr"}>
      <PageHeader
        title={isArabic ? "خريطة الأضرار" : "Damage Heatmap"}
        subtitle={isArabic ? "تصور جغرافي لمناطق الأخطار والأضرار في الطرق" : "Geographic hotspots of road hazards and damages"}
        icon={MapPin}
      />

      <div className="grid gap-5 lg:grid-cols-4">
        {/* Sidebar controls */}
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="lg:col-span-1 space-y-4">
          {/* Layer toggles */}
          <div className="card p-4">
            <div className="text-xs font-bold text-ink-soft uppercase tracking-wider mb-3">
              {isArabic ? "طبقات الخريطة" : "Map Layers"}
            </div>
            <div className="space-y-2">
              <label className="flex items-center gap-2 cursor-pointer rounded-lg border border-line bg-panel/50 p-2 hover:bg-panel transition">
                <input
                  type="checkbox"
                  checked={showLayers.accidents}
                  onChange={(e) => setShowLayers({ ...showLayers, accidents: e.target.checked })}
                  className="rounded"
                />
                <span className="flex items-center gap-1.5 text-xs font-semibold text-ink flex-1">
                  <span className="h-2 w-2 rounded-full bg-red-500" />
                  {isArabic ? "الحوادث المرورية" : "Accidents"}
                </span>
              </label>

              <label className="flex items-center gap-2 cursor-pointer rounded-lg border border-line bg-panel/50 p-2 hover:bg-panel transition">
                <input
                  type="checkbox"
                  checked={showLayers.potholes}
                  onChange={(e) => setShowLayers({ ...showLayers, potholes: e.target.checked })}
                  className="rounded"
                />
                <span className="flex items-center gap-1.5 text-xs font-semibold text-ink flex-1">
                  <span className="h-2 w-2 rounded-full bg-amber-500" />
                  {isArabic ? "الحفر والتشققات" : "Potholes"}
                </span>
              </label>

              <label className="flex items-center gap-2 cursor-pointer rounded-lg border border-line bg-panel/50 p-2 hover:bg-panel transition">
                <input
                  type="checkbox"
                  checked={showLayers.traffic}
                  onChange={(e) => setShowLayers({ ...showLayers, traffic: e.target.checked })}
                  className="rounded"
                />
                <span className="flex items-center gap-1.5 text-xs font-semibold text-ink flex-1">
                  <span className="h-2 w-2 rounded-full bg-blue-500" />
                  {isArabic ? "عيوب العلامات" : "Sign Defects"}
                </span>
              </label>
            </div>
          </div>

          {/* Filters */}
          <div className="card p-4 space-y-3">
            <div className="text-xs font-bold text-ink-soft uppercase tracking-wider">
              {isArabic ? "التصفية" : "Filters"}
            </div>

            <div>
              <label className="mb-1 block text-xs font-semibold text-ink-soft">{isArabic ? "من تاريخ" : "From Date"}</label>
              <input
                type="date"
                value={filters.startDate}
                onChange={(e) => setFilters({ ...filters, startDate: e.target.value })}
                className="w-full rounded-lg border border-line bg-panel px-2 py-1.5 text-xs text-ink outline-none transition focus:border-primary"
              />
            </div>

            <div>
              <label className="mb-1 block text-xs font-semibold text-ink-soft">{isArabic ? "إلى تاريخ" : "To Date"}</label>
              <input
                type="date"
                value={filters.endDate}
                onChange={(e) => setFilters({ ...filters, endDate: e.target.value })}
                className="w-full rounded-lg border border-line bg-panel px-2 py-1.5 text-xs text-ink outline-none transition focus:border-primary"
              />
            </div>

            <div>
              <label className="mb-1 block text-xs font-semibold text-ink-soft">{isArabic ? "المحافظة" : "Governorate"}</label>
              <select
                value={filters.governorate}
                onChange={(e) => setFilters({ ...filters, governorate: e.target.value })}
                className="w-full appearance-none rounded-lg border border-line bg-panel px-2 py-1.5 text-xs text-ink outline-none transition focus:border-primary"
              >
                <option value="">{isArabic ? "الكل" : "All"}</option>
                {GOVERNORATES.map((g) => (
                  <option key={g.value} value={g.value}>
                    {isArabic ? g.labelAr : g.labelEn}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="mb-1 block text-xs font-semibold text-ink-soft">{isArabic ? "أدنى درجة خطورة" : "Min Severity"}</label>
              <select
                value={filters.minSeverity}
                onChange={(e) => setFilters({ ...filters, minSeverity: e.target.value as any })}
                className="w-full appearance-none rounded-lg border border-line bg-panel px-2 py-1.5 text-xs text-ink outline-none transition focus:border-primary"
              >
                <option value="low">{isArabic ? "منخفضة" : "Low"}</option>
                <option value="medium">{isArabic ? "متوسطة" : "Medium"}</option>
                <option value="high">{isArabic ? "عالية" : "High"}</option>
              </select>
            </div>

            <div className="flex gap-2">
              <button
                onClick={resetFilters}
                className="flex-1 flex items-center justify-center gap-1.5 rounded-lg border border-line bg-panel px-2 py-1.5 text-xs font-semibold text-ink-soft hover:bg-panel/80 transition"
              >
                <RotateCcw className="h-3 w-3" />
                {isArabic ? "إعادة تعيين" : "Reset"}
              </button>
              <button
                onClick={handleExportFiltered}
                disabled={filteredReports.length === 0}
                className="flex-1 flex items-center justify-center gap-1.5 rounded-lg border border-line bg-panel px-2 py-1.5 text-xs font-semibold text-ink-soft hover:bg-panel/80 transition disabled:opacity-50"
              >
                <Download className="h-3 w-3" />
                {isArabic ? "تصدير" : "Export"}
              </button>
            </div>
          </div>

          {/* Statistics */}
          <div className="card p-4 space-y-2">
            <div className="text-xs font-bold text-ink-soft uppercase tracking-wider mb-2">
              {isArabic ? "الإحصائيات" : "Statistics"}
            </div>

            <div className="flex items-center justify-between rounded-lg bg-panel/50 p-2">
              <span className="text-xs text-ink-soft">{isArabic ? "إجمالي البلاغات" : "Total Reports"}</span>
              <span className="font-bold text-ink">{filteredReports.length}</span>
            </div>

            <div className="flex items-center justify-between rounded-lg bg-red-50 p-2">
              <span className="flex items-center gap-1 text-xs text-red-600">
                <AlertTriangle className="h-3 w-3" />
                {isArabic ? "الحوادث" : "Accidents"}
              </span>
              <span className="font-bold text-red-600">
                {filteredReports.filter((r) => extractConfidence(r.results, "accident") > 0).length}
              </span>
            </div>

            <div className="flex items-center justify-between rounded-lg bg-amber-50 p-2">
              <span className="flex items-center gap-1 text-xs text-amber-600">
                <Construction className="h-3 w-3" />
                {isArabic ? "الحفر" : "Potholes"}
              </span>
              <span className="font-bold text-amber-600">
                {filteredReports.filter((r) => extractConfidence(r.results, "pothole") > 0).length}
              </span>
            </div>

            <div className="flex items-center justify-between rounded-lg bg-blue-50 p-2">
              <span className="flex items-center gap-1 text-xs text-blue-600">
                <TrafficCone className="h-3 w-3" />
                {isArabic ? "العلامات" : "Signs"}
              </span>
              <span className="font-bold text-blue-600">
                {filteredReports.filter((r) => extractConfidence(r.results, "traffic") > 0).length}
              </span>
            </div>
          </div>

          {/* Cluster view toggle */}
          <button
            onClick={() => setClusterView(!clusterView)}
            className={`w-full flex items-center justify-center gap-2 rounded-lg border px-3 py-2.5 font-semibold transition ${
              clusterView ? "border-primary bg-primary/10 text-primary-700" : "border-line bg-panel text-ink-soft hover:bg-panel/80"
            }`}
          >
            {clusterView ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
            {isArabic ? "عرض القائمة" : "View List"}
          </button>
        </motion.div>

        {/* Map and cluster view */}
        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="lg:col-span-3 space-y-4">
          {/* Map */}
          <div className="card overflow-hidden">
            <div ref={mapContainerRef} className="h-96 w-full" />
          </div>

          {/* Cluster/Detail view */}
          {clusterView && (
            <div className="card p-4">
              <div className="mb-3 flex items-center justify-between">
                <div className="text-sm font-bold text-ink">{isArabic ? "البلاغات المطابقة" : "Matching Reports"}</div>
                <span className="text-xs font-semibold text-ink-soft">{filteredReports.length}</span>
              </div>

              {filteredReports.length === 0 ? (
                <div className="text-center py-8 text-ink-faint text-sm">
                  {isArabic ? "لا توجد بلاغات مطابقة للفلاتر المحددة" : "No reports match the applied filters"}
                </div>
              ) : (
                <div className="space-y-2 max-h-80 overflow-y-auto">
                  {filteredReports
                    .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
                    .map((r) => {
                      const hazards = [];
                      if (extractConfidence(r.results, "accident") > 0) hazards.push("🚨 Accident");
                      if (extractConfidence(r.results, "pothole") > 0) hazards.push("🕳️ Pothole");
                      if (extractConfidence(r.results, "traffic") > 0) hazards.push("⚠️ Sign");

                      return (
                        <motion.div
                          key={r.id}
                          whileHover={{ scale: 1.01 }}
                          onClick={() => setSelectedReport(selectedReport?.id === r.id ? null : r)}
                          className={`rounded-lg border p-3 cursor-pointer transition ${
                            selectedReport?.id === r.id
                              ? "border-primary bg-primary/5"
                              : "border-line bg-panel/30 hover:border-primary/50"
                          }`}
                        >
                          <div className="flex items-start justify-between gap-2 mb-1.5">
                            <div>
                              <div className="font-semibold text-ink text-sm">
                                {r.citizenName || (isArabic ? "مجهول" : "Anonymous")}
                              </div>
                              <div className="text-[10px] text-ink-faint font-mono">{r.id}</div>
                            </div>
                            <span className={`text-xs font-bold rounded-full px-2 py-0.5 ${
                              r.severity === "high" ? "bg-red-100 text-red-700" :
                              r.severity === "medium" ? "bg-amber-100 text-amber-700" :
                              "bg-blue-100 text-blue-700"
                            }`}>
                              {r.severity}
                            </span>
                          </div>

                          <div className="text-[11px] text-ink-soft mb-1.5">
                            <span className="flex items-center gap-1">
                              <MapPin className="h-3 w-3" />
                              {r.location.address || r.location.governorate || "--"}
                            </span>
                          </div>

                          <div className="text-[10px] text-ink-faint mb-1">
                            {new Date(r.createdAt).toLocaleString(isArabic ? "ar-EG" : "en-US")}
                          </div>

                          <div className="flex flex-wrap gap-1">
                            {hazards.map((h) => (
                              <span key={h} className="inline-flex items-center gap-0.5 rounded-full bg-ink/5 px-1.5 py-0.5 text-[10px] font-semibold text-ink">
                                {h}
                              </span>
                            ))}
                          </div>

                          {selectedReport?.id === r.id && r.description && (
                            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-2 pt-2 border-t border-line text-[11px] text-ink-soft italic">
                              "{r.description}"
                            </motion.div>
                          )}
                        </motion.div>
                      );
                    })}
                </div>
              )}
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
}

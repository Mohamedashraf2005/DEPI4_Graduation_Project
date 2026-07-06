import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ImageIcon,
  Loader2,
  ScanLine,
  Sparkles,
  UploadCloud,
  X,
  AlertTriangle,
  TrafficCone,
  Construction,
  CheckCircle,
  MapPin,
  Calendar,
  User,
  Phone,
  FileText,
  Navigation,
  Map as MapIcon,
  Search,
  Download,
  Trash2,
  Copy,
  Check,
  History,
  ExternalLink,
  Gauge,
} from "lucide-react";
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

// Options for localized Egyptian Governorates / Cities (kept as a manual fallback)
const locations = [
  { value: "cairo", labelAr: "القاهرة", labelEn: "Cairo" },
  { value: "giza", labelAr: "الجيزة", labelEn: "Giza" },
  { value: "alexandria", labelAr: "الإسكندرية", labelEn: "Alexandria" },
  { value: "menoufia", labelAr: "المنوفية", labelEn: "Menoufia" },
  { value: "qalyubia", labelAr: "القليوبية", labelEn: "Qalyubia" },
];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
// type LocationMode = "manual" | "gps" | "map" | "search";
type LocationMode = "manual" | "gps" | "map" | "search" | "coords";
type Severity = "none" | "low" | "medium" | "high";

interface LocationData {
  mode: LocationMode;
  lat?: number;
  lng?: number;
  address?: string;
  governorate?: string;
}

interface ReportRecord {
  id: string;
  citizenName?: string;
  phone?: string;
  description?: string;
  capturedAt?: string;
  location: LocationData;
  analysisMode: "all" | HazardType;
  severity: Severity;
  results: { accidents?: any; traffic?: any; potholes?: any };
  createdAt: string;
}

// ---------------------------------------------------------------------------
// Local JSON "database"
// A lightweight, dependency-free persistence layer. Every submitted report is
// stored as a JSON array in localStorage (acts as the local database file),
// and can be exported to a real .json file at any time. If VITE_REPORTS_API_URL
// is configured, each report is also pushed to a backend endpoint so it can be
// appended to a server-side db.json (see reports-server-example.js).
// ---------------------------------------------------------------------------
const REPORTS_DB_KEY = "raqib_reports_db_v1";

function loadReportsDB(): ReportRecord[] {
  try {
    const raw = localStorage.getItem(REPORTS_DB_KEY);
    return raw ? (JSON.parse(raw) as ReportRecord[]) : [];
  } catch {
    return [];
  }
}

function saveReportsDB(reports: ReportRecord[]) {
  try {
    localStorage.setItem(REPORTS_DB_KEY, JSON.stringify(reports));
  } catch (e) {
    console.warn("Failed to persist reports DB to localStorage", e);
  }
}

function generateReportId() {
  const rand = Math.random().toString(36).slice(2, 6).toUpperCase();
  return `RQB-${Date.now().toString(36).toUpperCase()}-${rand}`;
}

// ---------------------------------------------------------------------------
// Google Maps loader (dynamic script injection, no npm dependency required)
// ---------------------------------------------------------------------------
const GOOGLE_MAPS_API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY as string | undefined;
let googleMapsLoaderPromise: Promise<any> | null = null;

function loadGoogleMaps(): Promise<any> {
  if ((window as any).google?.maps) return Promise.resolve((window as any).google);
  if (!GOOGLE_MAPS_API_KEY) return Promise.reject(new Error("NO_KEY"));
  if (googleMapsLoaderPromise) return googleMapsLoaderPromise;

  googleMapsLoaderPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = `https://maps.googleapis.com/maps/api/js?key=${GOOGLE_MAPS_API_KEY}&libraries=places`;
    script.async = true;
    script.defer = true;
    script.onload = () => resolve((window as any).google);
    script.onerror = () => reject(new Error("SCRIPT_LOAD_FAILED"));
    document.head.appendChild(script);
  });
  return googleMapsLoaderPromise;
}

// Reverse geocoding: Google Geocoding API when a key is configured, otherwise
// falls back to OpenStreetMap's free Nominatim service so the feature still
// works out of the box with zero configuration.
async function reverseGeocode(lat: number, lng: number, lang: "ar" | "en"): Promise<string> {
  if (GOOGLE_MAPS_API_KEY) {
    try {
      const res = await fetch(
        `https://maps.googleapis.com/maps/api/geocode/json?latlng=${lat},${lng}&key=${GOOGLE_MAPS_API_KEY}&language=${lang}`
      );
      const data = await res.json();
      if (data?.results?.[0]?.formatted_address) return data.results[0].formatted_address;
    } catch {
      // fall through to Nominatim
    }
  }
  try {
    const res = await fetch(
      `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}&accept-language=${lang}`
    );
    const data = await res.json();
    if (data?.display_name) return data.display_name as string;
  } catch {
    // ignore, fall back to raw coordinates below
  }
  return `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
}

// Forward geocoding for the "search an address" flow.
async function forwardGeocode(query: string, lang: "ar" | "en"): Promise<{ lat: number; lng: number; address: string } | null> {
  if (GOOGLE_MAPS_API_KEY) {
    try {
      const res = await fetch(
        `https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(query)}&key=${GOOGLE_MAPS_API_KEY}&language=${lang}`
      );
      const data = await res.json();
      const r = data?.results?.[0];
      if (r) return { lat: r.geometry.location.lat, lng: r.geometry.location.lng, address: r.formatted_address };
    } catch {
      // fall through to Nominatim
    }
  }
  try {
    const res = await fetch(
      `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&accept-language=${lang}&limit=1`
    );
    const data = await res.json();
    const r = data?.[0];
    if (r) return { lat: parseFloat(r.lat), lng: parseFloat(r.lon), address: r.display_name };
  } catch {
    // ignore
  }
  return null;
}

function severityStyle(s: Severity) {
  switch (s) {
    case "high":
      return { bg: "bg-red-50", border: "border-red-200", text: "text-red-700", dot: "bg-red-500" };
    case "medium":
      return { bg: "bg-amber-50", border: "border-amber-200", text: "text-amber-700", dot: "bg-amber-500" };
    case "low":
      return { bg: "bg-blue-50", border: "border-blue-200", text: "text-blue-700", dot: "bg-blue-500" };
    default:
      return { bg: "bg-emerald-50", border: "border-emerald-200", text: "text-emerald-700", dot: "bg-emerald-500" };
  }
}

function computeSeverity(results: { accidents?: any; traffic?: any; potholes?: any } | null): Severity {
  if (!results) return "none";
  let maxConfidence = 0;
  let hasIssue = false;

  const scan = (block: any, ignoredClass?: string) => {
    block?.detections?.forEach((d: any) => {
      if (ignoredClass && d.class_name?.toLowerCase() === ignoredClass) return;
      hasIssue = true;
      if (d.confidence > maxConfidence) maxConfidence = d.confidence;
    });
  };

  scan(results.accidents, "normal");
  scan(results.traffic);
  scan(results.potholes);

  if (!hasIssue) return "none";
  if (maxConfidence >= 0.75) return "high";
  if (maxConfidence >= 0.4) return "medium";
  return "low";
}

export function Analyze() {
  const { t, lang } = useI18n();
  const isArabic = lang === "ar";
  const langCode: "ar" | "en" = isArabic ? "ar" : "en";
  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  // Report / citizen metadata
  const [citizenName, setCitizenName] = useState("");
  const [phone, setPhone] = useState("");
  const [description, setDescription] = useState("");
  const [incidentDateTime, setIncidentDateTime] = useState("");

  // Location state
  const [locationMode, setLocationMode] = useState<LocationMode>("manual");
  const [incidentLocation, setIncidentLocation] = useState(""); // manual governorate fallback
  const [coords, setCoords] = useState<{ lat: number; lng: number } | null>(null);
  const [addressText, setAddressText] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [coordsInput, setCoordsInput] = useState("");
  const [locationLoading, setLocationLoading] = useState(false);
  const [locationError, setLocationError] = useState<string | null>(null);

  // Map picker modal
  const [showMapPicker, setShowMapPicker] = useState(false);
  const [mapLoading, setMapLoading] = useState(false);
  const [mapLoadError, setMapLoadError] = useState<string | null>(null);
  const [pendingCoords, setPendingCoords] = useState<{ lat: number; lng: number } | null>(null);
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const markerInstanceRef = useRef<any>(null);

  const [selectedMode, setSelectedMode] = useState<"all" | HazardType>("all");
  const [results, setResults] = useState<{ accidents?: any; traffic?: any; potholes?: any } | null>(null);
  const [lastReportId, setLastReportId] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Reports "database"
  const [reports, setReports] = useState<ReportRecord[]>([]);

  useEffect(() => {
    setReports(loadReportsDB());
  }, []);

  // ---- Google Maps picker lifecycle ----------------------------------------
  useEffect(() => {
    if (!showMapPicker) return;
    setMapLoading(true);
    setMapLoadError(null);

    loadGoogleMaps()
      .then((google) => {
        if (!mapContainerRef.current) return;
        const center = coords ?? { lat: 30.0444, lng: 31.2357 }; // default: Cairo
        const map = new google.maps.Map(mapContainerRef.current, {
          center,
          zoom: coords ? 15 : 11,
          streetViewControl: false,
          mapTypeControl: false,
          fullscreenControl: false,
        });
        const marker = new google.maps.Marker({ position: center, map, draggable: true });
        mapInstanceRef.current = map;
        markerInstanceRef.current = marker;
        setPendingCoords(center);

        const updateFromLatLng = (latLng: any) => setPendingCoords({ lat: latLng.lat(), lng: latLng.lng() });
        marker.addListener("dragend", () => updateFromLatLng(marker.getPosition()));
        map.addListener("click", (e: any) => {
          marker.setPosition(e.latLng);
          updateFromLatLng(e.latLng);
        });
        setMapLoading(false);
      })
      .catch((err: Error) => {
        setMapLoading(false);
        setMapLoadError(
          err.message === "NO_KEY"
            ? isArabic
              ? "لم يتم إعداد مفتاح خرائط جوجل (VITE_GOOGLE_MAPS_API_KEY) في هذا المشروع."
              : "Google Maps API key isn't configured (set VITE_GOOGLE_MAPS_API_KEY)."
            : isArabic
              ? "تعذر تحميل خرائط جوجل، تحقق من الاتصال بالإنترنت."
              : "Failed to load Google Maps. Check your connection."
        );
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showMapPicker]);

  async function confirmMapPick() {
    if (!pendingCoords) return;
    setLocationLoading(true);
    const address = await reverseGeocode(pendingCoords.lat, pendingCoords.lng, langCode);
    setCoords(pendingCoords);
    setAddressText(address);
    setLocationMode("map");
    setLocationError(null);
    setLocationLoading(false);
    setShowMapPicker(false);
  }

  function useMyCurrentLocation() {
    if (!navigator.geolocation) {
      setLocationError(isArabic ? "المتصفح لا يدعم تحديد الموقع الجغرافي." : "Your browser doesn't support geolocation.");
      return;
    }
    setLocationLoading(true);
    setLocationError(null);
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const { latitude, longitude } = pos.coords;
        const address = await reverseGeocode(latitude, longitude, langCode);
        setCoords({ lat: latitude, lng: longitude });
        setAddressText(address);
        setLocationMode("gps");
        setLocationLoading(false);
      },
      () => {
        setLocationError(
          isArabic ? "تعذر الوصول إلى موقعك، تأكد من تفعيل صلاحية الموقع." : "Couldn't access your location. Check location permissions."
        );
        setLocationLoading(false);
      },
      { enableHighAccuracy: true, timeout: 10000 }
    );
  }

  async function handleSearchLocation() {
    if (!searchQuery.trim()) return;
    setLocationLoading(true);
    setLocationError(null);
    const result = await forwardGeocode(searchQuery.trim(), langCode);
    if (result) {
      setCoords({ lat: result.lat, lng: result.lng });
      setAddressText(result.address);
      setLocationMode("search");
    } else {
      setLocationError(isArabic ? "لم يتم العثور على هذا الموقع." : "Couldn't find that location.");
    }
    setLocationLoading(false);
  }
  async function handleUseTypedCoords() {
    const match = coordsInput.trim().match(/^(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)$/);
    if (!match) {
      setLocationError(isArabic ? "الصيغة يجب أن تكون: خط العرض, خط الطول" : "Format must be: latitude, longitude");
      return;
    }
    const lat = parseFloat(match[1]);
    const lng = parseFloat(match[3]);
    if (lat < -90 || lat > 90 || lng < -180 || lng > 180) {
      setLocationError(isArabic ? "قيم الإحداثيات غير صالحة." : "Coordinate values out of range.");
      return;
    }
    setLocationLoading(true);
    setLocationError(null);
    const address = await reverseGeocode(lat, lng, langCode);
    setCoords({ lat, lng });
    setAddressText(address);
    setLocationMode("coords");
    setLocationLoading(false);
  }

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
    setLastReportId(null);
    setPreview(f.type.startsWith("image") ? URL.createObjectURL(f) : null);
  }

  function clear() {
    setFile(null);
    setPreview(null);
    setResults(null);
    setApiError(null);
    setLastReportId(null);
    setCitizenName("");
    setPhone("");
    setDescription("");
    setIncidentLocation("");
    setIncidentDateTime("");
    setCoords(null);
    setAddressText("");
    setSearchQuery("");
    setLocationMode("manual");
    setLocationError(null);
  }

  function persistReport(report: ReportRecord) {
    setReports((prev) => {
      const next = [...prev, report];
      saveReportsDB(next);
      return next;
    });

    // Optional: sync to a real backend that appends the record to a JSON file
    // database (see the accompanying reports-server-example.js snippet).
    const apiUrl = import.meta.env.VITE_REPORTS_API_URL as string | undefined;
    if (apiUrl) {
      fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(report),
      }).catch((e) => console.warn("Failed to sync report to server database", e));
    }
  }

  function deleteReport(id: string) {
    setReports((prev) => {
      const next = prev.filter((r) => r.id !== id);
      saveReportsDB(next);
      return next;
    });
  }

  function clearAllReports() {
    setReports([]);
    saveReportsDB([]);
  }

  function exportReportsAsJSON() {
    const blob = new Blob([JSON.stringify(reports, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `raqib-reports-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function copyReportId(id: string) {
    navigator.clipboard.writeText(id).then(() => {
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 1500);
    });
  }

  async function run() {
    if (!file) return;
    setRunning(true);
    setResults(null);
    setApiError(null);
    setLastReportId(null);

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

    // Report metadata
    if (citizenName) formData.append("citizen_name", citizenName);
    if (phone) formData.append("phone", phone);
    if (description) formData.append("description", description);
    if (incidentDateTime) formData.append("captured_at", incidentDateTime);

    // Location metadata (Google/GPS-derived when available, manual fallback otherwise)
    formData.append("location_mode", locationMode);
    if (coords) {
      formData.append("latitude", String(coords.lat));
      formData.append("longitude", String(coords.lng));
    }
    if (addressText) formData.append("address", addressText);
    if (incidentLocation) formData.append("governorate", incidentLocation);

    const fetchAccident = selectedMode === "all" || selectedMode === "accident";
    const fetchTraffic = selectedMode === "all" || selectedMode === "sign_defect";
    const fetchPothole = selectedMode === "all" || selectedMode === "pothole";

    const promises = [];

    if (fetchAccident) {
      const url = import.meta.env.VITE_MODEL_ACCIDENT_URL || "https://mohamedachrvf-raqib-accident-api.hf.space/predict";
      promises.push(fetch(url, { method: "POST", body: formData }).then((r) => r.json()));
    } else {
      promises.push(Promise.resolve(null));
    }

    if (fetchTraffic) {
      const url = import.meta.env.VITE_MODEL_SIGN_DEFECT_URL || "https://mohamedachrvf-raqib-traffic-sign-api.hf.space/predict";
      promises.push(fetch(url, { method: "POST", body: formData }).then((r) => r.json()));
    } else {
      promises.push(Promise.resolve(null));
    }

    if (fetchPothole) {
      const url = import.meta.env.VITE_MODEL_POTHOLE_URL || "https://mohamedachrvf-raqib-pot-hole-api.hf.space/predict";
      promises.push(fetch(url, { method: "POST", body: formData }).then((r) => r.json()));
    } else {
      promises.push(Promise.resolve(null));
    }

    try {
      const [resAccident, resTraffic, resPothole] = await Promise.all(promises);
      const finalResults = { accidents: resAccident, traffic: resTraffic, potholes: resPothole };
      setResults(finalResults);

      const report: ReportRecord = {
        id: generateReportId(),
        citizenName: citizenName || undefined,
        phone: phone || undefined,
        description: description || undefined,
        capturedAt: incidentDateTime || undefined,
        location: {
          mode: locationMode,
          lat: coords?.lat,
          lng: coords?.lng,
          address: addressText || undefined,
          governorate: incidentLocation || undefined,
        },
        analysisMode: selectedMode,
        severity: computeSeverity(finalResults),
        results: finalResults,
        createdAt: new Date().toISOString(),
      };
      persistReport(report);
      setLastReportId(report.id);
    } catch (error) {
      console.error("Cloud connection failed", error);
      setApiError(
        isArabic
          ? "فشل الاتصال بذكاء رقيب السحابي. يرجى التحقق من تفعيل خوادم Hugging Face."
          : "Failed to reach Raqib Cloud AI. Make sure Hugging Face Spaces are active."
      );
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8" dir={isArabic ? "rtl" : "ltr"}>
      <PageHeader title={t("nav.analyze")} subtitle={t("رقيب")} icon={ScanLine} />

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
        <div className="card p-5 flex flex-col justify-between gap-4">
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
              <div className="space-y-4">
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

                {/* Citizen & incident details */}
                <div className="rounded-xl border border-line bg-panel/30 p-4 space-y-3">
                  <div className="text-xs font-bold text-ink-soft uppercase tracking-wider">
                    {isArabic ? "بيانات البلاغ" : "Report Details"}
                  </div>

                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <div>
                      <label className="mb-1 block text-xs font-semibold text-ink-soft">{isArabic ? "اسم المواطن" : "Citizen Name"}</label>
                      <div className="relative">
                        <User className="absolute top-3 h-4 w-4 text-ink-faint ltr:left-3 rtl:right-3" />
                        <input
                          type="text"
                          value={citizenName}
                          onChange={(e) => setCitizenName(e.target.value)}
                          placeholder={isArabic ? "اسم مقدم البلاغ" : "Reporter full name"}
                          className="w-full rounded-xl border border-line bg-panel py-2 text-sm text-ink outline-none transition focus:border-primary ltr:pl-9 ltr:pr-3 rtl:pr-9 rtl:pl-3"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="mb-1 block text-xs font-semibold text-ink-soft">{isArabic ? "رقم الهاتف (اختياري)" : "Phone Number (optional)"}</label>
                      <div className="relative">
                        <Phone className="absolute top-3 h-4 w-4 text-ink-faint ltr:left-3 rtl:right-3" />
                        <input
                          type="tel"
                          value={phone}
                          onChange={(e) => setPhone(e.target.value)}
                          placeholder={isArabic ? "01xxxxxxxxx" : "e.g. 01xxxxxxxxx"}
                          className="w-full rounded-xl border border-line bg-panel py-2 text-sm text-ink outline-none transition focus:border-primary ltr:pl-9 ltr:pr-3 rtl:pr-9 rtl:pl-3"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="mb-1 block text-xs font-semibold text-ink-soft">{isArabic ? "وقت التقاط الصورة" : "Time Captured"}</label>
                      <div className="relative">
                        <Calendar className="absolute top-3 h-4 w-4 text-ink-faint ltr:left-3 rtl:right-3" />
                        <input
                          type="datetime-local"
                          value={incidentDateTime}
                          onChange={(e) => setIncidentDateTime(e.target.value)}
                          className="w-full rounded-xl border border-line bg-panel py-2 text-sm text-ink outline-none transition focus:border-primary ltr:pl-9 ltr:pr-3 rtl:pr-9 rtl:pl-3"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="mb-1 block text-xs font-semibold text-ink-soft">{isArabic ? "المحافظة (احتياطي)" : "Governorate (fallback)"}</label>
                      <div className="relative">
                        <MapPin className="absolute top-3 h-4 w-4 text-ink-faint ltr:left-3 rtl:right-3" />
                        <select
                          value={incidentLocation}
                          onChange={(e) => setIncidentLocation(e.target.value)}
                          className="w-full appearance-none rounded-xl border border-line bg-panel py-2 text-sm text-ink outline-none transition focus:border-primary ltr:pl-9 ltr:pr-3 rtl:pr-9 rtl:pl-3"
                        >
                          <option value="">{isArabic ? "-- اختر المحافظة --" : "-- Select --"}</option>
                          {locations.map((loc) => (
                            <option key={loc.value} value={loc.value}>
                              {isArabic ? loc.labelAr : loc.labelEn}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>

                  <div>
                    <label className="mb-1 block text-xs font-semibold text-ink-soft">{isArabic ? "ملاحظات إضافية" : "Additional Notes"}</label>
                    <div className="relative">
                      <FileText className="absolute top-3 h-4 w-4 text-ink-faint ltr:left-3 rtl:right-3" />
                      <textarea
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        rows={2}
                        placeholder={isArabic ? "أي تفاصيل إضافية عن الضرر أو الموقع..." : "Any extra details about the damage or context..."}
                        className="w-full resize-none rounded-xl border border-line bg-panel py-2 text-sm text-ink outline-none transition focus:border-primary ltr:pl-9 ltr:pr-3 rtl:pr-9 rtl:pl-3"
                      />
                    </div>
                  </div>
                </div>

                {/* Google-powered location */}
                <div className="rounded-xl border border-line bg-panel/30 p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="text-xs font-bold text-ink-soft uppercase tracking-wider">
                      {isArabic ? "موقع الرصد" : "Incident Location"}
                    </div>
                    {coords && (
                      <a
                        href={`https://www.google.com/maps?q=${coords.lat},${coords.lng}`}
                        target="_blank"
                        rel="noreferrer"
                        className="flex items-center gap-1 text-[11px] font-semibold text-primary hover:underline"
                      >
                        {isArabic ? "فتح في خرائط جوجل" : "Open in Google Maps"}
                        <ExternalLink className="h-3 w-3" />
                      </a>
                    )}
                  </div>

                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    <button
                      type="button"
                      onClick={useMyCurrentLocation}
                      disabled={locationLoading}
                      className="flex items-center justify-center gap-1.5 rounded-lg border border-line bg-panel px-2 py-2 text-xs font-semibold text-ink transition hover:border-primary/50 hover:bg-primary/5 disabled:opacity-60"
                    >
                      {locationLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Navigation className="h-3.5 w-3.5" />}
                      {isArabic ? "موقعي الحالي" : "Use current location"}
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowMapPicker(true)}
                      className="flex items-center justify-center gap-1.5 rounded-lg border border-line bg-panel px-2 py-2 text-xs font-semibold text-ink transition hover:border-primary/50 hover:bg-primary/5"
                    >
                      <MapIcon className="h-3.5 w-3.5" />
                      {isArabic ? "تحديد على الخريطة" : "Pick on map"}
                    </button>
                    <div className="relative col-span-2 sm:col-span-1">
                      <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleSearchLocation()}
                        placeholder={isArabic ? "ابحث عن عنوان..." : "Search an address..."}
                        className="w-full rounded-lg border border-line bg-panel py-2 text-xs text-ink outline-none transition focus:border-primary ltr:pl-7 ltr:pr-2 rtl:pr-7 rtl:pl-2"
                      />
                      <button type="button" onClick={handleSearchLocation} className="absolute top-2 text-ink-faint hover:text-primary ltr:left-2 rtl:right-2">
                        <Search className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                    <div className="relative col-span-2 sm:col-span-1">
                      <input
                        type="text"
                        value={coordsInput}
                        onChange={(e) => setCoordsInput(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleUseTypedCoords()}
                        placeholder={isArabic ? "خط العرض, خط الطول" : "lat, lng e.g. 30.145, 31.298"}
                        className="w-full rounded-lg border border-line bg-panel py-2 text-xs text-ink outline-none transition focus:border-primary ltr:pl-7 ltr:pr-2 rtl:pr-7 rtl:pl-2 font-mono"
                      />
                      <button type="button" onClick={handleUseTypedCoords} className="absolute top-2 text-ink-faint hover:text-primary ltr:left-2 rtl:right-2">
                        <MapPin className="h-3.5 w-3.5" />
                      </button>
                    </div>
                    
                  {locationError && (
                    <div className="flex items-center gap-1.5 text-[11px] font-semibold text-red-600">
                      <AlertTriangle className="h-3 w-3" /> {locationError}
                    </div>
                  )}

                  {addressText && (
                    <div className="flex items-start gap-2 rounded-lg border border-primary/20 bg-primary/5 p-2.5 text-xs text-ink">
                      <MapPin className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" />
                      <div>
                        <div className="font-semibold">{addressText}</div>
                        {coords && (
                          <div className="mt-0.5 font-mono text-[10px] text-ink-faint">
                            {coords.lat.toFixed(5)}, {coords.lng.toFixed(5)}
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {!GOOGLE_MAPS_API_KEY && (
                    <div className="text-[10px] text-ink-faint">
                      {isArabic
                        ? "ملاحظة: أضف VITE_GOOGLE_MAPS_API_KEY لتفعيل الخريطة التفاعلية داخل التطبيق."
                        : "Tip: set VITE_GOOGLE_MAPS_API_KEY to enable the in-app interactive map."}
                    </div>
                  )}
                </div>
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
              <div className="flex items-center justify-between border-b border-line pb-3">
                <h3 className="text-base font-bold text-ink">
                  {isArabic ? "نتائج الكشف الذكي الموحدة:" : "Unified Intelligent Detections:"}
                </h3>
                {(() => {
                  const s = severityStyle(computeSeverity(results));
                  const sevLabel =
                    computeSeverity(results) === "high"
                      ? isArabic ? "خطورة عالية" : "High severity"
                      : computeSeverity(results) === "medium"
                        ? isArabic ? "خطورة متوسطة" : "Medium severity"
                        : computeSeverity(results) === "low"
                          ? isArabic ? "خطورة منخفضة" : "Low severity"
                          : isArabic ? "لا يوجد ضرر" : "No issues";
                  return (
                    <span className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-bold ${s.bg} ${s.border} ${s.text}`}>
                      <Gauge className="h-3 w-3" /> {sevLabel}
                    </span>
                  );
                })()}
              </div>

              {lastReportId && (
                <div className="flex items-center justify-between rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-semibold text-emerald-700">
                  <span>{isArabic ? `تم الحفظ محليًا كسجل رقم` : "Saved locally as record"} #{lastReportId}</span>
                  <button onClick={() => copyReportId(lastReportId)} className="flex items-center gap-1 hover:underline">
                    {copiedId === lastReportId ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                    {isArabic ? "نسخ" : "Copy"}
                  </button>
                </div>
              )}

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

      {/* Local reports "database" log */}
      <div className="card mt-5 p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-sm font-bold text-ink">
            <History className="h-4 w-4 text-primary" />
            {isArabic ? `سجل البلاغات المحفوظة (${reports.length})` : `Saved Reports Log (${reports.length})`}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={exportReportsAsJSON}
              disabled={reports.length === 0}
              className="flex items-center gap-1.5 rounded-lg border border-line bg-panel px-3 py-1.5 text-xs font-semibold text-ink transition hover:border-primary/50 hover:bg-primary/5 disabled:opacity-50"
            >
              <Download className="h-3.5 w-3.5" />
              {isArabic ? "تصدير JSON" : "Export JSON"}
            </button>
            <button
              onClick={clearAllReports}
              disabled={reports.length === 0}
              className="flex items-center gap-1.5 rounded-lg border border-red-200 bg-red-50 px-3 py-1.5 text-xs font-semibold text-red-600 transition hover:bg-red-100 disabled:opacity-50"
            >
              <Trash2 className="h-3.5 w-3.5" />
              {isArabic ? "مسح الكل" : "Clear all"}
            </button>
          </div>
        </div>

        {reports.length === 0 ? (
          <div className="rounded-xl border border-dashed border-line py-8 text-center text-xs text-ink-faint">
            {isArabic ? "لا توجد بلاغات محفوظة بعد. سيتم حفظ كل تحليل هنا تلقائيًا." : "No saved reports yet. Every analysis you run is stored here automatically."}
          </div>
        ) : (
          <div className="space-y-2">
            {reports
              .slice()
              .reverse()
              .slice(0, 8)
              .map((r) => {
                const s = severityStyle(r.severity);
                return (
                  <div key={r.id} className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-line bg-panel/40 p-3 text-xs">
                    <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                      <div className="flex items-center gap-2 font-semibold text-ink">
                        <span className={`h-1.5 w-1.5 rounded-full ${s.dot}`} />
                        {r.citizenName || (isArabic ? "مواطن مجهول" : "Anonymous reporter")}
                        <span className="font-mono text-[10px] text-ink-faint">#{r.id}</span>
                      </div>
                      <div className="truncate text-ink-soft">
                        {r.location.address || r.location.governorate || (isArabic ? "بدون موقع" : "No location")}
                      </div>
                      <div className="text-[10px] text-ink-faint">{new Date(r.createdAt).toLocaleString(isArabic ? "ar-EG" : "en-US")}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={`rounded-full border px-2 py-0.5 text-[10px] font-bold ${s.bg} ${s.border} ${s.text}`}>{r.severity}</span>
                      <button onClick={() => copyReportId(r.id)} className="grid h-7 w-7 place-items-center rounded-lg text-ink-faint hover:bg-panel hover:text-primary">
                        {copiedId === r.id ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                      </button>
                      <button onClick={() => deleteReport(r.id)} className="grid h-7 w-7 place-items-center rounded-lg text-ink-faint hover:bg-red-50 hover:text-red-600">
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                );
              })}
          </div>
        )}
      </div>

      {/* Map picker modal */}
      <AnimatePresence>
        {showMapPicker && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 grid place-items-center bg-black/50 p-4"
            onClick={() => setShowMapPicker(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="w-full max-w-xl overflow-hidden rounded-2xl bg-white shadow-xl"
            >
              <div className="flex items-center justify-between border-b border-line px-4 py-3">
                <div className="flex items-center gap-2 text-sm font-bold text-ink">
                  <MapIcon className="h-4 w-4 text-primary" />
                  {isArabic ? "اضغط على الخريطة لتحديد الموقع" : "Tap the map to drop a pin"}
                </div>
                <button onClick={() => setShowMapPicker(false)} className="grid h-8 w-8 place-items-center rounded-lg text-ink-faint hover:bg-panel">
                  <X className="h-4 w-4" />
                </button>
              </div>

              <div className="relative h-80 w-full bg-panel">
                {mapLoading && (
                  <div className="absolute inset-0 grid place-items-center">
                    <Loader2 className="h-6 w-6 animate-spin text-primary" />
                  </div>
                )}
                {mapLoadError && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center">
                    <AlertTriangle className="h-6 w-6 text-amber-500" />
                    <p className="text-sm text-ink-soft">{mapLoadError}</p>
                    <a
                      href="https://www.google.com/maps"
                      target="_blank"
                      rel="noreferrer"
                      className="flex items-center gap-1 text-xs font-semibold text-primary hover:underline"
                    >
                      {isArabic ? "افتح خرائط جوجل يدويًا" : "Open Google Maps manually"}
                      <ExternalLink className="h-3 w-3" />
                    </a>
                  </div>
                )}
                <div ref={mapContainerRef} className="h-full w-full" />
              </div>

              <div className="flex items-center justify-between gap-3 border-t border-line px-4 py-3">
                <div className="font-mono text-xs text-ink-faint">
                  {pendingCoords ? `${pendingCoords.lat.toFixed(5)}, ${pendingCoords.lng.toFixed(5)}` : "--"}
                </div>
                <div className="flex gap-2">
                  <button onClick={() => setShowMapPicker(false)} className="rounded-lg border border-line px-3 py-1.5 text-xs font-semibold text-ink-soft hover:bg-panel">
                    {isArabic ? "إلغاء" : "Cancel"}
                  </button>
                  <button
                    onClick={confirmMapPick}
                    disabled={!pendingCoords || locationLoading}
                    className="btn-primary rounded-lg px-3 py-1.5 text-xs disabled:opacity-60"
                  >
                    {isArabic ? "تأكيد الموقع" : "Confirm location"}
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
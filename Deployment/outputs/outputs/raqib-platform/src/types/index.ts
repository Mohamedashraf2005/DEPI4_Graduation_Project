// ─────────────────────────────────────────────────────────────
//  Shared domain types for the Raqib platform
// ─────────────────────────────────────────────────────────────

/** A string available in both platform languages. */
export interface Localized {
  ar: string;
  en: string;
}

export const L = (ar: string, en: string): Localized => ({ ar, en });

export type Lang = "ar" | "en";

/** The three computer-vision hazard categories + a generic fallback. */
export type HazardType = "sign_defect" | "pothole" | "accident" | "other";

export type Severity = "low" | "medium" | "high" | "critical";

/**
 * Lifecycle of a hazard report.
 * new → analyzing → pending_dispatch → dispatched → acknowledged → resolved
 */
export type ReportStatus =
  | "new"
  | "analyzing"
  | "pending_dispatch"
  | "dispatched"
  | "acknowledged"
  | "resolved";

/**
 * How the capture entered the platform.
 *  - manual:       a media file uploaded by an operator (current input method)
 *  - field_camera: an automatic capture from a field radar/camera (production)
 *  - citizen:      a report submitted through the public citizen portal
 */
export type ReportSource = "manual" | "field_camera" | "citizen";

/** A single model detection drawn as a bounding box (normalized 0..1). */
export interface Detection {
  label: Localized;
  confidence: number; // 0..1
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface TimelineEvent {
  at: string; // ISO datetime
  label: Localized;
  kind: "capture" | "analysis" | "report" | "dispatch" | "ack" | "resolve";
}

export interface Authority {
  id: string;
  name: Localized;
  region: Localized;
}

export interface GeoPoint {
  /** Normalized 0..100 position on the stylized map canvas. */
  x: number;
  y: number;
  city: Localized;
  road: Localized;
}

export interface HazardReport {
  id: string;
  type: HazardType;
  severity: Severity;
  status: ReportStatus;
  source: ReportSource;
  title: Localized;
  description: Localized;
  modelKey: ModelKey;
  confidence: number; // 0..1
  createdAt: string; // ISO
  location: GeoPoint;
  authority: Authority;
  detections: Detection[];
  timeline: TimelineEvent[];
  /** Optional media reference (wired to backend storage later). */
  mediaUrl?: string;
  mediaKind?: "video" | "image";
}

// ── Models ────────────────────────────────────────────────────

export type ModelKey = "sign_defect" | "pothole" | "accident" | "risk";

export type ModelStatus = "online" | "degraded" | "not_connected";

export interface ModelInfo {
  key: ModelKey;
  name: Localized;
  family: "cv" | "ml";
  task: Localized;
  status: ModelStatus;
  version: string;
  /** Headline accuracy / mAP / F1 depending on the model. */
  accuracy: number; // 0..100
  metricLabel: Localized;
  latencyMs: number;
  throughput: Localized;
  lastRun: string;
  /** Name of the env var that holds this model's inference endpoint. */
  endpointEnv: string;
  description: Localized;
}

// ── Road risk prediction (ML) ─────────────────────────────────

export interface RiskInput {
  segment: string;
  timeOfDay: "dawn" | "day" | "dusk" | "night";
  lighting: "good" | "moderate" | "poor" | "none";
  weather: "clear" | "rain" | "fog" | "dust";
  traffic: "low" | "medium" | "high";
  dayType: "weekday" | "weekend" | "holiday";
  surface: "dry" | "wet" | "damaged";
  speedLimit: number;
}

export interface RiskFactor {
  name: Localized;
  weight: number; // 0..1 contribution
}

export interface RiskResult {
  score: number; // 0..100
  level: Severity;
  factors: RiskFactor[];
  recommendation: Localized;
}

// ── RAG assistant ─────────────────────────────────────────────

export interface RagSource {
  refId: string;
  title: Localized;
  kind: "report" | "regulation" | "standard" | "history";
  snippet: Localized;
}

export interface RagMessage {
  id: string;
  role: "user" | "assistant";
  content: Localized;
  sources?: RagSource[];
}

// ── Citizen portal ────────────────────────────────────────────

export interface CitizenReport {
  trackingId: string;
  type: HazardType;
  severity: Severity;
  status: ReportStatus;
  description: Localized;
  location: Localized;
  createdAt: string;
}

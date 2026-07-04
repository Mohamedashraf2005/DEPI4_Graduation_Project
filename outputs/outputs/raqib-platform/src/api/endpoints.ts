// ─────────────────────────────────────────────────────────────
//  Central endpoint map — every route the frontend may call.
//  Edit these (or the matching VITE_* vars in .env) to point at
//  your FastAPI. Per-model inference URLs come straight from env
//  so plugging in a trained model is a one-line change.
// ─────────────────────────────────────────────────────────────

import type { ModelKey } from "@/types";

export const ENDPOINTS = {
  // Reports & dispatch
  reports: "/reports",
  report: (id: string) => `/reports/${id}`,
  dispatchReport: (id: string) => `/reports/${id}/dispatch`,

  // Capture & analyze (manual upload → models → draft report)
  analyze: "/analyze",

  // Model registry + per-model inference (CV)
  models: "/models",
  infer: {
    sign_defect: import.meta.env.VITE_MODEL_SIGN_DEFECT_URL ?? "/models/sign-defect/infer",
    pothole: import.meta.env.VITE_MODEL_POTHOLE_URL ?? "/models/pothole/infer",
    accident: import.meta.env.VITE_MODEL_ACCIDENT_URL ?? "/models/accident/infer",
  } as Record<Exclude<ModelKey, "risk">, string>,

  // Road risk prediction (ML)
  risk: import.meta.env.VITE_MODEL_RISK_URL ?? "/models/risk/predict",
  riskSegments: "/risk/segments",

  // RAG assistant
  rag: import.meta.env.VITE_RAG_QUERY_URL ?? "/rag/query",

  // Citizen portal
  citizenAnalyze: "/citizen/analyze",
  citizenReports: "/citizen/reports",
  citizenTrack: (trackingId: string) => `/citizen/reports/${trackingId}`,
} as const;

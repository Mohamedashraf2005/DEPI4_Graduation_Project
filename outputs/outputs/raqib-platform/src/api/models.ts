import { upload } from "./client";
import { ENDPOINTS } from "./endpoints";
import type { Detection, HazardType, Localized, ModelKey, Severity } from "@/types";

export type CvModelKey = Exclude<ModelKey, "risk">;

/** Output shape every CV model should return after analysing a capture. */
export interface AnalyzeResult {
  type: HazardType;
  severity: Severity;
  confidence: number; // 0..1
  detections: Detection[];
  description: Localized;
}

/**
 * POST {VITE_MODEL_*_URL} — send a captured image/video to the CV model.
 * No mock fallback: results only appear once a real model API is connected,
 * so nothing is ever fabricated on screen.
 */
export async function runInference(modelKey: CvModelKey, file: File): Promise<AnalyzeResult> {
  const form = new FormData();
  form.append("file", file);
  return upload<AnalyzeResult>(ENDPOINTS.infer[modelKey], form);
}

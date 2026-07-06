import { request } from "./client";
import { ENDPOINTS } from "./endpoints";
import type { RiskResult } from "@/types";

/**
 * POST {VITE_MODEL_RISK_URL} — road risk prediction.
 * Input features are still being defined, so the UI is a placeholder for now.
 * Send your finalized feature payload here once the model is ready.
 */
export async function predictRisk(features: Record<string, unknown>): Promise<RiskResult> {
  return request<RiskResult>(ENDPOINTS.risk, {
    method: "POST",
    body: JSON.stringify(features),
  });
}

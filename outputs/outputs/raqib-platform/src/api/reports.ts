import { USE_MOCK, request } from "./client";
import { ENDPOINTS } from "./endpoints";
import type { HazardReport } from "@/types";

/**
 * GET /reports — list hazard reports.
 * Until the backend is connected this returns an empty list (no mock data),
 * so the UI shows a clean empty state. Once live, real reports flow straight in.
 */
export async function listReports(): Promise<HazardReport[]> {
  if (USE_MOCK) return [];
  return request<HazardReport[]>(ENDPOINTS.reports);
}

/** POST /reports/:id/dispatch — route a report to its authority. */
export async function dispatchReport(id: string): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>(ENDPOINTS.dispatchReport(id), { method: "POST" });
}

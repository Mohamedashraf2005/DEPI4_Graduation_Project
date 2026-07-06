import { useSyncExternalStore } from "react";

// ─────────────────────────────────────────────────────────────
//  In-session store for the citizen report → review → reply loop.
//  It holds ONLY what the user submits in this session (no seeded
//  data). Swap these functions for real API calls to make reports
//  persistent and shared across users once the backend is live.
// ─────────────────────────────────────────────────────────────

export type CaseStatus = "new" | "under_review" | "resolved" | "rejected";

export interface CitizenCase {
  ref: string;
  description: string;
  location: string;
  mediaName?: string;
  createdAt: string;
  status: CaseStatus;
  reply?: string;
  repliedAt?: string;
}

export const CASE_STATUS: Record<CaseStatus, { key: string; color: string }> = {
  new: { key: "status.new", color: "#2F6FED" },
  under_review: { key: "status.under_review", color: "#E0A008" },
  resolved: { key: "status.resolved", color: "#1A9E54" },
  rejected: { key: "status.rejected", color: "#54666D" },
};

let cases: CitizenCase[] = [];
const listeners = new Set<() => void>();

function emit() {
  cases = cases.slice();
  listeners.forEach((l) => l());
}

function subscribe(l: () => void) {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
}

function snapshot() {
  return cases;
}

export function addCase(c: CitizenCase) {
  cases = [c, ...cases];
  listeners.forEach((l) => l());
}

export function getCase(ref: string) {
  return cases.find((c) => c.ref === ref.trim());
}

export function replyCase(ref: string, status: CaseStatus, reply: string) {
  cases = cases.map((c) =>
    c.ref === ref ? { ...c, status, reply, repliedAt: new Date().toISOString() } : c
  );
  listeners.forEach((l) => l());
}

/** Reactive list of this session's cases. */
export function useCases(): CitizenCase[] {
  return useSyncExternalStore(subscribe, snapshot, snapshot);
}

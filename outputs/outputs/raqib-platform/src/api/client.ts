// ─────────────────────────────────────────────────────────────
//  HTTP client — the single place the UI talks to your FastAPI.
//  While VITE_USE_MOCK=true everything resolves from local mock
//  data, so the whole UI runs without a backend. Flip the flag in
//  .env once your endpoints are live.
// ─────────────────────────────────────────────────────────────

export const API_BASE: string = import.meta.env.VITE_API_BASE_URL ?? "";

export const USE_MOCK: boolean = (import.meta.env.VITE_USE_MOCK ?? "true") !== "false";

/**
 * Whether the platform is wired to a live backend. While false the UI shows
 * clean empty templates (no fabricated data). Flip VITE_USE_MOCK=false once
 * your models' APIs are ready, and every screen goes live automatically.
 */
export const IS_CONNECTED: boolean = !USE_MOCK;

/** Small artificial latency so mock interactions feel real. */
export const delay = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) throw new ApiError(res.status, `${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

/** Multipart upload helper for sending media to a model endpoint. */
export async function upload<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { method: "POST", body: form });
  if (!res.ok) throw new ApiError(res.status, `${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

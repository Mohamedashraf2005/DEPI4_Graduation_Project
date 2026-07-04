import { USE_MOCK, request, upload } from "./client";
import { ENDPOINTS } from "./endpoints";
import { addCase } from "./store";

export interface CitizenSubmission {
  description: string;
  location: string;
  /** Attached media is sent as multipart when a backend is connected. */
  file?: File;
}

function localRef(): string {
  const n = Math.floor(100000 + Math.random() * 899999);
  return `RQ-${n}`;
}

/**
 * POST /citizen/reports — submit a citizen hazard report.
 * Offline: records the report in the in-session store (so the authority can
 * review it and reply, and the citizen can track it) and returns a reference.
 * Connected: posts the real submission (with media) to the backend.
 */
export async function submitCitizen(payload: CitizenSubmission): Promise<{ ref: string }> {
  if (USE_MOCK) {
    const ref = localRef();
    addCase({
      ref,
      description: payload.description,
      location: payload.location,
      mediaName: payload.file?.name,
      createdAt: new Date().toISOString(),
      status: "new",
    });
    return { ref };
  }

  const form = new FormData();
  if (payload.file) form.append("file", payload.file);
  form.append("description", payload.description);
  form.append("location", payload.location);
  return upload<{ ref: string }>(ENDPOINTS.citizenReports, form);
}

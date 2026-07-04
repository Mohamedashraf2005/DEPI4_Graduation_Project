import { request } from "./client";
import { ENDPOINTS } from "./endpoints";
import type { RagMessage } from "@/types";

/**
 * POST {VITE_RAG_QUERY_URL} — optional retrieval-augmented assistant.
 * Reserved endpoint for a future phase; no UI wired yet.
 */
export async function askRag(query: string): Promise<RagMessage> {
  return request<RagMessage>(ENDPOINTS.rag, {
    method: "POST",
    body: JSON.stringify({ query }),
  });
}

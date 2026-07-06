/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_USE_MOCK: string;
  readonly VITE_MODEL_SIGN_DEFECT_URL: string;
  readonly VITE_MODEL_POTHOLE_URL: string;
  readonly VITE_MODEL_ACCIDENT_URL: string;
  readonly VITE_MODEL_RISK_URL: string;
  readonly VITE_RAG_QUERY_URL: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  BookText,
  FileText,
  History,
  Loader2,
  type LucideIcon,
  Scale,
  Send,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { askRag } from "@/api/rag";
import { L } from "@/types";
import { useI18n } from "@/i18n/I18nContext";
import type { RagMessage, RagSource } from "@/types";

const sourceIcon: Record<RagSource["kind"], LucideIcon> = {
  report: FileText,
  regulation: Scale,
  standard: ShieldCheck,
  history: History,
};

export function Assistant() {
  const { t, tl } = useI18n();
  const [messages, setMessages] = useState<RagMessage[]>([
    { id: "welcome", role: "assistant", content: L("", "") },
  ]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [sources, setSources] = useState<RagSource[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, thinking]);

  async function send(text: string) {
    const q = text.trim();
    if (!q || thinking) return;
    setInput("");
    setMessages((m) => [...m, { id: `u-${Date.now()}`, role: "user", content: L(q, q) }]);
    setThinking(true);
    const res = await askRag(q);
    setThinking(false);
    setMessages((m) => [...m, res]);
    setSources(res.sources ?? []);
  }

  const suggestions = [1, 2, 3, 4].map((i) => t(`rag.suggested.${i}`));
  const hasUserMsg = messages.some((m) => m.role === "user");

  return (
    <div>
      <PageHeader icon={<Sparkles className="h-5 w-5" />} title={t("rag.title")} subtitle={t("rag.subtitle")} />

      <div className="grid gap-5 lg:grid-cols-3">
        {/* chat */}
        <div className="card flex h-[600px] flex-col lg:col-span-2">
          <div ref={scrollRef} className="scrollbar-slim flex-1 space-y-4 overflow-y-auto p-5">
            <div className="flex flex-col gap-4">
              {messages.map((m) => (
                <motion.div
                  key={m.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={
                    "max-w-[82%] " + (m.role === "user" ? "self-end" : "self-start")
                  }
                >
                  {m.role === "assistant" && (
                    <div className="mb-1.5 flex items-center gap-1.5 text-[11px] font-semibold text-primary-700">
                      <Sparkles className="h-3 w-3" />
                      {t("brand.name")}
                    </div>
                  )}
                  <div
                    className={
                      "rounded-2xl px-4 py-3 text-[15px] leading-relaxed " +
                      (m.role === "user"
                        ? "gradient-primary text-white rtl:rounded-tr-sm ltr:rounded-tl-sm"
                        : "border border-line bg-white text-ink rtl:rounded-tl-sm ltr:rounded-tr-sm")
                    }
                  >
                    {m.id === "welcome" ? t("rag.welcome") : tl(m.content)}
                  </div>
                  {m.sources && m.sources.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {m.sources.map((s) => (
                        <span key={s.refId} className="chip bg-panel text-ink-soft ring-1 ring-inset ring-line">
                          <FileText className="h-3 w-3" />
                          <span className="mono text-[10px]">{s.refId}</span>
                        </span>
                      ))}
                    </div>
                  )}
                </motion.div>
              ))}

              {thinking && (
                <div className="self-start">
                  <div className="flex items-center gap-2 rounded-2xl border border-line bg-white px-4 py-3 text-sm text-ink-soft">
                    <Loader2 className="h-4 w-4 animate-spin text-primary" />
                    {t("rag.thinking")}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* suggestions */}
          {!hasUserMsg && (
            <div className="flex flex-wrap gap-2 px-5 pb-3">
              {suggestions.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="rounded-full border border-line bg-white px-3 py-1.5 text-xs text-ink-soft transition hover:border-primary/40 hover:text-primary-700"
                >
                  {s}
                </button>
              ))}
            </div>
          )}

          {/* input */}
          <div className="border-t border-line p-3">
            <div className="flex items-end gap-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send(input);
                  }
                }}
                rows={1}
                placeholder={t("rag.placeholder")}
                className="input max-h-32 flex-1 resize-none"
              />
              <button onClick={() => send(input)} disabled={thinking || !input.trim()} className="btn-primary h-[46px] px-4">
                <Send className="h-4 w-4" />
              </button>
            </div>
            <div className="mt-2 px-1 text-[10px] text-ink-faint">{t("rag.note")}</div>
          </div>
        </div>

        {/* side: sources + KB */}
        <div className="space-y-5">
          <div className="card p-5">
            <h2 className="mb-3 text-sm font-bold text-ink">{t("rag.sources")}</h2>
            {sources.length === 0 ? (
              <p className="text-xs text-ink-faint">{t("rag.noSources")}</p>
            ) : (
              <div className="space-y-2.5">
                {sources.map((s) => {
                  const Icon = sourceIcon[s.kind];
                  return (
                    <div key={s.refId} className="rounded-xl border border-line p-3">
                      <div className="flex items-center gap-2">
                        <span className="grid h-7 w-7 place-items-center rounded-lg bg-primary/10 text-primary">
                          <Icon className="h-3.5 w-3.5" />
                        </span>
                        <span className="flex-1 text-xs font-semibold text-ink">{tl(s.title)}</span>
                        <span className="mono text-[10px] text-ink-faint">{s.refId}</span>
                      </div>
                      <p className="mt-2 text-[11px] leading-relaxed text-ink-soft">{tl(s.snippet)}</p>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="card p-5">
            <h2 className="mb-3 text-sm font-bold text-ink">{t("rag.kb")}</h2>
            <div className="space-y-2">
              {[
                { icon: FileText, label: t("rag.kb.reports"), n: "1,284" },
                { icon: Scale, label: t("rag.kb.regs"), n: "96" },
                { icon: BookText, label: t("rag.kb.standards"), n: "212" },
              ].map((k) => (
                <div key={k.label} className="flex items-center gap-3 rounded-xl bg-panel px-3 py-2.5">
                  <span className="grid h-8 w-8 place-items-center rounded-lg bg-white text-primary">
                    <k.icon className="h-4 w-4" />
                  </span>
                  <span className="flex-1 text-sm font-medium text-ink">{k.label}</span>
                  <span className="mono text-xs font-bold text-ink-soft">{k.n}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

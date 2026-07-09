import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageSquare,
  Send,
  Loader2,
  Bot,
  User,
  AlertTriangle,
  FileText,
  Trash2,
  Copy,
  RotateCcw,
  Check
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { useI18n } from "@/i18n/I18nContext";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Message {
  id: string;
  role: "user" | "bot";
  text: string;
  sources?: string[];
  timestamp: Date;
  error?: boolean;
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

const SUGGESTED_PROMPTS_EN = [
  "What are the most reported road damages?",
  "Show me statistics for this month",
  "Which governorate has the most incidents?",
  "How many reports have been submitted?",
];

const SUGGESTED_PROMPTS_AR = [
  "ما هي أكثر الأضرار المبلغ عنها؟",
  "أظهر لي إحصائيات هذا الشهر",
  "أي محافظة بها أكثر الحوادث؟",
  "كم عدد التقارير المرسلة؟",
];

const STORAGE_KEY = "chatbot_messages";

export function Chatbot() {
  const { t, lang } = useI18n();
  const isArabic = lang === "ar";
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Load messages from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        const restored = parsed.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp),
        }));
        setMessages(restored);
      } catch (e) {
        console.error("Failed to restore messages:", e);
      }
    }
    inputRef.current?.focus();
  }, []);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  }, [messages]);

  // Auto-scroll to the bottom when messages change
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Dynamically adjust textarea height based on typing content
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 96)}px`; // maximum 96px (equivalent to max-h-24)
    }
  }, [inputValue]);

  const handleSendMessage = async (e?: React.FormEvent, retryText?: string) => {
    e?.preventDefault();
    const textToSend = retryText || inputValue.trim();
    if (!textToSend || isLoading) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      text: textToSend,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    if (!retryText) setInputValue("");
    setIsLoading(true);
    setError(null);

    try {
      const apiUrl = import.meta.env.VITE_RAG_API_URL || "http://localhost:8000";
      const res = await fetch(`${apiUrl}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: textToSend }),
      });

      if (!res.ok) {
        throw new Error(
          isArabic ? "فشل الاتصال بخادم الاستعلامات" : "Failed to connect to RAG server"
        );
      }

      const data = await res.json();

      const botMsg: Message = {
        id: crypto.randomUUID(),
        role: "bot",
        text: data.answer,
        sources: data.sources,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, botMsg]);
      setCopiedId(null);
    } catch (err) {
      const errorMsg = isArabic
        ? "عذراً، حدث خطأ. انقر على 'إعادة محاولة' لإرسال الرسالة مرة أخرى."
        : "Sorry, an error occurred. Click 'Retry' to send again.";
      setError(errorMsg);
      setMessages((prev) => {
        const lastMsg = prev[prev.length - 1];
        if (lastMsg?.role === "bot") {
          return [...prev.slice(0, -1), { ...lastMsg, error: true }];
        }
        return prev;
      });
    } finally {
      setIsLoading(false);
      setTimeout(() => inputRef.current?.focus(), 10);
    }
  };

  const handleRetry = () => {
    if (messages.length < 2) return;
    const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
    if (lastUserMsg) {
      setMessages((prev) => prev.filter((m) => m.id !== prev[prev.length - 1]?.id));
      handleSendMessage(undefined, lastUserMsg.text);
    }
  };

  const handleCopyMessage = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // If Enter is pressed without holding Shift, trigger form submission
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
    // Shift+Enter natively performs a newline carriage return inside textareas, no manual additions required
  };

  const handleClearChat = () => {
    setMessages([]);
    setError(null);
  };

  const formatMessageTime = (date: Date) => {
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    const dateOnly = new Date(date.getFullYear(), date.getMonth(), date.getDate());
    const todayOnly = new Date(today.getFullYear(), today.getMonth(), today.getDate());
    const yesterdayOnly = new Date(yesterday.getFullYear(), yesterday.getMonth(), yesterday.getDate());

    let dateStr = "";
    if (dateOnly.getTime() === todayOnly.getTime()) {
      dateStr = isArabic ? "اليوم" : "Today";
    } else if (dateOnly.getTime() === yesterdayOnly.getTime()) {
      dateStr = isArabic ? "أمس" : "Yesterday";
    } else {
      dateStr = date.toLocaleDateString(isArabic ? "ar-EG" : "en-US", {
        month: "short",
        day: "numeric",
      });
    }

    const timeStr = date.toLocaleTimeString(isArabic ? "ar-EG" : "en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });

    return `${dateStr} ${timeStr}`;
  };

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 h-[calc(100vh-4rem)] flex flex-col" dir={isArabic ? "rtl" : "ltr"}>
      <PageHeader
        title={isArabic ? "المساعد الذكي" : "AI Assistant"}
        subtitle={
          isArabic 
            ? "استعلم عن بيانات وتقارير أضرار الطرق باستخدام الذكاء الاصطناعي" 
            : "Query road damage reports and data using AI"
        }
        icon={MessageSquare}
      />

      <div className="card flex-1 flex flex-col overflow-hidden border border-line bg-panel shadow-sm mt-4">
        
        {/* Chat Header Actions */}
        <div className="flex items-center justify-between border-b border-line px-4 py-3 bg-panel/50">
          <div className="flex items-center gap-2 text-sm font-semibold text-ink-soft">
            <Bot className="h-4 w-4 text-primary" />
            {isArabic ? "مساعد التقارير الذكي" : "Reports AI Assistant"}
          </div>
          {messages.length > 0 && (
            <button
              onClick={handleClearChat}
              className="flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs font-semibold text-red-600 hover:bg-red-50 transition"
              title={isArabic ? "مسح المحادثة" : "Clear chat"}
            >
              <Trash2 className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">{isArabic ? "مسح" : "Clear"}</span>
            </button>
          )}
        </div>

        {/* Chat Messages Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6 bg-background/30">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center text-center space-y-6 opacity-90">
              <div className="space-y-3">
                <div className="rounded-full bg-primary/10 p-4 mx-auto">
                  <Bot className="h-8 w-8 text-primary" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-ink">
                    {isArabic ? "كيف يمكنني مساعدتك؟" : "How can I help you?"}
                  </h3>
                  <p className="text-xs text-ink-soft max-w-sm mt-1">
                    {isArabic 
                      ? "اسأل عن إحصائيات الطرق والأضرار والتقارير" 
                      : "Ask about road statistics, damage reports, and analysis"}
                  </p>
                </div>
              </div>

              {/* Suggested Prompts */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-md px-4">
                {(isArabic ? SUGGESTED_PROMPTS_AR : SUGGESTED_PROMPTS_EN).map((prompt, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setInputValue(prompt);
                      setTimeout(() => inputRef.current?.focus(), 0);
                    }}
                    className="text-left rounded-lg border border-line bg-panel hover:border-primary hover:bg-primary/5 px-3 py-2 text-xs text-ink transition cursor-pointer"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <AnimatePresence initial={false}>
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex group ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div className={`flex max-w-[85%] lg:max-w-[75%] gap-3 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}>
                    
                    {/* Avatar */}
                    <div className="flex-shrink-0 mt-1">
                      <div className={`flex h-8 w-8 items-center justify-center rounded-full ${
                        msg.role === "user" ? "bg-primary text-white" : "bg-line/50 text-ink"
                      }`}>
                        {msg.role === "user" ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                      </div>
                    </div>

                    {/* Message Bubble */}
                    <div className={`flex flex-col gap-1.5 ${msg.role === "user" ? "items-end" : "items-start"}`}>
                      <div className={`rounded-2xl px-4 py-2.5 text-sm ${
                        msg.role === "user" 
                          ? "bg-primary text-white rounded-tr-sm" 
                          : msg.error
                          ? "bg-red-50 border border-red-200 text-red-900 rounded-tl-sm shadow-sm"
                          : "bg-panel border border-line text-ink rounded-tl-sm shadow-sm"
                      }`}>
                        <div className="whitespace-pre-wrap leading-relaxed">{msg.text}</div>
                      </div>
                      
                      {/* Sources Display */}
                      {msg.role === "bot" && msg.sources && msg.sources.length > 0 && (
                        <div className="flex flex-wrap gap-1.5 mt-1">
                          {msg.sources.map((source, idx) => (
                            <span 
                              key={idx} 
                              className="inline-flex items-center gap-1 rounded-md bg-panel border border-line px-2 py-0.5 text-[10px] text-ink-soft shadow-sm"
                            >
                              <FileText className="h-3 w-3" />
                              {source.replace('.txt', '')}
                            </span>
                          ))}
                        </div>
                      )}
                      
                      {/* Message Actions */}
                      <div className="flex items-center gap-1 mt-0.5">
                        {msg.role === "bot" && (
                          <button
                            onClick={() => handleCopyMessage(msg.text, msg.id)}
                            className="flex items-center gap-1 rounded px-2 py-0.5 text-[10px] text-ink-soft hover:bg-panel transition opacity-0 group-hover:opacity-100"
                            title={isArabic ? "نسخ" : "Copy"}
                          >
                            {copiedId === msg.id ? (
                              <>
                                <Check className="h-3 w-3 text-green-600" />
                                <span className="text-green-600">{isArabic ? "تم النسخ" : "Copied"}</span>
                              </>
                            ) : (
                              <>
                                <Copy className="h-3 w-3" />
                                <span className="hidden sm:inline">{isArabic ? "نسخ" : "Copy"}</span>
                              </>
                            )}
                          </button>
                        )}
                        <span className="text-[10px] text-ink-faint">
                          {formatMessageTime(msg.timestamp)}
                        </span>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          )}

          {/* Loading Indicator */}
          {isLoading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex justify-start">
              <div className="flex gap-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-line/50 text-ink">
                  <Bot className="h-4 w-4" />
                </div>
                <div className="flex items-center rounded-2xl bg-panel border border-line px-4 py-3 shadow-sm rounded-tl-sm">
                  <div className="flex gap-1">
                    <span className="h-1.5 w-1.5 rounded-full bg-ink-soft animate-bounce" style={{ animationDelay: "0ms" }}></span>
                    <span className="h-1.5 w-1.5 rounded-full bg-ink-soft animate-bounce" style={{ animationDelay: "150ms" }}></span>
                    <span className="h-1.5 w-1.5 rounded-full bg-ink-soft animate-bounce" style={{ animationDelay: "300ms" }}></span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Error Message */}
          {error && (
            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="flex justify-center my-2">
              <div className="flex items-center gap-2 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-xs text-red-600 shadow-sm">
                <AlertTriangle className="h-4 w-4 flex-shrink-0" />
                <span>{error}</span>
                {messages.some((m) => m.error) && (
                  <button
                    onClick={handleRetry}
                    disabled={isLoading}
                    className="ml-2 flex items-center gap-1 rounded bg-red-100 px-2 py-0.5 font-semibold hover:bg-red-200 transition disabled:opacity-50"
                  >
                    <RotateCcw className="h-3 w-3" />
                    {isArabic ? "إعادة محاولة" : "Retry"}
                  </button>
                )}
              </div>
            </motion.div>
          )}
          
          <div ref={messagesEndRef} className="h-1" />
        </div>

        {/* Input Area */}
        <div className="border-t border-line bg-panel p-4">
          <form onSubmit={handleSendMessage} className="space-y-2 max-w-4xl mx-auto">
            <div className="relative flex items-center gap-2">
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isLoading}
                placeholder={isArabic ? "اكتب سؤالك هنا... (Shift+Enter للسطر الجديد)" : "Type your question here... (Shift+Enter for new line)"}
                className={`flex-1 rounded-2xl border border-line bg-background py-3 ${isArabic ? 'pr-4 pl-4' : 'pl-4 pr-4'} text-sm text-ink outline-none transition focus:border-primary disabled:opacity-50 resize-none overflow-y-auto max-h-24`}
                rows={1}
                style={{ minHeight: "44px" }}
              />
              <button
                type="submit"
                disabled={!inputValue.trim() || isLoading}
                className="flex h-9 w-9 items-center justify-center rounded-full bg-primary text-white transition hover:bg-primary/90 disabled:bg-line disabled:text-ink-soft flex-shrink-0"
                title={isArabic ? "إرسال" : "Send"}
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className={`h-4 w-4 ${isArabic ? 'rotate-180' : ''}`} />
                )}
              </button>
            </div>
            <div className="text-center">
              <span className="text-[10px] text-ink-faint">
                {isArabic 
                  ? "قد يرتكب الذكاء الاصطناعي أخطاء. يرجى التحقق من المعلومات الهامة." 
                  : "AI can make mistakes. Please verify important information."}
              </span>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
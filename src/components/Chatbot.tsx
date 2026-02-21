import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  MessageCircle,
  Send,
  X,
  Loader2,
  ShieldAlert,
  Brain,
  ArrowRight,
  Sparkles,
  Zap,
} from "lucide-react";
import { chatGemini } from "@/lib/gemini";
import { chatGroq } from "@/lib/groq";

/* ─────────────────────────────── Types ─────────────────────────────── */
interface Msg {
  role: "user" | "bot";
  text: string;
  navSuggestions?: NavSuggestion[];
}

interface NavSuggestion {
  label: string;
  path: string;
  description: string;
}

/* ─────────────────────────── Site Knowledge ────────────────────────── */
const SITE_KNOWLEDGE = `
You are the Cog.ai Assistant — a warm, empathetic human-like companion. 
Your goal is to help users understand dementia risks and navigate this platform.

TONE & STYLE:
- BE BRIEF: Never write more than 2-3 short paragraphs. 
- BE HUMAN: Avoid clinical "bot-speak". Talk like a helpful friend.
- BE SUPPORTIVE: Use warm language (e.g., "I understand," "We're here for you").
- CALL TO ACTION: Always suggest a relevant next step in the app (like the Assessment or Brain Gym).
- FORMATTING: Use bold text sparingly for key terms. Avoid long lists.

PAGES & ROUTES:
- Home: / (Main landing page)
- Assessment: /assessment (The 4-step AI test - Speech & Games)
- Brain Gym: /brain-gym (Training games like Stroop & N-Back)
- Clinician Portal: /clinician (Patient list for doctors)
- Hospital Finder: /hospital-finder (Find specialists)
- Resources: /resources (Articles & guides)
- How It Works: /how-it-works (Process explanation)
- Features: /features (App capabilities)
- Contact: /contact (Support)
- About: /about (Mission)

AI INFO: XGBoost model, ~90% accuracy, uses voice biomarkers + cognitive logs.
`;

/* ─────────────────────── Navigation Intents ───────────────────────── */
const NAV_INTENTS: Array<{ pattern: RegExp; suggestions: NavSuggestion[] }> = [
  { pattern: /start|test|begin|assess|check|take.*test/i, suggestions: [{ label: "Start Assessment", path: "/assessment", description: "Begin screening" }] },
  { pattern: /brain|gym|game|practice|exercise/i, suggestions: [{ label: "Brain Gym", path: "/brain-gym", description: "Practice games" }] },
  { pattern: /doctor|clinician|patient|portal/i, suggestions: [{ label: "Clinician Portal", path: "/clinician", description: "Doctor dashboard" }] },
  { pattern: /hospital|specialist|specialist|finder/i, suggestions: [{ label: "Hospital Finder", path: "/hospital-finder", description: "Locate specialists" }] },
  { pattern: /resource|article|learn/i, suggestions: [{ label: "Resources", path: "/resources", description: "Guides & research" }] },
  { pattern: /how.*work|explain|process/i, suggestions: [{ label: "How It Works", path: "/how-it-works", description: "Model explanation" }] },
  { pattern: /contact|support|reach/i, suggestions: [{ label: "Contact Us", path: "/contact", description: "Get in touch" }] },
  { pattern: /home|landing|main|index/i, suggestions: [{ label: "Go to Home", path: "/", description: "Main landing page" }] },
];

/* ─────────────────────────── Safety ──────────────────────────────── */
const MEDICAL_DANGER = /stop.*medic|discontinue.*drug|change.*dose|should i (take|avoid|stop)/i;
const DISCLAIMER = "*This is general information, not medical advice.*";

const addDisclaimer = (text: string): string => {
  if (text.toLowerCase().includes("not medical advice")) return text.trim();
  return `${text.trim()}\n\n${DISCLAIMER}`;
};

const isMedicalQuestion = (q: string) => /dementia|alzheimer|memory|diagnos|symptom|treat|medicat/i.test(q);

/* ─────────────────────────── Component ────────────────────────────── */
export default function Chatbot() {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [showHint, setShowHint] = useState(false);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [activeProvider, setActiveProvider] = useState<"gemini" | "groq" | "offline">("gemini");

  const [messages, setMessages] = useState<Msg[]>([{
    role: "bot",
    text: "Hi! 👋 I'm your **Cog.ai Assistant V3**.\n\nI can help you:\n• 🧠 Answer dementia & health questions\n• 🗺️ Navigate the website\n• 📋 Explain how our AI works\n\nHow can I help you today?",
    navSuggestions: [
      { label: "Start Assessment", path: "/assessment", description: "Begin screening" },
      { label: "Brain Gym", path: "/brain-gym", description: "Practice exercises" },
    ],
  }]);

  const QUICK_SUGGESTIONS = [
    "Start Assessment",
    "What is Dementia?",
    "Brain Training Games",
    "Find a Hospital",
    "How does the AI work?",
  ];

  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const shown = sessionStorage.getItem("cogai_chatbot_hint_v2");
    if (!shown) {
      setShowHint(true);
      sessionStorage.setItem("cogai_chatbot_hint_v2", "1");
      const t = setTimeout(() => setShowHint(false), 8000);
      return () => clearTimeout(t);
    }
  }, []);

  useEffect(() => {
    if (open) setShowHint(false);
  }, [open]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const detectNavSuggestions = (q: string): NavSuggestion[] => {
    for (const intent of NAV_INTENTS) {
      if (intent.pattern.test(q)) return intent.suggestions;
    }
    return [];
  };

  const callAI = async (q: string): Promise<{ text: string; navSuggestions: NavSuggestion[] }> => {
    const navS = detectNavSuggestions(q);

    // Safety guard
    if (MEDICAL_DANGER.test(q)) {
      return {
        text: "⚠️ For medication or treatment decisions, please consult a **licensed neurologist**. I cannot provide specific medical advice on medications.\n\n*Not medical advice.*",
        navSuggestions: [{ label: "Find Specialist", path: "/hospital-finder", description: "Search nearby doctors" }],
      };
    }

    const geminiKey = import.meta.env.VITE_GEMINI_API_KEY;
    const groqKey = import.meta.env.VITE_GROQ_API_KEY;

    // 1. Try Gemini
    if (geminiKey && geminiKey !== "your_gemini_api_key_here") {
      try {
        const text = await chatGemini(messages, SITE_KNOWLEDGE + "\n\nCRITICAL: Keep your response under 60 words and be very human-like.", q);
        setActiveProvider("gemini");
        return { text: isMedicalQuestion(q) ? addDisclaimer(text) : text, navSuggestions: navS };
      } catch (err) {
        console.warn("[Chatbot] Gemini failed, attempting fallback...", err);
      }
    }

    // 2. Try Groq
    if (groqKey && groqKey !== "your_groq_api_key_here") {
      try {
        const text = await chatGroq(messages, SITE_KNOWLEDGE + "\n\nCRITICAL: Keep your response under 60 words and be very human-like.", q);
        setActiveProvider("groq");
        return { text: isMedicalQuestion(q) ? addDisclaimer(text) : text, navSuggestions: navS };
      } catch (err) {
        console.error("[Chatbot] Groq also failed:", err);
      }
    }

    // 3. Final Fallback
    setActiveProvider("offline");
    return {
      text: addDisclaimer("I'm currently in lightweight mode. I can answer basic dementia questions or direct you to pages on our site. What would you like to do?"),
      navSuggestions: navS.length > 0 ? navS : [{ label: "Help Guide", path: "/how-it-works", description: "System info" }],
    };
  };

  const handleSend = async (prompt?: string) => {
    const q = (prompt ?? input).trim();
    if (!q || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setLoading(true);
    try {
      const resp = await callAI(q);
      setMessages((m) => [...m, { role: "bot", text: resp.text, navSuggestions: resp.navSuggestions }]);
    } catch (err) {
      console.error("[Chatbot] Critical error:", err);
      setMessages((m) => [...m, { role: "bot", text: "Oops, something went wrong. Try refreshing the page! 👋" }]);
    } finally {
      setLoading(false);
    }
  };

  const handleNavClick = (path: string) => {
    console.log("[Chatbot] Navigating to:", path);
    navigate(path);
    setOpen(false);
  };

  const renderText = (text: string) => {
    return text.split("\n").map((line, i) => (
      <span key={i}>
        {line.split(/(\*\*[^*]+\*\*)/g).map((part, j) =>
          part.startsWith("**") && part.endsWith("**") ? <strong key={j}>{part.slice(2, -2)}</strong> : part
        )}
        <br />
      </span>
    ));
  };

  return (
    <div className="fixed z-50 bottom-6 right-6 flex flex-col items-end gap-3">
      {!open && showHint && (
        <div className="absolute bottom-20 right-0 w-72 rounded-2xl border border-primary/20 bg-card shadow-2xl p-4 text-sm ">
          <div className="flex items-center gap-2 font-semibold text-primary mb-1"><Sparkles className="h-4 w-4" /> Cog.ai Assistant</div>
          <p className="text-xs text-muted-foreground">How can I help you today?</p>
          <div className="absolute right-4 -bottom-2 h-4 w-4 rotate-45 bg-card border-r border-b border-primary/20" />
        </div>
      )}

      {open && (
        <div className="w-[360px] h-[580px] bg-card border border-border rounded-2xl shadow-2xl flex flex-col overflow-hidden animate-in fade-in slide-in-from-bottom-4">
          <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-gradient-to-r from-primary/10 to-purple-500/10 backdrop-blur-sm">
            <div className="flex items-center gap-2.5">
              <div className="h-8 w-8 rounded-full bg-gradient-to-br from-primary to-purple-600 flex items-center justify-center shadow-md">
                <Brain className="h-4 w-4 text-white" />
              </div>
              <div>
                <div className="text-sm font-bold">Cog.ai Assistant</div>
                <div className="text-[10px] font-medium flex items-center gap-1">
                  <span className={`h-1.5 w-1.5 rounded-full ${activeProvider === "offline" ? "bg-gray-400" : "bg-green-500 animate-pulse"}`} />
                  {activeProvider === "offline" ? "Offline" : "Online"}
                </div>
              </div>
            </div>
            <button onClick={() => setOpen(false)} className="p-1.5 hover:bg-muted rounded-lg"><X className="h-4 w-4" /></button>
          </div>

          <div className="flex-1 overflow-y-auto p-3 space-y-3 text-[13px] scroll-smooth">
            {messages.map((m, i) => (
              <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className="max-w-[85%] space-y-2">
                  <div className={`px-3.5 py-2.5 rounded-2xl ${m.role === "user" ? "bg-primary text-white" : "bg-muted text-foreground"}`}>
                    {renderText(m.text)}
                  </div>
                  {m.role === "bot" && m.navSuggestions && (
                    <div className="flex flex-col gap-1.5 mt-1">
                      {m.navSuggestions.map((nav) => (
                        <button key={nav.path} type="button" onClick={() => handleNavClick(nav.path)} className="flex items-center justify-between px-3 py-2 rounded-xl bg-primary/10 hover:bg-primary/20 border border-primary/20 transition-all text-left shadow-sm group">
                          <div className="flex-1">
                            <div className="font-bold text-primary text-[11px] flex items-center gap-1">Go to {nav.label} <ArrowRight className="h-3 w-3" /></div>
                            <div className="text-[10px] text-muted-foreground">{nav.description}</div>
                          </div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {loading && <div className="flex items-center gap-2 text-muted-foreground"><Loader2 className="h-4 w-4 animate-spin" /> Thinking...</div>}
            <div ref={endRef} />
          </div>

          <div className="p-2.5 border-t border-border bg-muted/20">
            {showSuggestions && (
              <div className="flex flex-wrap gap-1.5 mb-2">
                {QUICK_SUGGESTIONS.map((s) => (
                  <button key={s} onClick={() => handleSend(s)} disabled={loading} className="text-[10px] px-2 py-0.5 rounded-full border border-border bg-background hover:bg-primary/5">{s}</button>
                ))}
              </div>
            )}
            <div className="flex items-center gap-2">
              <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && handleSend()} placeholder="Ask something..." className="flex-1 px-3 py-2 rounded-xl border border-border bg-background text-sm outline-none focus:ring-1 focus:ring-primary" />
              <button onClick={() => handleSend()} disabled={loading || !input.trim()} className="p-2 rounded-xl bg-primary text-white disabled:opacity-50"><Send className="h-4 w-4" /></button>
            </div>
          </div>
        </div>
      )}

      {!open && (
        <button onClick={() => setOpen(true)} className="h-14 w-14 rounded-full bg-gradient-to-br from-primary to-purple-600 text-white shadow-xl flex items-center justify-center animate-bounce-slow">
          <MessageCircle className="h-6 w-6" />
        </button>
      )}
    </div>
  );
}

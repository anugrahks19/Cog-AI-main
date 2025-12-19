import { useEffect, useRef, useState } from "react";
import { MessageCircle, Send, X, Loader2, ShieldAlert } from "lucide-react";
import { chatDeepSeek, type ChatMessage } from "@/lib/deepseek";

interface Msg { role: "user" | "bot"; text: string; meta?: "error" | "hint" }

const SAFETY_SUFFIX =
  "I'm mainly here to support dementia-related questions—please feel free to ask more about that. This is general information, not medical advice.";

const appendReminder = (text: string): string => {
  const trimmed = text.trim();
  return trimmed.toLowerCase().includes("this is general information") ? trimmed : `${trimmed}\n\n${SAFETY_SUFFIX}`;
};

const FALLBACKS: Array<{ q: RegExp; a: string }> = [
  { q: /(what\s+is\s+)?dementia|alzheimer/i, a: "Dementia describes a decline in memory, thinking, and daily functioning. Alzheimer’s disease is its most common cause." },
  { q: /symptom|sign|memory|forget|confus/i, a: "Typical signs include forgetfulness, confusion about time or place, trouble finding words, mood swings, and difficulty with familiar tasks." },
  { q: /risk|cause/i, a: "Ageing, genetics, heart disease, diabetes, head injury, and limited physical or social activity can raise risk." },
  { q: /prevent|reduce|lifestyle|diet/i, a: "Helpful habits: regular exercise, balanced diet, social connection, brain games, adequate sleep, and managing blood pressure or diabetes." },
  { q: /diagnos|screen|test/i, a: "Doctors combine history, cognitive tests, lab work, and imaging to diagnose. Early screening helps plan support." },
  { q: /treat|cure|medicat|therapy/i, a: "There’s no full cure yet, but medicines, occupational therapy, counselling, and caregiver support improve quality of life." },
  { q: /caregiver|family|support/i, a: "Caregivers benefit from respite breaks, routines, support groups, and sharing tasks among family members." },
  { q: /stress|anxiety|mental/i, a: "Try paced breathing, short walks, and social conversations. If stress feels heavy, speak with a mental health professional." },
  { q: /(grandma|grandpa|mother|father|parent).*(decline|confus|forget)/i, a: "Changes such as memory lapses, confusion, or withdrawing socially can be early signs of cognitive decline. Encourage her to speak with a doctor for a full assessment, note when symptoms started, and provide reassurance that you’ll support her through the process." },
  { q: /(do i|does).*have dementia|diagnose|tell me if/i, a: "Only a trained clinician can confirm dementia. A thorough cognitive exam, discussion of medical history, and possibly brain imaging are needed. It’s best to book an appointment with a neurologist or geriatric specialist soon." },
  { q: /how are you|how r u|who are you/i, a: "I’m doing well and ready to listen. Tell me how I can make your day easier." },
];

const SMALL_TALK: Array<{ q: RegExp; a: string }> = [
  { q: /hello|hi|hey/i, a: "Hello! I’m here to answer questions or just chat. Tell me what’s on your mind." },
  { q: /thank/i, a: "Glad to help! Feel free to ask me anything else—dementia-related or otherwise." },
  { q: /how are you|how's it going/i, a: "I’m doing well and ready to assist. How are you feeling today?" },
  { q: /joke|funny/i, a: "Here’s one: Why did the neuron take a break? It needed to pause and reflect!" },
];

const GENERAL_FALLBACK =
  "I’m happy to chat about wellness or everyday life, and I’m especially equipped to support dementia questions. Would you like to talk about memory changes, caregiving, or early screening next?";

function fallbackAnswer(query: string): string {
  for (const item of SMALL_TALK) if (item.q.test(query)) return appendReminder(item.a);
  for (const item of FALLBACKS) if (item.q.test(query)) return appendReminder(item.a);
  return appendReminder(GENERAL_FALLBACK);
}

export default function Chatbot() {
  const [open, setOpen] = useState(false);
  const [showHint, setShowHint] = useState(false);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showTips, setShowTips] = useState(true);
  const [messages, setMessages] = useState<Msg[]>([{
    role: "bot",
    text: "Hi! I'm your Cog.ai companion. Ask me anything—dementia, wellness, stress, or just how you’re doing today. This is general information, not medical advice.",
  }]);
  const SUGGESTIONS = [
    "What lifestyle supports healthy ageing?",
    "Tips to help a caregiver relax",
    "Share a productivity routine",
    "How to stay calm before sleep",
  ];
  const SUGGESTION_RESPONSES: Record<string, string> = {
    "What lifestyle supports healthy ageing?":
      "Here are a few gentle habits that protect brain and body health:\n- Aim for at least 30 minutes of moderate activity most days (walking, yoga, cycling).\n- Fill half the plate with colourful fruits and vegetables and drink plenty of water.\n- Stay socially and mentally engaged through conversations, puzzles, music, or learning something new.",
    "Tips to help a caregiver relax":
      "Caring for someone is meaningful and tiring—try these mini-reset moments:\n- Schedule tiny pauses for breathing exercises or a short walk every day.\n- Share tasks with family or trusted friends so you’re not carrying everything alone.\n- Keep a note of support lines or groups, so you can talk to someone who understands when stress builds up.",
    "Share a productivity routine":
      "A balanced day can feel calmer when you frame it like this:\n- Morning: light movement, healthy breakfast, review a simple priority list.\n- Midday: focus blocks of 45 minutes with 5-minute stretch breaks.\n- Evening: wind down, reflect on wins, set one gentle goal for tomorrow.",
    "How to stay calm before sleep":
      "A short bedtime ritual can ease the mind:\n- Dim lights and switch off screens 30 minutes before bed.\n- Practice slow breathing—inhale for 4 counts, exhale for 6—for about five minutes.\n- Write down tomorrow’s tasks so worries leave your head and stay on paper.",
  };
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const alreadyShown = sessionStorage.getItem("ms_chatbot_hint");
    if (!alreadyShown) {
      setShowHint(true);
      sessionStorage.setItem("ms_chatbot_hint", "1");
      const t = setTimeout(() => setShowHint(false), 8000);
      return () => clearTimeout(t);
    }
  }, []);

  useEffect(() => {
    if (open) setShowHint(false);
  }, [open]);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, open, loading]);

  const callDeepSeek = async (q: string): Promise<string> => {
    const apiKey = import.meta.env.VITE_DEEPSEEK_API_KEY as string | undefined;
    if (!apiKey) return fallbackAnswer(q);
    const sys: ChatMessage = {
      role: "system",
      content:
        "You are a warm, encouraging health & lifestyle companion for a cognitive screening app. You can answer dementia questions plus general wellness, stress, productivity, and small talk. Use friendly tone, short paragraphs or bullet lists, and end with: 'This is general information, not medical advice.' Offer practical steps, normalise emotions, and suggest consulting a qualified professional for personalised care.",
    };
    const historyMsgs: ChatMessage[] = messages.map<ChatMessage>((m) => ({
      role: (m.role === "bot" ? "assistant" : "user") as "assistant" | "user",
      content: m.text,
    }));
    const msgs: ChatMessage[] = [sys, ...historyMsgs, { role: "user", content: q }];
    try {
      const text = await chatDeepSeek(msgs, { max_tokens: 450, temperature: 0.5 });
      return appendReminder(text);
    } catch (error) {
      console.error(error);
      return fallbackAnswer(q);
    }
  };

  const handleSend = async (prompt?: string) => {
    const q = (prompt ?? input).trim();
    if (!q || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setLoading(true);
    try {
      const predefined = SUGGESTION_RESPONSES[q];
      const response = predefined ? appendReminder(predefined) : await callDeepSeek(q);
      setMessages((m) => [...m, { role: "bot", text: response }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed z-50 bottom-6 right-6">
      {/* Bubble button */}
      {!open && (
        <>
          {showHint && (
            <div className="absolute bottom-20 right-4 w-64 rounded-xl border border-border bg-card shadow-xl p-3 text-sm animate-in fade-in slide-in-from-bottom-4">
              <div className="font-semibold">Hey there! 👋</div>
              <p className="text-xs text-muted-foreground mt-1">
                I’m your dementia info assistant. Tap the bubble if you need quick answers or guidance.
              </p>
              <div className="absolute right-12 -bottom-2 h-4 w-4 rotate-45 bg-card border-r border-b border-border" />
            </div>
          )}
          <button
            aria-label="Chatbot"
            onClick={() => setOpen(true)}
            className="relative rounded-full h-14 w-14 flex items-center justify-center bg-primary text-white shadow-lg hover:brightness-110 focus:outline-none"
          >
            <MessageCircle className="h-6 w-6" />
          </button>
        </>
      )}

      {/* Panel */}
      {open && (
        <div className="w-80 h-[520px] bg-card border border-border rounded-xl shadow-xl flex flex-col overflow-hidden">
          <div className="flex items-center justify-between px-3 py-2 border-b border-border bg-muted/50">
            <div className="text-sm font-medium">Dementia Assistant</div>
            <button className="p-1 hover:bg-muted rounded" onClick={() => setOpen(false)} aria-label="Close">
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="flex-1 p-3 overflow-y-auto space-y-2 text-sm">
            {messages.map((m, i) => (
              <div key={i} className={m.role === "user" ? "text-right" : "text-left"}>
                <span className={`inline-block px-3 py-2 rounded-lg ${m.role === "user" ? "bg-primary text-white" : "bg-muted"}`}>
                  {m.text}
                </span>
              </div>
            ))}
            {loading && (
              <div className="text-left">
                <span className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-muted">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Thinking...
                </span>
              </div>
            )}
            <div ref={endRef} />
          </div>
          <div className="border-t border-border bg-muted/30 px-3 py-2 space-y-3 text-[11px] text-muted-foreground">
            <div className="flex items-start gap-2">
              <ShieldAlert className="h-4 w-4 mt-[2px]" />
              <p>
                Feeling distressed or unsafe? Reach out to a healthcare professional or local helpline. I can only share supportive information.
              </p>
            </div>
            <div>
              <button
                type="button"
                onClick={() => setShowTips((value) => !value)}
                className="text-[11px] font-semibold text-primary hover:underline"
              >
                {showTips ? "Hide quick suggestions" : "Show quick suggestions"}
              </button>
              {showTips && (
                <div className="mt-2 flex flex-wrap gap-2 max-h-20 overflow-y-auto pr-1">
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      key={suggestion}
                      type="button"
                      onClick={() => handleSend(suggestion)}
                      disabled={loading}
                      className="text-xs px-3 py-1 rounded-full border border-border bg-muted/60 hover:bg-muted transition disabled:opacity-60"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
          <div className="p-2 border-t border-border flex items-center gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" ? handleSend() : undefined}
              placeholder="Type your question..."
              className="flex-1 px-3 py-2 rounded-lg border border-border bg-background text-foreground text-sm focus:outline-none"
            />
            <button onClick={() => handleSend()} className="p-2 rounded-lg bg-primary text-white hover:brightness-110 disabled:opacity-50" aria-label="Send" disabled={loading}>
              <Send className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

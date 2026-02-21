/// <reference types="vite/client" />
// src/lib/groq.ts
// Groq API wrapper — OpenAI-compatible endpoint
// Free tier: https://console.groq.com  (6,000 tokens/min, no credit card needed)
// Best free model: llama-3.3-70b-versatile

export interface GroqMessage {
    role: "system" | "user" | "assistant";
    content: string;
}

/**
 * Call Groq with a conversation history + system prompt.
 * Groq uses the standard OpenAI chat completions format.
 */
export async function chatGroq(
    history: Array<{ role: "user" | "bot"; text: string }>,
    systemPrompt: string,
    userQuery: string,
    opts?: { max_tokens?: number; temperature?: number; model?: string }
): Promise<string> {
    const apiKey = import.meta.env.VITE_GROQ_API_KEY as string | undefined;
    if (!apiKey || apiKey === "your_groq_api_key_here") {
        throw new Error("Missing VITE_GROQ_API_KEY");
    }

    // Build OpenAI-format messages with system prompt first
    const messages: GroqMessage[] = [
        { role: "system", content: systemPrompt },
    ];

    // Add conversation history — Groq accepts assistant/user alternating freely
    for (const msg of history) {
        // Skip the initial bot greeting to keep context clean
        if (msg.role === "bot" && messages.length === 1) continue;
        messages.push({
            role: msg.role === "user" ? "user" : "assistant",
            content: msg.text,
        });
    }

    // Add the current user query
    messages.push({ role: "user", content: userQuery });

    const resp = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
            model: opts?.model ?? "llama-3.3-70b-versatile",
            messages,
            max_tokens: opts?.max_tokens ?? 500,
            temperature: opts?.temperature ?? 0.55,
        }),
    });

    if (!resp.ok) {
        const errText = await resp.text();
        throw new Error(`Groq API error ${resp.status}: ${errText}`);
    }

    const json = await resp.json();
    const content = json?.choices?.[0]?.message?.content;
    if (!content) throw new Error("No content from Groq");
    return content as string;
}

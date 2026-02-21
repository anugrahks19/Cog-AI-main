// src/lib/gemini.ts
// Google Gemini API wrapper — gemini-1.5-flash

export type GeminiMessage = { role: "user" | "model"; parts: [{ text: string }] };

/**
 * Call Gemini with a conversation history + system prompt.
 *
 * Rules enforced here so the API never rejects:
 *  1. History must strictly alternate user → model → user → …
 *  2. History must START with a "user" turn (Gemini rejects leading "model" turns)
 *  3. The LAST turn in `contents` must be the new user query (added inside this fn)
 *  4. We exclude the initial bot greeting (index 0) from history to satisfy rule 2
 */
export async function chatGemini(
    history: Array<{ role: "user" | "bot"; text: string }>,
    systemPrompt: string,
    userQuery: string,
    opts?: { max_tokens?: number; temperature?: number }
): Promise<string> {
    const apiKey = import.meta.env.VITE_GEMINI_API_KEY as string | undefined;
    if (!apiKey || apiKey === "your_gemini_api_key_here") {
        throw new Error("Missing VITE_GEMINI_API_KEY");
    }

    // ── Build valid alternating history ──────────────────────────────────
    // Skip the initial welcome bot message (index 0) and only include
    // messages from the first user turn onwards, strictly alternating.
    const contents: GeminiMessage[] = [];

    let foundFirstUser = false;
    let lastRole: "user" | "model" | null = null;

    for (const msg of history) {
        const geminiRole = msg.role === "user" ? "user" : "model";

        // Skip leading bot messages until we hit the first user turn
        if (!foundFirstUser) {
            if (geminiRole === "user") foundFirstUser = true;
            else continue;
        }

        // Skip consecutive same-role messages (keep only the latest of each run)
        if (geminiRole === lastRole) {
            // Replace the last entry so we always have the most recent
            contents[contents.length - 1] = { role: geminiRole, parts: [{ text: msg.text }] };
            continue;
        }

        contents.push({ role: geminiRole, parts: [{ text: msg.text }] });
        lastRole = geminiRole;
    }

    // Ensure the last item before our new query is a "model" turn.
    // If it's a "user" turn, pop it — the new userQuery will replace it.
    if (contents.length > 0 && contents[contents.length - 1].role === "user") {
        contents.pop();
    }

    // Append the current user query as the final turn
    contents.push({ role: "user", parts: [{ text: userQuery }] });

    // ── Fire the request ─────────────────────────────────────────────────
    const resp = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`,
        {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                systemInstruction: { parts: [{ text: systemPrompt }] },
                contents,
                generationConfig: {
                    maxOutputTokens: opts?.max_tokens ?? 450,
                    temperature: opts?.temperature ?? 0.55,
                },
                safetySettings: [
                    { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_ONLY_HIGH" },
                    { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_ONLY_HIGH" },
                    { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_ONLY_HIGH" },
                    { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_ONLY_HIGH" },
                ],
            }),
        }
    );

    if (!resp.ok) {
        const errText = await resp.text();
        throw new Error(`Gemini API error ${resp.status}: ${errText}`);
    }

    const json = await resp.json();
    const content = json?.candidates?.[0]?.content?.parts?.[0]?.text;
    if (!content) {
        // Surface any blocking reason if available
        const reason = json?.candidates?.[0]?.finishReason ?? "unknown";
        throw new Error(`No content from Gemini (finishReason: ${reason})`);
    }
    return content as string;
}
export let isAudioGuideEnabled = false;

export const setAudioGuideEnabled = (enabled: boolean) => {
    isAudioGuideEnabled = enabled;
    if (!enabled) {
        stopAudioGuide();
    }
};

const VOICE_LOCALE_MAP: Record<string, string> = {
    en: "en-US",
    hi: "hi-IN",
    bn: "bn-IN",
    ta: "ta-IN",
    te: "te-IN",
    kn: "kn-IN",
    ml: "ml-IN",
    mr: "mr-IN",
    gu: "gu-IN",
    pa: "pa-IN",
};

export const playAudioGuide = (text: string, language: string = "en") => {
    if (!isAudioGuideEnabled || !("speechSynthesis" in window)) return;

    // Stop any ongoing speech
    stopAudioGuide();

    // Give a tiny delay for natural flow
    setTimeout(() => {
        const utterance = new SpeechSynthesisUtterance(text);
        const locale = VOICE_LOCALE_MAP[language] || "en-US";
        utterance.lang = locale;

        // Try to find a specific voice for the locale
        const voices = window.speechSynthesis.getVoices();
        const preferredVoice = voices.find((v) => v.lang.startsWith(locale) || v.lang.startsWith(language));
        if (preferredVoice) {
            utterance.voice = preferredVoice;
        }

        // Slightly slower and friendlier pitch for accessibility
        utterance.rate = 0.9;
        utterance.pitch = 1.0;

        window.speechSynthesis.speak(utterance);
    }, 300);
};

export const stopAudioGuide = () => {
    if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
    }
};

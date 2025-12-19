import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Quote, Trophy, RotateCcw } from "lucide-react";

export default function WordRecall() {
    const [phase, setPhase] = useState<"idle" | "memorize" | "recall" | "result">("idle");
    const [words, setWords] = useState<string[]>([]);
    const [currentWordIndex, setCurrentWordIndex] = useState(0);
    const [targetWord, setTargetWord] = useState("");
    const [score, setScore] = useState(0);

    const WORD_POOL = ["Apple", "Chair", "River", "Cloud", "Piano", "Tiger", "Bread", "Shoes", "Watch", "Light", "House", "Grass"];

    const startGame = () => {
        const shuffled = [...WORD_POOL].sort(() => 0.5 - Math.random()).slice(0, 5);
        setWords(shuffled);
        setPhase("memorize");
        setCurrentWordIndex(0);
        setScore(0);
    };

    useEffect(() => {
        if (phase === "memorize") {
            if (currentWordIndex < words.length) {
                const timer = setTimeout(() => {
                    setCurrentWordIndex(prev => prev + 1);
                }, 2000);
                return () => clearTimeout(timer);
            } else {
                setPhase("recall");
                generateQuestion();
            }
        }
    }, [phase, currentWordIndex, words]);

    const generateQuestion = () => {
        // Pick a word that was in the list (Correct) or not (Incorrect)
        const isTargetPresent = Math.random() > 0.5;
        let qWord = "";

        if (isTargetPresent) {
            qWord = words[Math.floor(Math.random() * words.length)];
        } else {
            const unused = WORD_POOL.filter(w => !words.includes(w));
            qWord = unused[Math.floor(Math.random() * unused.length)];
        }
        setTargetWord(qWord);
    };

    const handleAnswer = (seen: boolean) => {
        const actuallySeen = words.includes(targetWord);
        if (seen === actuallySeen) {
            setScore(s => s + 1);
            generateQuestion(); // Endless logic for now
        } else {
            setPhase("result");
        }
    };

    if (phase === "result") {
        return (
            <Card className="p-6 text-center space-y-6 flex flex-col items-center justify-center min-h-[300px]">
                <Trophy className="h-16 w-16 text-orange-500 animate-bounce" />
                <div className="space-y-2">
                    <h3 className="text-3xl font-bold">Memory Check Complete</h3>
                    <p className="text-xl text-muted-foreground">Words Recalled: <span className="font-bold text-primary">{score}</span></p>
                </div>
                <Button onClick={startGame} size="lg" className="w-48 gap-2">
                    <RotateCcw className="h-4 w-4" /> Try Again
                </Button>
            </Card>
        );
    }

    return (
        <Card className="p-8 text-center min-h-[300px] flex flex-col items-center justify-center">
            {phase === "idle" && (
                <div className="space-y-6">
                    <Quote className="h-12 w-12 mx-auto text-blue-500" />
                    <h3 className="text-2xl font-bold">Word Recall</h3>

                    <div className="bg-muted/30 p-4 rounded-xl text-left text-sm space-y-2 max-w-xs mx-auto">
                        <p className="font-semibold text-foreground">How to Play:</p>
                        <ul className="list-disc list-inside text-muted-foreground space-y-1">
                            <li>Memorize the stream of words shown.</li>
                            <li>You will then be asked if a word was in the list.</li>
                            <li>Answer <b>YES</b> or <b>NO</b>.</li>
                        </ul>
                    </div>

                    <Button onClick={startGame} size="lg">Start Memory Test</Button>
                </div>
            )}

            {phase === "memorize" && (
                <div className="animate-in fade-in zoom-in duration-500">
                    <h2 className="text-4xl font-bold text-primary">
                        {currentWordIndex < words.length ? words[currentWordIndex] : "Ready..."}
                    </h2>
                    <p className="text-sm text-muted-foreground mt-4">Memorize this</p>
                </div>
            )}

            {phase === "recall" && (
                <div className="space-y-8 animate-in slide-in-from-right duration-300">
                    <h2 className="text-3xl font-semibold">Did you see this word?</h2>
                    <div className="text-5xl font-bold text-foreground py-4 border-2 border-dashed rounded-xl bg-muted/30">
                        {targetWord}
                    </div>
                    <div className="flex gap-4 justify-center">
                        <Button variant="destructive" size="lg" className="w-32" onClick={() => handleAnswer(false)}>NO</Button>
                        <Button className="w-32 bg-green-600 hover:bg-green-700" size="lg" onClick={() => handleAnswer(true)}>YES</Button>
                    </div>
                    <p className="text-sm font-bold">Streak: {score}</p>
                </div>
            )}
        </Card>
    );
}

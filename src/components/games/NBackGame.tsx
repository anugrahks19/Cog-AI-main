import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Play, RotateCcw, Brain, Trophy } from "lucide-react";
import { Progress } from "@/components/ui/progress";

export default function NBackGame() {
    const [isPlaying, setIsPlaying] = useState(false);
    const [isFinished, setIsFinished] = useState(false);
    const [sequence, setSequence] = useState<string[]>([]);
    const [currentIndex, setCurrentIndex] = useState(-1);
    const [score, setScore] = useState(0);
    const [feedback, setFeedback] = useState<"correct" | "wrong" | null>(null);

    // N=2 Back
    const N = 2;
    const LETTERS = ["A", "B", "C", "H", "K", "L", "O", "T"];
    const ROUND_LENGTH = 20;

    useEffect(() => {
        let interval: number;
        if (isPlaying && currentIndex < ROUND_LENGTH) {
            interval = window.setInterval(() => {
                setFeedback(null); // Reset feedback for next letter
                if (currentIndex < ROUND_LENGTH - 1) {
                    nextTurn();
                } else {
                    endGame();
                }
            }, 2500); // 2.5 seconds per letter
        }
        return () => clearInterval(interval);
    }, [isPlaying, currentIndex]);

    const startGame = () => {
        setIsPlaying(true);
        setIsFinished(false);
        setSequence([]);
        setCurrentIndex(-1);
        setScore(0);
        // Pre-generate
        const newSeq: string[] = [];
        for (let i = 0; i < ROUND_LENGTH; i++) {
            // 30% chance of a match if i >= N
            if (i >= N && Math.random() < 0.3) {
                newSeq.push(newSeq[i - N]);
            } else {
                newSeq.push(LETTERS[Math.floor(Math.random() * LETTERS.length)]);
            }
        }
        setSequence(newSeq);
        setCurrentIndex(0);
    };

    const nextTurn = () => {
        setCurrentIndex((prev) => prev + 1);
    };

    const endGame = () => {
        setIsPlaying(false);
        setIsFinished(true);
    };

    const handleMatch = () => {
        if (currentIndex < N) return; // Can't match yet

        const isMatch = sequence[currentIndex] === sequence[currentIndex - N];
        if (isMatch) {
            setScore((prev) => prev + 20);
            setFeedback("correct");
        } else {
            setScore((prev) => Math.max(0, prev - 10));
            setFeedback("wrong");
        }
    };

    const currentLetter = sequence[currentIndex];

    if (isFinished) {
        return (
            <Card className="p-6 text-center space-y-6 flex flex-col items-center justify-center min-h-[350px]">
                <Trophy className="h-16 w-16 text-blue-500 animate-bounce" />
                <div className="space-y-2">
                    <h3 className="text-3xl font-bold">Memory Block Complete!</h3>
                    <p className="text-xl text-muted-foreground">Working Memory Score: <span className="font-bold text-primary">{score}</span></p>
                </div>
                <Button onClick={startGame} size="lg" className="w-48 gap-2">
                    <RotateCcw className="h-4 w-4" /> Train Again
                </Button>
            </Card>
        );
    }

    return (
        <Card className="p-6 text-center space-y-6 bg-white/80 dark:bg-card/80 backdrop-blur-md border-primary/10 min-h-[400px] flex flex-col justify-center">
            <div className="flex justify-between items-center text-sm font-medium text-muted-foreground">
                <span className="flex items-center gap-2"><Brain className="h-4 w-4" /> N-Back Level: {N}</span>
                <span>Score: {score}</span>
            </div>

            {isPlaying && (
                <Progress value={((currentIndex + 1) / ROUND_LENGTH) * 100} className="h-2" />
            )}

            {!isPlaying ? (
                <div className="space-y-6 py-8">
                    <div className="space-y-2">
                        <h3 className="text-2xl font-bold text-foreground">2-Back Memory Challenge</h3>

                        <div className="bg-muted/30 p-4 rounded-xl text-left text-sm space-y-2 max-w-xs mx-auto">
                            <p className="font-semibold text-foreground">How to Play:</p>
                            <ul className="list-disc list-inside text-muted-foreground space-y-1">
                                <li>A series of letters will appear one by one.</li>
                                <li>Press <b>MATCH</b> if the current letter is the same as the one shown <b>2 steps ago</b>.</li>
                                <li>Don't press anything if it doesn't match.</li>
                            </ul>
                        </div>
                    </div>
                    <Button onClick={startGame} size="lg" className="w-48 gap-2 shadow-lg hover:scale-105 transition-transform">
                        <Play className="h-5 w-5" /> Start Training
                    </Button>
                </div>
            ) : (
                <div className="space-y-8 animate-in fade-in duration-300">
                    {/* The Stimulus */}
                    <div className={`
             h-48 w-48 mx-auto flex items-center justify-center rounded-2xl shadow-inner border-4 transition-all duration-300
             ${feedback === 'correct' ? 'bg-green-100 border-green-500 scale-105' :
                            feedback === 'wrong' ? 'bg-red-100 border-red-500 shake' : 'bg-white dark:bg-card border-primary/20'}
           `}>
                        <h1 className="text-8xl font-black text-primary">
                            {currentLetter}
                        </h1>
                    </div>

                    {/* Controls */}
                    <div className="flex justify-center gap-4">
                        <Button
                            size="lg"
                            className={`h-20 w-48 text-xl font-bold shadow-xl transition-all active:scale-95 ${currentIndex < N ? "opacity-50 cursor-not-allowed" : "hover:-translate-y-1 hover:shadow-2xl bg-gradient-to-r from-blue-600 to-indigo-600 border border-blue-400"
                                }`}
                            onClick={handleMatch}
                            disabled={currentIndex < N}
                        >
                            MATCH !
                        </Button>
                    </div>
                    <p className="text-xs text-muted-foreground animate-pulse">
                        New letter every 2.5 seconds...
                    </p>
                </div>
            )}
        </Card>
    );
}

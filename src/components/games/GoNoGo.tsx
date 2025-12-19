import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Trophy, RotateCcw } from "lucide-react";

export default function GoNoGo() {
    const [isPlaying, setIsPlaying] = useState(false);
    const [isFinished, setIsFinished] = useState(false);
    const [shape, setShape] = useState<"circle" | "square" | null>(null);
    const [score, setScore] = useState(0);
    const [round, setRound] = useState(0);
    const [feedback, setFeedback] = useState<"hit" | "miss" | "wrong" | null>(null);
    const [timeLeft, setTimeLeft] = useState(60);

    useEffect(() => {
        let interval: number;
        if (isPlaying && timeLeft > 0) {
            interval = window.setInterval(() => setTimeLeft(prev => prev - 1), 1000);
        } else if (isPlaying && timeLeft === 0) {
            setIsPlaying(false);
            setIsFinished(true);
            setShape(null);
        }
        return () => clearInterval(interval);
    }, [isPlaying, timeLeft]);

    useEffect(() => {
        if (!isPlaying || isFinished) return;

        const delay = 800 + Math.random() * 1000;
        const timer = setTimeout(() => {
            setShape(Math.random() > 0.3 ? "circle" : "square");
            setFeedback(null);
            // Auto hide
            setTimeout(() => {
                setShape(null);
                setRound(r => r + 1);
            }, 800); // 800ms visible
        }, delay);

        return () => clearTimeout(timer);
    }, [isPlaying, isFinished, round]);

    const startGame = () => {
        setIsPlaying(true);
        setIsFinished(false);
        setScore(0);
        setTimeLeft(60);
        setRound(0);
    };

    const handleClick = () => {
        if (!shape || isFinished) return;

        if (shape === "circle") {
            setScore(s => s + 10);
            setFeedback("hit");
            setShape(null); // Clear immediately on hit
        } else {
            setScore(s => Math.max(0, s - 10));
            setFeedback("wrong");
            setShape(null);
        }
    };

    if (isFinished) {
        return (
            <Card className="p-6 text-center space-y-6 flex flex-col items-center justify-center min-h-[400px]">
                <Trophy className="h-16 w-16 text-yellow-500 animate-bounce" />
                <div className="space-y-2">
                    <h3 className="text-3xl font-bold">Session Complete!</h3>
                    <p className="text-xl text-muted-foreground">Final Score: <span className="font-bold text-primary">{score}</span></p>
                </div>
                <Button onClick={startGame} size="lg" className="w-48 gap-2">
                    <RotateCcw className="h-4 w-4" /> Play Again
                </Button>
            </Card>
        );
    }

    return (
        <Card className="p-8 text-center min-h-[400px] flex flex-col items-center justify-center">
            {!isPlaying ? (
                <div className="space-y-6">
                    <div className="h-16 w-16 bg-green-500 rounded-full mx-auto" />
                    <h3 className="text-2xl font-bold">Go / No-Go</h3>

                    <div className="bg-muted/30 p-4 rounded-xl text-left text-sm space-y-2 max-w-xs mx-auto">
                        <p className="font-semibold text-foreground">How to Play:</p>
                        <ul className="list-disc list-inside text-muted-foreground space-y-1">
                            <li>Click <span className="text-green-600 font-bold">GREEN CIRCLE</span> immediately.</li>
                            <li>Ignore <span className="text-red-500 font-bold">RED SQUARE</span> (Do not click).</li>
                            <li>Be fast! You have 60 seconds.</li>
                        </ul>
                    </div>

                    <Button onClick={startGame} size="lg" className="w-full">Start Inhibition Task</Button>
                </div>
            ) : (
                <div
                    className="w-full h-full flex flex-col items-center justify-center cursor-pointer min-h-[300px] select-none"
                    onMouseDown={handleClick}
                >
                    <div className="absolute top-4 right-4 font-mono text-muted-foreground">Time: {timeLeft}s</div>

                    {shape === "circle" && (
                        <div className="h-40 w-40 bg-green-500 rounded-full shadow-xl animate-in zoom-in duration-100" />
                    )}
                    {shape === "square" && (
                        <div className="h-40 w-40 bg-red-500 rounded-xl shadow-xl animate-in zoom-in duration-100" />
                    )}

                    {feedback === "hit" && <div className="absolute top-20 text-green-500 font-bold text-xl animate-ping">NICE!</div>}
                    {feedback === "wrong" && <div className="absolute top-20 text-red-500 font-bold text-xl animate-shake">OOPS!</div>}

                    <div className="absolute bottom-4 font-mono text-xl">Score: {score}</div>
                    <div className="text-sm text-muted-foreground mt-8">Tap quickly on Green!</div>
                </div>
            )}
        </Card>
    );
}

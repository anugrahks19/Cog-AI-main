import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Play, RotateCcw, Trophy } from "lucide-react";

type Color = "red" | "blue" | "green" | "yellow" | "purple" | "orange";
const COLORS: Color[] = ["red", "blue", "green", "yellow", "purple", "orange"];

export default function StroopTest() {
    const [isPlaying, setIsPlaying] = useState(false);
    const [isFinished, setIsFinished] = useState(false);
    const [score, setScore] = useState(0);
    const [timeLeft, setTimeLeft] = useState(30);
    const [currentWord, setCurrentWord] = useState<Color>("red");
    const [currentColor, setCurrentColor] = useState<Color>("blue");

    const timerRef = useRef<number | null>(null);

    useEffect(() => {
        if (isPlaying && timeLeft > 0) {
            timerRef.current = window.setInterval(() => {
                setTimeLeft((prev) => prev - 1);
            }, 1000);
        } else if (isPlaying && timeLeft === 0) {
            endGame();
        }
        return () => { if (timerRef.current) clearInterval(timerRef.current) };
    }, [isPlaying, timeLeft]);

    const startGame = () => {
        setIsPlaying(true);
        setIsFinished(false);
        setScore(0);
        setTimeLeft(30); // 30 second rounds
        nextRound();
    };

    const endGame = () => {
        setIsPlaying(false);
        setIsFinished(true);
        if (timerRef.current) clearInterval(timerRef.current);
    };

    const nextRound = () => {
        const word = COLORS[Math.floor(Math.random() * COLORS.length)];
        const color = COLORS[Math.floor(Math.random() * COLORS.length)];
        setCurrentWord(word);
        setCurrentColor(color);
    };

    const handleAnswer = (match: boolean) => {
        const isMatch = currentWord === currentColor;
        if (match === isMatch) {
            setScore((prev) => prev + 10);
        } else {
            setScore((prev) => Math.max(0, prev - 5));
        }
        nextRound();
    };

    if (isFinished) {
        return (
            <Card className="p-6 text-center space-y-6 flex flex-col items-center justify-center min-h-[350px]">
                <Trophy className="h-16 w-16 text-yellow-500 animate-bounce" />
                <div className="space-y-2">
                    <h3 className="text-3xl font-bold">Stroop Session Complete!</h3>
                    <p className="text-xl text-muted-foreground">Focus Score: <span className="font-bold text-primary">{score}</span></p>
                </div>
                <Button onClick={startGame} size="lg" className="w-48 gap-2">
                    <RotateCcw className="h-4 w-4" /> Train Again
                </Button>
            </Card>
        );
    }

    return (
        <Card className="p-6 text-center space-y-6 min-h-[400px] flex flex-col justify-center">
            {!isPlaying ? (
                <div className="space-y-6">
                    <div className="h-16 w-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto text-3xl">🎨</div>
                    <div className="bg-muted/30 p-4 rounded-xl text-left text-sm space-y-2 max-w-xs mx-auto">
                        <p className="font-semibold text-foreground">How to Play:</p>
                        <ul className="list-disc list-inside text-muted-foreground space-y-1">
                            <li>Look at the <b>INK COLOR</b> of the word.</li>
                            <li>Ignore what the word actually says.</li>
                            <li>Click <b>Match</b> if the ink color equals the word text.</li>
                            <li>Otherwise, click <b>Mismatch</b>.</li>
                        </ul>
                    </div>

                    <Button onClick={startGame} className="w-full gap-2" size="lg">
                        <Play className="h-4 w-4" /> Start Brain Training
                    </Button>
                </div>
            ) : (
                <>
                    <div className="flex justify-between items-center text-sm font-medium text-muted-foreground">
                        <span>Score: {score}</span>
                        <span>Time: {timeLeft}s</span>
                    </div>
                    <div className="space-y-8 animate-in fade-in zoom-in duration-300">
                        {/* The Stimulus */}
                        <div
                            className="h-40 flex items-center justify-center rounded-xl bg-white shadow-inner"
                            style={{ border: `4px solid ${currentColor === "yellow" ? "#facc15" : currentColor}` }} // colored border helper
                        >
                            <h1
                                className="text-6xl font-black uppercase tracking-wider transition-colors"
                                style={{ color: currentColor === "yellow" ? "#eab308" : currentColor }}
                            >
                                {currentWord}
                            </h1>
                        </div>

                        {/* Controls */}
                        <div className="grid grid-cols-2 gap-4">
                            <Button
                                variant="destructive"
                                size="lg"
                                className="h-16 text-xl"
                                onClick={() => handleAnswer(false)}
                            >
                                Mismatch
                            </Button>
                            <Button
                                variant="default" // Greenish in theme usually, or standard primary
                                className="h-16 text-xl bg-green-600 hover:bg-green-700 text-white"
                                onClick={() => handleAnswer(true)}
                            >
                                Match
                            </Button>
                        </div>
                    </div>
                </>
            )}
        </Card>
    );
}

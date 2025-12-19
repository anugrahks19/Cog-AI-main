import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Search, Trophy, RotateCcw } from "lucide-react";

export default function VisualSearchGame() {
    const [isPlaying, setIsPlaying] = useState(false);
    const [isFinished, setIsFinished] = useState(false);
    const [items, setItems] = useState<string[]>([]);
    const [targetIndex, setTargetIndex] = useState(0);
    const [score, setScore] = useState(0);
    const [timeLeft, setTimeLeft] = useState(30);

    const TARGET = "🍎";
    const DISTRACTOR = "🍅";
    const GRID_SIZE = 25; // 5x5

    useEffect(() => {
        let interval: number;
        if (isPlaying && timeLeft > 0) {
            interval = window.setInterval(() => setTimeLeft(t => t - 1), 1000);
        } else if (isPlaying && timeLeft === 0) {
            setIsPlaying(false);
            setIsFinished(true); // Trigger finish screen
        }
        return () => clearInterval(interval);
    }, [isPlaying, timeLeft]);

    const generateLevel = () => {
        const newItems = Array(GRID_SIZE).fill(DISTRACTOR);
        const idx = Math.floor(Math.random() * GRID_SIZE);
        newItems[idx] = TARGET;
        setItems(newItems);
        setTargetIndex(idx);
    };

    const startGame = () => {
        setIsPlaying(true);
        setIsFinished(false);
        setScore(0);
        setTimeLeft(30);
        generateLevel();
    };

    const handleItemClick = (index: number) => {
        if (!isPlaying) return;
        if (index === items.indexOf(TARGET)) {
            setScore(s => s + 1);
            generateLevel();
        } else {
            setScore(s => Math.max(0, s - 1));
        }
    };

    if (isFinished) {
        return (
            <Card className="p-6 text-center space-y-6 flex flex-col items-center justify-center min-h-[300px]">
                <Trophy className="h-16 w-16 text-yellow-500 animate-bounce" />
                <div className="space-y-2">
                    <h3 className="text-3xl font-bold">Time's Up!</h3>
                    <p className="text-xl text-muted-foreground">You found <span className="font-bold text-primary">{score}</span> targets.</p>
                </div>
                <Button onClick={startGame} size="lg" className="w-48 gap-2">
                    <RotateCcw className="h-4 w-4" /> Play Again
                </Button>
            </Card>
        );
    }

    return (
        <Card className="p-6 text-center space-y-4">
            <div className="flex justify-between font-bold text-muted-foreground">
                <span>Score: {score}</span>
                <span>Time: {timeLeft}s</span>
            </div>

            {!isPlaying ? (
                <div className="py-8 space-y-6">
                    <Search className="h-12 w-12 mx-auto text-primary" />
                    <h3 className="text-2xl font-bold">Visual Search</h3>

                    <div className="bg-muted/30 p-4 rounded-xl text-left text-sm space-y-2 max-w-xs mx-auto">
                        <p className="font-semibold text-foreground">How to Play:</p>
                        <ul className="list-disc list-inside text-muted-foreground space-y-1">
                            <li>Find the hidden <span className="text-xl">🍎</span> among the tomatoes.</li>
                            <li>Click it as fast as you can.</li>
                            <li>You have 30 seconds!</li>
                        </ul>
                    </div>

                    <Button onClick={startGame} size="lg" className="w-full">Start Scanning</Button>
                </div>
            ) : (
                <div className="grid grid-cols-5 gap-1 sm:gap-2 max-w-sm mx-auto">
                    {items.map((item, i) => (
                        <button
                            key={i}
                            onClick={() => handleItemClick(i)}
                            className="text-2xl sm:text-4xl hover:scale-110 hover:bg-muted rounded-lg p-1 sm:p-2 transition-transform h-12 w-12 sm:h-16 sm:w-16 flex items-center justify-center cursor-pointer select-none"
                        >
                            {item}
                        </button>
                    ))}
                </div>
            )}
        </Card>
    );
}

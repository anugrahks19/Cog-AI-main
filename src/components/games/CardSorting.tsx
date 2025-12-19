import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Trophy, RotateCcw } from "lucide-react";

export default function CardSorting() {
    const [isPlaying, setIsPlaying] = useState(false);
    const [isFinished, setIsFinished] = useState(false);
    const [score, setScore] = useState(0);
    const [rule, setRule] = useState<"color" | "shape">("color");
    const [message, setMessage] = useState("");
    const [timeLeft, setTimeLeft] = useState(60);

    const COLORS = ["red", "blue", "green", "yellow"];
    const SHAPES = ["circle", "square", "triangle", "star"];

    const [currentCard, setCurrentCard] = useState({ color: "red", shape: "circle" });

    useEffect(() => {
        let interval: number;
        if (isPlaying && timeLeft > 0) {
            interval = window.setInterval(() => setTimeLeft(t => t - 1), 1000);
        } else if (isPlaying && timeLeft === 0) {
            setIsPlaying(false);
            setIsFinished(true);
        }
        return () => clearInterval(interval);
    }, [isPlaying, timeLeft]);

    useEffect(() => {
        // Change rule every 5 points implicitly
        if (score > 0 && score % 5 === 0) {
            setRule(r => r === "color" ? "shape" : "color");
            setMessage("Rule Changed!");
            setTimeout(() => setMessage(""), 1000);
        }
        generateCard();
    }, [score]);

    const generateCard = () => {
        setCurrentCard({
            color: COLORS[Math.floor(Math.random() * COLORS.length)],
            shape: SHAPES[Math.floor(Math.random() * SHAPES.length)]
        });
    };

    const startGame = () => {
        setIsPlaying(true);
        setIsFinished(false);
        setScore(0);
        setTimeLeft(60);
        generateCard();
    };

    const handleMatch = (type: "color" | "shape") => {
        if (!isPlaying) return;

        if (type === rule) {
            setScore(s => s + 1);
            setMessage("Correct!");
        } else {
            setMessage("Wrong Rule!");
        }
        setTimeout(() => setMessage(""), 500);
        generateCard();
    };

    if (isFinished) {
        return (
            <Card className="p-6 text-center space-y-6 flex flex-col items-center justify-center min-h-[400px]">
                <Trophy className="h-16 w-16 text-yellow-500 animate-bounce" />
                <div className="space-y-2">
                    <h3 className="text-3xl font-bold">Time's Up!</h3>
                    <p className="text-xl text-muted-foreground">Cards Sorted: <span className="font-bold text-primary">{score}</span></p>
                </div>
                <Button onClick={startGame} size="lg" className="w-48 gap-2">
                    <RotateCcw className="h-4 w-4" /> Play Again
                </Button>
            </Card>
        );
    }

    return (
        <Card className="p-8 text-center min-h-[400px] flex flex-col items-center justify-center space-y-8">
            {!isPlaying ? (
                <div className="space-y-6">
                    <h3 className="text-2xl font-bold">Rule Shift</h3>

                    <div className="bg-muted/30 p-4 rounded-xl text-left text-sm space-y-2 max-w-xs mx-auto">
                        <p className="font-semibold text-foreground">How to Play:</p>
                        <ul className="list-disc list-inside text-muted-foreground space-y-1">
                            <li>Guess how to sort the card: by <b>Color</b> or <b>Shape</b>.</li>
                            <li>If "Correct", keep sorting that way.</li>
                            <li>If "Wrong Rule", switch to the other method!</li>
                            <li>The rule changes secretly every 5 points.</li>
                        </ul>
                    </div>

                    <Button onClick={startGame} size="lg">Start Sorting</Button>
                </div>
            ) : (
                <>
                    <div className="flex justify-between w-full font-bold text-muted-foreground px-8 absolute top-4">
                        <span>Score: {score}</span>
                        <span>Time: {timeLeft}s</span>
                    </div>

                    <div className="space-y-2 mt-8">
                        <h3 className="text-xl font-bold">Sort This Card</h3>
                    </div>

                    {/* The Card */}
                    <div
                        className={`h-40 w-32 rounded-xl border-4 shadow-xl flex items-center justify-center bg-white dark:bg-card transition-all duration-300
                    ${message === "Correct!" ? "border-green-500 scale-105" : message === "Wrong Rule!" ? "border-red-500 shake" : "border-gray-200"}
                  `}
                    >
                        <div
                            className={`h-20 w-20 
                            ${currentCard.shape === "circle" ? "rounded-full" : currentCard.shape === "square" ? "rounded-none" : "rounded-lg rotate-45"}
                        `}
                            style={{ backgroundColor: currentCard.color === "red" ? "#ef4444" : currentCard.color === "blue" ? "#3b82f6" : currentCard.color === "green" ? "#22c55e" : "#eab308" }}
                        />
                    </div>

                    <div className={`text-xl font-bold h-8 transition-colors ${message === "Correct!" ? "text-green-600" : "text-red-500"}`}>
                        {message}
                    </div>

                    <div className="grid grid-cols-2 gap-4 w-full max-w-xs">
                        <Button variant="outline" className="h-16 text-lg border-2 hover:bg-muted" onClick={() => handleMatch("color")}>
                            Match Color
                        </Button>
                        <Button variant="outline" className="h-16 text-lg border-2 hover:bg-muted" onClick={() => handleMatch("shape")}>
                            Match Shape
                        </Button>
                    </div>
                </>
            )}
        </Card>
    );
}

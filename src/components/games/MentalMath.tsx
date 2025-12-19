import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Calculator, Trophy, RotateCcw } from "lucide-react";

export default function MentalMath() {
    const [isPlaying, setIsPlaying] = useState(false);
    const [isFinished, setIsFinished] = useState(false);
    const [problem, setProblem] = useState({ q: "", a: 0 });
    const [input, setInput] = useState("");
    const [score, setScore] = useState(0);
    const [timeLeft, setTimeLeft] = useState(60);

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

    const generateProblem = () => {
        const ops = ["+", "-", "*"];
        const op = ops[Math.floor(Math.random() * ops.length)];
        const n1 = Math.floor(Math.random() * 20) + 1;
        const n2 = Math.floor(Math.random() * 10) + 1;

        let q = `${n1} ${op} ${n2}`;
        let a = 0;
        if (op === "+") a = n1 + n2;
        if (op === "-") a = n1 - n2;
        if (op === "*") a = n1 * n2;

        setProblem({ q, a });
        setInput("");
    };

    const startGame = () => {
        setIsPlaying(true);
        setIsFinished(false);
        setScore(0);
        setTimeLeft(60);
        generateProblem();
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (parseInt(input) === problem.a) {
            setScore(s => s + 10);
        } else {
            setScore(s => Math.max(0, s - 5));
        }
        generateProblem();
    };

    if (isFinished) {
        return (
            <Card className="p-6 text-center space-y-6 flex flex-col items-center justify-center min-h-[350px]">
                <Trophy className="h-16 w-16 text-yellow-500 animate-bounce" />
                <div className="space-y-2">
                    <h3 className="text-3xl font-bold">Calculation Complete!</h3>
                    <p className="text-xl text-muted-foreground">Total Score: <span className="font-bold text-primary">{score}</span></p>
                </div>
                <Button onClick={startGame} size="lg" className="w-48 gap-2">
                    <RotateCcw className="h-4 w-4" /> Play Again
                </Button>
            </Card>
        );
    }

    return (
        <Card className="p-8 text-center min-h-[350px] flex flex-col items-center justify-center">
            {!isPlaying ? (
                <div className="space-y-6">
                    <Calculator className="h-12 w-12 mx-auto text-indigo-500" />
                    <h3 className="text-2xl font-bold">Speed Math</h3>

                    <div className="bg-muted/30 p-4 rounded-xl text-left text-sm space-y-2 max-w-xs mx-auto">
                        <p className="font-semibold text-foreground">How to Play:</p>
                        <ul className="list-disc list-inside text-muted-foreground space-y-1">
                            <li>Solve the arithmetic problems.</li>
                            <li>Type your answer and press <b>Enter</b> or <b>Go</b>.</li>
                            <li>Correct = +10 pts, Wrong = -5 pts.</li>
                        </ul>
                    </div>

                    <Button onClick={startGame} size="lg">Start Calculation</Button>
                </div>
            ) : (
                <div className="space-y-6 w-full max-w-xs mx-auto">
                    <div className="flex justify-between font-bold text-muted-foreground">
                        <span>Score: {score}</span>
                        <span>Time: {timeLeft}s</span>
                    </div>

                    <div className="text-5xl font-black text-foreground py-8">
                        {problem.q}
                    </div>

                    <form onSubmit={handleSubmit} className="flex gap-2">
                        <Input
                            autoFocus
                            type="number"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            className="text-center text-2xl h-14"
                            placeholder="?"
                        />
                        <Button type="submit" size="lg" className="h-14 px-8">Go</Button>
                    </form>
                </div>
            )}
        </Card>
    );
}

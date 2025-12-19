import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { BrainCircuit, Trophy, RotateCcw } from "lucide-react";

export default function SequenceMemory() {
    const [isPlaying, setIsPlaying] = useState(false);
    const [isFinished, setIsFinished] = useState(false);
    const [sequence, setSequence] = useState<number[]>([]);
    const [userSequence, setUserSequence] = useState<number[]>([]);
    const [isShowingSequence, setIsShowingSequence] = useState(false);
    const [score, setScore] = useState(0);
    const [activeSquare, setActiveSquare] = useState<number | null>(null);

    const COLORS = ["bg-red-500", "bg-blue-500", "bg-green-500", "bg-yellow-500"];

    const startGame = () => {
        setIsPlaying(true);
        setIsFinished(false);
        setSequence([]);
        setUserSequence([]);
        setScore(0);
        addToSequence([]);
    };

    const addToSequence = async (currentSeq: number[]) => {
        setIsShowingSequence(true);
        const nextColor = Math.floor(Math.random() * 4);
        const newSeq = [...currentSeq, nextColor];
        setSequence(newSeq);
        setUserSequence([]);

        // Play sequence
        for (let i = 0; i < newSeq.length; i++) {
            await new Promise(r => setTimeout(r, 500));
            setActiveSquare(newSeq[i]);
            await new Promise(r => setTimeout(r, 500));
            setActiveSquare(null);
        }
        setIsShowingSequence(false);
    };

    const handleSquareClick = (index: number) => {
        if (isShowingSequence || !isPlaying) return;

        setActiveSquare(index);
        setTimeout(() => setActiveSquare(null), 200);

        const newUserSeq = [...userSequence, index];
        setUserSequence(newUserSeq);

        // Initial check
        if (newUserSeq[newUserSeq.length - 1] !== sequence[newUserSeq.length - 1]) {
            // Game Over
            setIsPlaying(false);
            setIsFinished(true); // Trigger finish screen
            setActiveSquare(null);
            return;
        }

        // Complete sequence check
        if (newUserSeq.length === sequence.length) {
            setScore(sequence.length);
            setTimeout(() => addToSequence(sequence), 1000);
        }
    };

    if (isFinished) {
        return (
            <Card className="p-6 text-center space-y-6 flex flex-col items-center justify-center min-h-[400px]">
                <Trophy className="h-16 w-16 text-red-500 animate-pulse" />
                <div className="space-y-2">
                    <h3 className="text-3xl font-bold">Game Over</h3>
                    <p className="text-xl text-muted-foreground">You memorized <span className="font-bold text-primary">{score}</span> steps.</p>
                </div>
                <Button onClick={startGame} size="lg" className="w-48 gap-2">
                    <RotateCcw className="h-4 w-4" /> Try Again
                </Button>
            </Card>
        );
    }

    return (
        <Card className="p-6 text-center space-y-6 min-h-[400px] flex flex-col items-center justify-center">
            {!isPlaying ? (
                <div className="space-y-6">
                    <BrainCircuit className="h-12 w-12 mx-auto text-purple-500" />
                    <h3 className="text-2xl font-bold">Sequence Memory</h3>

                    <div className="bg-muted/30 p-4 rounded-xl text-left text-sm space-y-2 max-w-xs mx-auto">
                        <p className="font-semibold text-foreground">How to Play:</p>
                        <ul className="list-disc list-inside text-muted-foreground space-y-1">
                            <li>Watch the colored blocks light up.</li>
                            <li>Repeat the exact pattern by clicking them.</li>
                            <li>The sequence gets longer each round!</li>
                        </ul>
                    </div>

                    <Button onClick={startGame} size="lg">Start Game</Button>
                </div>
            ) : (
                <div className="space-y-4">
                    <h3 className="text-xl font-bold text-muted-foreground">Level {sequence.length}</h3>
                    <div className="grid grid-cols-2 gap-4 w-64 mx-auto">
                        {COLORS.map((color, idx) => (
                            <div
                                key={idx}
                                onClick={() => handleSquareClick(idx)}
                                className={`h-32 w-32 rounded-2xl cursor-pointer transition-opacity duration-100 shadow-md ${color} 
                        ${activeSquare === idx ? "opacity-100 scale-105 brightness-125" : "opacity-60 hover:opacity-80"}`}
                            />
                        ))}
                    </div>
                    <p className="text-sm text-muted-foreground">{isShowingSequence ? "Watch..." : "Your Turn"}</p>
                </div>
            )}
        </Card>
    );
}

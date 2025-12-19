import { useState, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Zap, RotateCcw } from "lucide-react";

export default function ReactionTest() {
    const [status, setStatus] = useState<"idle" | "waiting" | "ready" | "finished">("idle");
    const [startTime, setStartTime] = useState(0);
    const [reactionTime, setReactionTime] = useState<number | null>(null);
    const timeoutRef = useRef<number | null>(null);

    const startTest = () => {
        setStatus("waiting");
        setReactionTime(null);
        const delay = Math.floor(Math.random() * 2000) + 1000; // 1-3 seconds
        timeoutRef.current = window.setTimeout(() => {
            setStatus("ready");
            setStartTime(Date.now());
        }, delay);
    };

    const handleClick = () => {
        if (status === "waiting") {
            // Too early
            clearTimeout(timeoutRef.current!);
            setStatus("idle");
            alert("Too early! Wait for green.");
            return;
        }

        if (status === "ready") {
            const time = Date.now() - startTime;
            setReactionTime(time);
            setStatus("finished");
        }
    };

    if (status === "finished") {
        return (
            <Card className="p-6 text-center space-y-6 flex flex-col items-center justify-center min-h-[300px]">
                <Zap className="h-16 w-16 text-yellow-500" />
                <div className="space-y-2">
                    <h3 className="text-3xl font-bold">Test Complete</h3>
                    <div className="text-6xl font-bold text-primary py-4">{reactionTime} ms</div>
                    <p className="text-sm text-muted-foreground">Average human reaction: 250ms</p>
                </div>
                <Button onClick={(e) => { e.stopPropagation(); startTest(); }} size="lg" className="w-48 gap-2">
                    <RotateCcw className="h-4 w-4" /> Try Again
                </Button>
            </Card>
        );
    }

    return (
        <Card
            className={`h-64 flex flex-col items-center justify-center cursor-pointer transition-colors duration-200 select-none min-h-[300px]
        ${status === "waiting" ? "bg-red-500 hover:bg-red-600 border-red-600" :
                    status === "ready" ? "bg-green-500 border-green-600" : "bg-white/80"}`}
            onClick={(status === "waiting" || status === "ready") ? handleClick : undefined}
        >
            {status === "idle" && (
                <div className="text-center space-y-6 p-4">
                    <Zap className="h-12 w-12 mx-auto text-yellow-500" />
                    <h3 className="text-2xl font-bold">Reaction Time</h3>

                    <div className="bg-muted/30 p-4 rounded-xl text-left text-sm space-y-2 max-w-xs mx-auto">
                        <p className="font-semibold text-foreground">How to Play:</p>
                        <ul className="list-disc list-inside text-muted-foreground space-y-1">
                            <li>Wait for the screen to turn <span className="text-green-600 font-bold">GREEN</span>.</li>
                            <li>Click anywhere as fast as you can!</li>
                        </ul>
                    </div>

                    <Button onClick={(e) => { e.stopPropagation(); startTest(); }} size="lg" className="w-full">Start Test</Button>
                </div>
            )}

            {status === "waiting" && (
                <h3 className="text-3xl font-bold text-white animate-pulse">Wait for it...</h3>
            )}

            {status === "ready" && (
                <h3 className="text-5xl font-black text-white uppercase">CLICK NOW!</h3>
            )}
        </Card>
    );
}

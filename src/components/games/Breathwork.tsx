import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Wind, PlayCircle } from "lucide-react";

export default function Breathwork() {
    const [isActive, setIsActive] = useState(false);
    const [phase, setPhase] = useState<"Inhale" | "Hold" | "Exhale">("Inhale");
    const [timeLeft, setTimeLeft] = useState(60);

    useEffect(() => {
        if (!isActive) return;

        // Timer for session
        const sessionTimer = setInterval(() => {
            setTimeLeft((prev) => {
                if (prev <= 1) {
                    setIsActive(false);
                    return 60;
                }
                return prev - 1;
            });
        }, 1000);

        // Breathing Cycle: 4-4-4 (Box Breathing simplified)
        // Actually, usually Inhale (4), Hold (4), Exhale (4), Hold (4).
        // Let's do Inhale (4), Exhale (4) for simplicity or 4-7-8.
        // Let's do simple 4-4-4 for visual symmetry.

        const cycle = async () => {
            while (isActive) {
                setPhase("Inhale");
                await new Promise(r => setTimeout(r, 4000));
                if (!isActive) break;
                setPhase("Hold");
                await new Promise(r => setTimeout(r, 4000));
                if (!isActive) break;
                setPhase("Exhale");
                await new Promise(r => setTimeout(r, 4000));
            }
        };

        // This hook structure is a bit tricky for async loops. 
        // Let's use CSS animations for the visual and just simple text updates via interval if needed.
        // Actually, CSS is better for smoothness.

        return () => clearInterval(sessionTimer);
    }, [isActive]);

    return (
        <Card className="min-h-[400px] flex flex-col items-center justify-center p-8 bg-gradient-to-b from-cyan-50 to-blue-50 relative overflow-hidden">
            {!isActive ? (
                <div className="text-center space-y-6 z-10">
                    <div className="h-24 w-24 bg-cyan-100 rounded-full flex items-center justify-center mx-auto mb-4">
                        <Wind className="h-10 w-10 text-cyan-600" />
                    </div>
                    <h3 className="text-2xl font-semibold text-cyan-900">4-7-8 Relaxation</h3>
                    <p className="text-cyan-700/80 max-w-xs mx-auto">
                        Reduce anxiety and improve focus with this 60-second guided breathing session.
                    </p>
                    <Button
                        onClick={() => setIsActive(true)}
                        className="bg-cyan-600 hover:bg-cyan-700 text-white rounded-full px-8"
                    >
                        <PlayCircle className="mr-2 h-5 w-5" /> Begin
                    </Button>
                </div>
            ) : (
                <div className="relative z-10 flex flex-col items-center">
                    {/* Breathing Circle Animation */}
                    <div className="relative h-64 w-64 flex items-center justify-center">
                        <div className="absolute inset-0 bg-cyan-400/20 rounded-full animate-ping-slow"></div>
                        <div className="absolute inset-0 bg-blue-400/20 rounded-full animate-pulse-slow delay-75"></div>
                        <div className="h-48 w-48 bg-white/80 dark:bg-card/80 backdrop-blur-md rounded-full shadow-lg flex items-center justify-center z-20 animate-breathe">
                            <span className="text-2xl font-bold text-cyan-800 tracking-widest">
                                BREATHE
                            </span>
                        </div>
                    </div>

                    <p className="mt-8 text-cyan-900 font-medium">Session ends in {timeLeft}s</p>
                    <Button variant="ghost" className="mt-4 text-cyan-700 hover:text-cyan-900" onClick={() => setIsActive(false)}>
                        Stop
                    </Button>
                </div>
            )}
        </Card>
    );
}

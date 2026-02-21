import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dumbbell, Brain, Zap, Activity, Flame, Trophy, Wind, BookOpen, Calculator, Target } from "lucide-react";
import StroopTest from "@/components/games/StroopTest";
import NBackGame from "@/components/games/NBackGame";
import Breathwork from "@/components/games/Breathwork";
import ReactionTest from "@/components/games/ReactionTest";
import VisualSearchGame from "@/components/games/VisualSearchGame";
import SequenceMemory from "@/components/games/SequenceMemory";
import WordRecall from "@/components/games/WordRecall";
import GoNoGo from "@/components/games/GoNoGo";
import JigsawPuzzle from "@/components/games/JigsawPuzzle";
import MentalMath from "@/components/games/MentalMath";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useState } from "react";

export default function BrainGym() {
    const [streak] = useState(3);
    const [xp] = useState(1450);
    const [level] = useState(6);

    return (
        <div className="min-h-screen relative overflow-hidden">
            {/* Removed opacity-30/bg-fixed/bg-background to let particles show */}
            <div className="absolute -top-20 -left-20 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-pulse delay-1000" />
            <div className="absolute top-40 right-20 w-72 h-72 bg-secondary/10 rounded-full blur-3xl animate-pulse" />

            <div className="max-w-7xl mx-auto px-4 py-8 relative z-10">

                {/* Engagement Header */}
                <div className="flex flex-col md:flex-row justify-between items-center bg-white/60 dark:bg-card/60 backdrop-blur-xl border border-white/40 dark:border-white/10 rounded-2xl p-4 mb-8 shadow-sm animate-fade-in">
                    <div className="flex items-center gap-4 mb-4 md:mb-0">
                        <div className="p-3 bg-gradient-to-br from-orange-400 to-red-500 text-white rounded-xl shadow-md">
                            <Flame className="h-6 w-6 animate-pulse" />
                        </div>
                        <div>
                            <div className="text-2xl font-bold text-foreground">{streak} Day Streak</div>
                            <div className="text-xs text-muted-foreground">Level {level} Brain Athlete</div>
                        </div>
                    </div>

                    <div className="flex items-center gap-6 w-full md:w-auto">
                        <div className="flex-1 md:w-64">
                            <div className="flex justify-between text-xs font-semibold mb-1">
                                <span>XP Progress</span>
                                <span className="text-primary">{xp} / 2000</span>
                            </div>
                            <Progress value={(xp / 2000) * 100} className="h-2.5" />
                        </div>
                        <div className="p-2 bg-yellow-100 rounded-full text-yellow-700">
                            <Trophy className="h-5 w-5" />
                        </div>
                    </div>
                </div>

                <header className="text-center space-y-4 mb-12">
                    <Badge variant="outline" className="px-4 py-1.5 text-sm bg-purple-100/50 text-purple-800 border-purple-200 backdrop-blur-md shadow-sm">
                        Full Access • 10 Modules Unlocked
                    </Badge>
                    <h1 className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-purple-600 pb-2">
                        Neuro-Gym Library
                    </h1>
                </header>

                <Tabs defaultValue="attention" className="w-full">
                    <div className="flex justify-center mb-8 overflow-x-auto pb-2">
                        <TabsList className="flex w-max p-1.5 bg-muted/50 backdrop-blur-sm border border-white/20 rounded-xl">
                            <TabsTrigger value="attention" className="rounded-lg gap-2 min-w-[100px]"><Zap className="h-4 w-4" /> Attention</TabsTrigger>
                            <TabsTrigger value="memory" className="rounded-lg gap-2 min-w-[100px]"><Brain className="h-4 w-4" /> Memory</TabsTrigger>
                            <TabsTrigger value="executive" className="rounded-lg gap-2 min-w-[100px]"><Activity className="h-4 w-4" /> Executive</TabsTrigger>
                            <TabsTrigger value="logic" className="rounded-lg gap-2 min-w-[100px]"><Calculator className="h-4 w-4" /> Logic</TabsTrigger>
                            <TabsTrigger value="relax" className="rounded-lg gap-2 min-w-[100px]"><Wind className="h-4 w-4" /> Relax</TabsTrigger>
                        </TabsList>
                    </div>

                    {/* ATTENTION GAMES */}
                    <TabsContent value="attention" className="space-y-6 animate-slide-up">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <GameCard title="Reaction Time" desc="Test your raw reflexes" icon={Zap} component={<ReactionTest />} />
                            <GameCard title="Stroop Test" desc="Don't get tricked by the color!" icon={Dumbbell} component={<StroopTest />} />
                            <div className="md:col-span-2">
                                <GameCard title="Visual Search" desc="Find targets in a cluttered field" icon={Target} component={<VisualSearchGame />} />
                            </div>
                        </div>
                    </TabsContent>

                    {/* MEMORY GAMES */}
                    <TabsContent value="memory" className="space-y-6 animate-slide-up">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <GameCard title="N-Back" desc="Working memory challenge" icon={Brain} component={<NBackGame />} />
                            <GameCard title="Sequence Memory" desc="Recall patterns (Simon)" icon={Activity} component={<SequenceMemory />} />
                            <div className="md:col-span-2">
                                <GameCard title="Word Recall" desc="Short-term verbal retention" icon={BookOpen} component={<WordRecall />} />
                            </div>
                        </div>
                    </TabsContent>

                    {/* EXECUTIVE FUNCTION GAMES */}
                    <TabsContent value="executive" className="space-y-6 animate-slide-up">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <GameCard title="Go / No-Go" desc="Impulse control & inhibition" icon={Zap} component={<GoNoGo />} />
                            <GameCard title="Jigsaw Puzzle" desc="Spatial reasoning & logic" icon={Dumbbell} component={<JigsawPuzzle />} />
                        </div>
                    </TabsContent>

                    {/* LOGIC & SPEED */}
                    <TabsContent value="logic" className="space-y-6 animate-slide-up">
                        <div className="max-w-3xl mx-auto">
                            <GameCard title="Speed Math" desc="Rapid arithmetic processing" icon={Calculator} component={<MentalMath />} />
                        </div>
                    </TabsContent>

                    {/* RELAX */}
                    <TabsContent value="relax" className="animate-slide-up">
                        <div className="max-w-2xl mx-auto">
                            <GameCard title="4-7-8 Breathing" desc="Reduce anxiety and reset" icon={Wind} component={<Breathwork />} />
                        </div>
                    </TabsContent>
                </Tabs>
            </div>
        </div>
    );
}

function GameCard({ title, desc, icon: Icon, component }: { title: string, desc: string, icon: any, component: React.ReactNode }) {
    return (
        <div className="bg-white/80 dark:bg-card/80 backdrop-blur-xl rounded-2xl shadow-lg border border-white/50 dark:border-white/10 overflow-hidden flex flex-col h-full">
            <div className="p-4 bg-muted/30 border-b border-white/20 flex items-center gap-3">
                <div className="p-2 bg-primary/10 rounded-lg text-primary">
                    <Icon className="h-5 w-5" />
                </div>
                <div>
                    <h3 className="font-bold text-lg leading-none">{title}</h3>
                    <p className="text-xs text-muted-foreground">{desc}</p>
                </div>
            </div>
            <div className="p-1 bg-muted/10 flex-1">
                {component}
            </div>
        </div>
    );
}

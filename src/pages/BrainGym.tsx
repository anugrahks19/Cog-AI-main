import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dumbbell, Brain, Zap, Activity, Flame, Trophy, Wind, BookOpen, Calculator, Target, Star } from "lucide-react";
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

                {/* Engagement Dashboard */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-12 animate-fade-in">
                    {/* Card 1: Streak */}
                    <div className="bg-white/80 dark:bg-[#1a172c]/80 backdrop-blur-xl border border-slate-200 dark:border-white/5 rounded-3xl p-6 shadow-xl relative overflow-hidden flex flex-col justify-between">
                        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-indigo-500/5 dark:from-indigo-500/10 to-transparent pointer-events-none" />
                        <div className="flex items-center gap-4 mb-8 relative z-10">
                            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#7e57c2]/20 dark:from-[#7e57c2]/40 to-[#5e35b1]/20 dark:to-[#5e35b1]/40 flex items-center justify-center border border-indigo-200 dark:border-white/10 shadow-[inset_0_2px_10px_rgba(255,255,255,0.1)]">
                                <Flame className="h-8 w-8 text-indigo-600 dark:text-white" />
                            </div>
                            <div>
                                <h3 className="text-2xl font-bold text-slate-800 dark:text-white flex items-center gap-2">
                                    {streak} Day Streak <span className="text-orange-500 animate-pulse text-xl">🔥</span>
                                </h3>
                                <p className="text-sm text-slate-500 dark:text-slate-400">Keep it going!</p>
                            </div>
                        </div>
                        <div className="flex justify-between items-center relative z-10">
                            {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((day, i) => {
                                const isStreakActive = i >= 4; // Visual mock for Fri, Sat, Sun
                                return (
                                    <div key={day} className="flex flex-col items-center gap-2">
                                        <div className={`w-10 h-10 sm:w-11 sm:h-11 rounded-full flex items-center justify-center border transition-all ${isStreakActive ? 'bg-[#c58dfc] border-transparent shadow-[0_0_20px_rgba(197,141,252,0.4)]' : 'bg-slate-100 dark:bg-[#1e1b33] border-slate-200 dark:border-transparent shadow-inner'}`}>
                                            {isStreakActive ? <Flame className="h-5 w-5 text-white" /> : <div className="w-1.5 h-1.5 rounded-full bg-slate-300 dark:bg-slate-600" />}
                                        </div>
                                        <span className="text-xs text-slate-500 font-medium">{day}</span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* Card 2: Level */}
                    <div className="bg-white/80 dark:bg-[#1a172c]/80 backdrop-blur-xl border border-slate-200 dark:border-white/5 rounded-3xl p-6 shadow-xl relative overflow-hidden flex flex-col justify-between">
                        <div className="flex justify-between items-start mb-6 relative z-10">
                            <div className="flex items-center gap-4">
                                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-purple-500/10 dark:from-purple-500/30 to-[#ba68c8]/10 dark:to-[#ba68c8]/30 flex items-center justify-center border border-purple-200 dark:border-white/10 shadow-[inset_0_2px_10px_rgba(255,255,255,0.1)]">
                                    <Trophy className="h-6 w-6 text-purple-600 dark:text-white" />
                                </div>
                                <div>
                                    <h3 className="text-2xl font-bold text-slate-800 dark:text-white">Level {level}</h3>
                                    <p className="text-sm text-slate-500 dark:text-slate-400">Brain Athlete</p>
                                </div>
                            </div>
                            <div className="bg-indigo-50 dark:bg-indigo-500/20 px-4 py-1.5 rounded-full border border-indigo-200 dark:border-indigo-500/30">
                                <span className="text-sm font-bold text-indigo-600 dark:text-indigo-300">{xp} XP</span>
                            </div>
                        </div>
                        <div className="relative z-10">
                            <div className="flex justify-between text-sm mb-3 text-slate-600 dark:text-slate-300">
                                <span className="font-medium">Next: Level 7</span>
                                <span className="font-medium">73%</span>
                            </div>
                            <div className="h-2.5 rounded-full bg-slate-100 dark:bg-[#1e1b33] mb-3 overflow-hidden border border-slate-200 dark:border-white/5 shadow-inner">
                                <div className="h-full bg-gradient-to-r from-[#4dd0e1] to-[#81d4fa] rounded-full shadow-[0_0_15px_rgba(77,208,225,0.6)]" style={{ width: '73%' }} />
                            </div>
                            <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 font-medium">
                                <span>{xp} XP Current</span>
                                <span className="text-indigo-600 dark:text-indigo-400">550 to Mastery</span>
                            </div>
                        </div>
                    </div>

                    {/* Card 3: Recent Gains */}
                    <div className="bg-white/80 dark:bg-[#1a172c]/80 backdrop-blur-xl border border-slate-200 dark:border-white/5 rounded-3xl p-6 shadow-xl relative overflow-hidden flex flex-col">
                        <div className="flex items-center gap-2 mb-5 relative z-10">
                            <Star className="h-5 w-5 text-purple-600 dark:text-purple-400 fill-purple-600 dark:fill-purple-400" />
                            <h3 className="font-semibold text-slate-800 dark:text-white">Recent Gains</h3>
                        </div>
                        <div className="space-y-4 relative z-10">
                            {[
                                { name: "Stroop Test", time: "2 mins ago", xp: "+120", icon: <Zap className="h-3.5 w-3.5" /> },
                                { name: "Daily Streak Bonus", time: "1 hour ago", xp: "+50", icon: <Flame className="h-3.5 w-3.5" /> },
                                { name: "N-Back", time: "3 hours ago", xp: "+200", icon: <Brain className="h-3.5 w-3.5" /> },
                            ].map((gain, i) => (
                                <div key={i} className={`flex items-center justify-between ${i !== 2 ? 'pb-4 border-b border-slate-100 dark:border-white/5' : ''}`}>
                                    <div className="flex items-center gap-3">
                                        <div className="w-9 h-9 rounded-full bg-indigo-50 dark:bg-indigo-500/10 flex items-center justify-center text-indigo-600 dark:text-indigo-400 border border-indigo-100 dark:border-indigo-500/20">
                                            {gain.icon}
                                        </div>
                                        <div>
                                            <div className="text-sm font-semibold text-slate-800 dark:text-slate-200 leading-tight mb-0.5">{gain.name}</div>
                                            <div className="text-xs text-slate-500">{gain.time}</div>
                                        </div>
                                    </div>
                                    <div className="font-bold text-sm text-slate-600 dark:text-slate-300">{gain.xp}</div>
                                </div>
                            ))}
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

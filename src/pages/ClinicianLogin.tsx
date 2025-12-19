import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Lock, Stethoscope, ArrowRight, Loader2 } from "lucide-react";
import { toast } from "sonner";

interface Props {
    onLogin: () => void;
}

export default function ClinicianLogin({ onLogin }: Props) {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    const handleLogin = (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);

        // Mock authentication delay
        setTimeout(() => {
            if (email === "doctor@cog.ai" && password === "demo") {
                toast.success("Welcome back, Dr. Iyer");
                onLogin();
            } else {
                toast.error("Invalid credentials. Try doctor@cog.ai / demo");
            }
            setIsLoading(false);
        }, 1500);
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-background relative overflow-hidden">
            {/* Background Gradients */}
            <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-background to-secondary/5" />
            <div className="absolute top-[-10%] right-[-5%] w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-pulse" />
            <div className="absolute bottom-[-10%] left-[-5%] w-96 h-96 bg-secondary/10 rounded-full blur-3xl animate-pulse delay-700" />

            <Card className="w-full max-w-md relative z-10 shadow-2xl border-white/20 backdrop-blur-xl bg-white/50">
                <CardHeader className="space-y-4 text-center pb-2">
                    <div className="mx-auto w-16 h-16 bg-primary/10 rounded-2xl flex items-center justify-center mb-2">
                        <Stethoscope className="h-8 w-8 text-primary" />
                    </div>
                    <CardTitle className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-purple-600">
                        Clinician Portal
                    </CardTitle>
                    <CardDescription className="text-base">
                        Secure access for authorized medical personnel only.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <form onSubmit={handleLogin} className="space-y-5">
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-muted-foreground ml-1">Email ID</label>
                            <div className="relative">
                                <Input
                                    type="email"
                                    placeholder="doctor@cog.ai"
                                    className="pl-10 h-11 bg-white/60 border-gray-200 focus:bg-white transition-all"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    required
                                />
                                <Stethoscope className="absolute left-3 top-3 h-5 w-5 text-muted-foreground/50" />
                            </div>
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-medium text-muted-foreground ml-1">Password</label>
                            <div className="relative">
                                <Input
                                    type="password"
                                    placeholder="••••••••"
                                    className="pl-10 h-11 bg-white/60 border-gray-200 focus:bg-white transition-all"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    required
                                />
                                <Lock className="absolute left-3 top-3 h-5 w-5 text-muted-foreground/50" />
                            </div>
                        </div>

                        <Button
                            type="submit"
                            className="w-full h-11 text-base font-medium shadow-lg hover:shadow-xl transition-all duration-300 bg-gradient-to-r from-primary to-purple-600 hover:scale-[1.02]"
                            disabled={isLoading}
                        >
                            {isLoading ? (
                                <>
                                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                                    Verifying...
                                </>
                            ) : (
                                <>
                                    Access Dashboard
                                    <ArrowRight className="ml-2 h-5 w-5" />
                                </>
                            )}
                        </Button>

                        <div className="text-center">
                            <p className="text-xs text-muted-foreground mt-4">
                                Strictly confidential. Patient data handling is subject to DPDPA 2023 compliance.
                            </p>
                        </div>
                    </form>
                </CardContent>
            </Card>
        </div>
    );
}

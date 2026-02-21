import { useEffect, useState } from "react";
import { useUserSession } from "@/context/UserSessionContext";
import { loadAssessmentHistory, loadEncryptedAssessmentHistory } from "@/lib/history";
import { loadReports } from "@/lib/firebase";
import { AssessmentResult } from "@/services/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";

export default function PastAssessments() {
    const { auth } = useUserSession();
    const [history, setHistory] = useState<AssessmentResult[]>([]);
    const navigate = useNavigate();

    useEffect(() => {
        async function fetchHistory() {
            if (!auth) return;

            if (auth.provider === "firebase") {
                try {
                    const arr = await loadReports(auth.userId);
                    setHistory(arr);
                } catch (e) {
                    console.error("Failed to load firebase reports", e);
                }
            } else if (auth.password) {
                const arr = await loadEncryptedAssessmentHistory(auth.userId, auth.password);
                setHistory(arr);
            } else {
                setHistory(loadAssessmentHistory(auth.userId));
            }
        }
        fetchHistory();
    }, [auth]);

    return (
        <div className="container py-8 max-w-4xl mx-auto space-y-6">
            <Button variant="ghost" onClick={() => navigate(-1)} className="mb-2">
                <ArrowLeft className="w-4 h-4 mr-2" /> Back
            </Button>

            <div className="text-center space-y-2 mb-8">
                <h1 className="text-3xl font-bold">Past Assessments</h1>
                <p className="text-muted-foreground">Review your previously completed cognitive screening sessions.</p>
            </div>

            {!auth && (
                <Card className="border-yellow-500/60 bg-yellow-500/5">
                    <CardContent className="py-6 text-center">
                        <p className="text-yellow-700 font-medium">Please sign in to view your past assessments.</p>
                        <Button className="mt-4" onClick={() => navigate("/assessment")}>Go to Sign In</Button>
                    </CardContent>
                </Card>
            )}

            {auth && history.length === 0 && (
                <Card>
                    <CardContent className="py-12 text-center text-muted-foreground">
                        <p>You have no past assessments yet.</p>
                    </CardContent>
                </Card>
            )}

            {auth && history.length > 0 && (
                <Card className="shadow-card">
                    <CardHeader>
                        <CardTitle className="text-lg font-semibold text-muted-foreground">Assessment History</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-4">
                            {history.slice().reverse().map((h, i) => (
                                <div key={h.assessmentId || i} className="flex flex-wrap justify-between items-center p-4 rounded-xl border border-border bg-muted/20 gap-4">
                                    <div>
                                        <p className="font-medium text-lg text-foreground">
                                            {h.generatedAt ? new Date(h.generatedAt).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : "Unknown Date"}
                                        </p>
                                        <p className="text-sm text-muted-foreground mt-1">ID: {h.assessmentId || "Local-Session"}</p>
                                    </div>
                                    <div className="text-right flex flex-col items-end">
                                        <Badge
                                            variant="outline"
                                            className={
                                                h.riskLevel === "High" ? "border-red-500 text-red-500 bg-red-500/10 text-sm py-1" :
                                                    h.riskLevel === "Medium" ? "border-yellow-500 text-yellow-600 bg-yellow-500/10 text-sm py-1" :
                                                        "border-green-500 text-green-600 bg-green-500/10 text-sm py-1"
                                            }
                                        >
                                            {h.riskLevel} Risk
                                        </Badge>
                                        <p className="text-sm font-medium text-muted-foreground mt-2">Score: {((h.probability || 0) * 100).toFixed(1)}%</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>
            )}
        </div>
    );
}

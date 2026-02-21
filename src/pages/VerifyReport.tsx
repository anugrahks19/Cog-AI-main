import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { fetchAssessmentResult, AssessmentResult } from "@/services/api";
import { RiskResultCard } from "@/components/assessment/RiskResultCard";
import { Button } from "@/components/ui/button";
import { Loader2, ArrowLeft, AlertCircle } from "lucide-react";

export default function VerifyReport() {
    const { id } = useParams<{ id: string }>();
    const [result, setResult] = useState<AssessmentResult | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        async function loadResult() {
            if (!id) return;
            try {
                setIsLoading(true);
                setError(null);
                if (id.startsWith("offline-")) {
                    setError("This report was generated offline or locally and is not synced to the cloud. It can only be viewed on the original device.");
                    setIsLoading(false);
                    return;
                }

                // Note: For public verification, we ideally wouldn't need an access token,
                // or the backend would allow GET /api/assessments/{id}/result without auth
                const data = await fetchAssessmentResult(id);
                setResult(data);
            } catch (err: any) {
                console.error("Failed to load assessment report:", err);
                setError("Could not load the medical report. It may not exist, is private, or hasn't been synced.");
            } finally {
                setIsLoading(false);
            }
        }

        loadResult();
    }, [id]);

    return (
        <div className="container max-w-5xl py-12 animate-fade-in">
            <div className="mb-6">
                <Button variant="ghost" asChild className="mb-4">
                    <Link to="/">
                        <ArrowLeft className="mr-2 h-4 w-4" /> Back to Home
                    </Link>
                </Button>
                <h1 className="text-3xl font-bold tracking-tight">Report Verification</h1>
                <p className="text-muted-foreground mt-1">Viewing clinical report for Assessment ID: {id}</p>
            </div>

            {isLoading ? (
                <div className="flex flex-col items-center justify-center p-24 border rounded-xl bg-card">
                    <Loader2 className="h-10 w-10 animate-spin text-primary mb-4" />
                    <p className="text-muted-foreground font-medium">Fetching verified medical report...</p>
                </div>
            ) : error ? (
                <div className="flex flex-col items-center justify-center p-16 border rounded-xl bg-destructive/5 text-center">
                    <AlertCircle className="h-12 w-12 text-destructive mb-4" />
                    <h2 className="text-xl font-bold text-destructive mb-2">Verification Failed</h2>
                    <p className="text-muted-foreground max-w-md">{error}</p>
                </div>
            ) : result ? (
                <div className="space-y-8 animate-slide-up">
                    <RiskResultCard result={result} />
                </div>
            ) : null}
        </div>
    );
}

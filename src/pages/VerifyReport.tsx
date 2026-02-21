import { useEffect, useState } from "react";
import { useParams, useSearchParams, Link } from "react-router-dom";
import { fetchAssessmentResult, AssessmentResult } from "@/services/api";
import { RiskResultCard } from "@/components/assessment/RiskResultCard";
import { Button } from "@/components/ui/button";
import { Loader2, ArrowLeft, AlertCircle } from "lucide-react";
import { loadGlobalReports } from "@/lib/firebase";

export default function VerifyReport() {
    const { id } = useParams<{ id: string }>();
    const [searchParams] = useSearchParams();
    const [result, setResult] = useState<AssessmentResult | null>(null);
    const [patientName, setPatientName] = useState<string | undefined>(undefined);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        async function loadResult() {
            if (!id) return;
            try {
                setIsLoading(true);
                setError(null);

                // Option 1: Direct payload decoding from QR code
                const dataParam = searchParams.get("data");
                if (dataParam) {
                    try {
                        const parsed = JSON.parse(atob(dataParam));
                        setPatientName(parsed.n);
                        setResult({
                            assessmentId: parsed.id,
                            riskLevel: parsed.r,
                            probability: parsed.p,
                            subScores: parsed.s,
                            generatedAt: parsed.d,
                            featureImportances: [], // Omitted from QR to save space
                            recommendations: []     // Omitted from QR to save space
                        } as AssessmentResult);
                        setIsLoading(false);
                        return;
                    } catch (e) {
                        console.error("Failed to decode QR payload", e);
                        // Fallback to fetch if decode fails
                    }
                }

                // Option 2: Fallback to global database lookup
                try {
                    const reports = await loadGlobalReports();
                    const found = reports.find(r => r.assessmentId === id);
                    if (found) {
                        const anyFound = found as any;
                        setPatientName(anyFound.patientSummary?.name);
                        setResult(found);
                        return;
                    }
                } catch (e) {
                    console.error("Firebase lookup failed", e);
                }

                setError("Could not load the medical report. Scan a valid QR code or ensure the report was permanently saved.");
            } catch (err: any) {
                console.error("Failed to load assessment report:", err);
                setError("Could not load the medical report. It may not exist, is private, or hasn't been synced.");
            } finally {
                setIsLoading(false);
            }
        }

        loadResult();
    }, [id, searchParams]);

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
                    <RiskResultCard result={result} patientName={patientName} />
                </div>
            ) : null}
        </div>
    );
}

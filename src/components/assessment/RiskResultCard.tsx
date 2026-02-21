import { useMemo } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AssessmentResult } from "@/services/api";
import { cn } from "@/lib/utils";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Download, FileText, Activity, Share2 } from "lucide-react";
import PatientTrendChart from "@/components/dashboard/PatientTrendChart";
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from "recharts";
import QRCode from "react-qr-code";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";

interface RiskResultCardProps {
  result: AssessmentResult;
  languageLabel?: string;
  history?: AssessmentResult[];
  patientName?: string;
}

const RISK_COLOR_MAP: Record<AssessmentResult["riskLevel"], string> = {
  Low: "bg-green-100 text-green-800 border-green-200",
  Medium: "bg-yellow-100 text-yellow-800 border-yellow-200",
  High: "bg-red-100 text-red-800 border-red-200",
};

export const RiskResultCard = ({ result, languageLabel, history = [], patientName }: RiskResultCardProps) => {
  const formatFeatureName = (key: string) =>
    key
      .replace(/[_-]+/g, " ")
      .trim()
      .replace(/\s+/g, " ")
      .split(" ")
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ");

  const riskClassName = RISK_COLOR_MAP[result.riskLevel];
  const confidence = Math.round(result.probability * 100);

  const donutColor = useMemo(() => {
    if (result.riskLevel === "High") return "#dc2626"; // red-600
    if (result.riskLevel === "Medium") return "#ca8a04"; // yellow-600
    return "#16a34a"; // green-600
  }, [result.riskLevel]);

  // Session-level heuristics for max attainable scores (prevents division by zero)
  const domainMax = {
    memoryScore: Math.max(1, 2),
    attentionScore: Math.max(1, 4),
    languageScore: Math.max(1, 1),
    executiveScore: Math.max(1, 1),
  } as const;

  const dementiaBenchmark = {
    memoryScore: 0.6,
    attentionScore: 1.2,
    languageScore: 0.4,
    executiveScore: 0.3,
  } as const;

  const normalized = {
    memory: Math.min(100, (result.subScores.memoryScore / domainMax.memoryScore) * 100),
    attention: Math.min(100, (result.subScores.attentionScore / domainMax.attentionScore) * 100),
    language: Math.min(100, (result.subScores.languageScore / domainMax.languageScore) * 100),
    executive: Math.min(100, (result.subScores.executiveScore / domainMax.executiveScore) * 100),
  };

  const dementiaPct = {
    memory: Math.min(100, (dementiaBenchmark.memoryScore / domainMax.memoryScore) * 100),
    attention: Math.min(100, (dementiaBenchmark.attentionScore / domainMax.attentionScore) * 100),
    language: Math.min(100, (dementiaBenchmark.languageScore / domainMax.languageScore) * 100),
    executive: Math.min(100, (dementiaBenchmark.executiveScore / domainMax.executiveScore) * 100),
  };

  const radarData = useMemo(() => [
    { subject: "Memory", score: Math.round(normalized.memory), fullMark: 100 },
    { subject: "Attention", score: Math.round(normalized.attention), fullMark: 100 },
    { subject: "Language", score: Math.round(normalized.language), fullMark: 100 },
    { subject: "Executive", score: Math.round(normalized.executive), fullMark: 100 },
  ], [normalized]);

  const handleDownloadPDF = async () => {
    const element = document.getElementById("risk-report-container");
    if (!element) return;

    try {
      const actionButtons = document.getElementById("report-action-buttons");
      if (actionButtons) actionButtons.style.display = "none";

      const originalStyle = element.style.cssText;
      element.style.width = '1000px';
      element.style.height = 'auto';
      element.style.padding = '20px';

      const canvas = await html2canvas(element, {
        scale: 2,
        useCORS: true,
        logging: false,
        backgroundColor: '#ffffff',
        windowWidth: 1000
      });

      element.style.cssText = originalStyle;
      if (actionButtons) actionButtons.style.display = "flex";

      const imgData = canvas.toDataURL("image/png");
      const pdf = new jsPDF({
        orientation: "portrait",
        unit: "mm",
        format: "a4"
      });

      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

      pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
      pdf.save(`Medical_Report_${patientName ? patientName.replace(/\s+/g, '_') : 'Patient'}.pdf`);
    } catch (error) {
      console.error(error);
      const actionButtons = document.getElementById("report-action-buttons");
      if (actionButtons) actionButtons.style.display = "flex";
    }
  };

  return (
    <Card id="risk-report-container" className="shadow-card overflow-hidden">
      <CardHeader className="space-y-2 border-b border-border/50 bg-muted/10 pb-6">
        <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-4">
          <div>
            <CardTitle className="flex items-center gap-2 text-2xl">
              <FileText className="h-6 w-6 text-primary" />
              Comprehensive Clinical Report
            </CardTitle>
            <CardDescription className="mt-1">
              Generated on {new Date().toLocaleString()} {patientName ? `for ${patientName}` : ""}
            </CardDescription>
          </div>
          <div id="report-action-buttons" className="flex items-center gap-2">
            {languageLabel ? <Badge variant="secondary">{languageLabel}</Badge> : null}
            <Button onClick={handleDownloadPDF} size="sm" className="bg-primary shadow-sm hover:bg-primary/90">
              <Download className="mr-2 h-4 w-4" /> Download PDF
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-8 pt-6">
        {/* Top Section: Risk probability & Gauge */}
        <div className="grid gap-6 md:grid-cols-2">
          {/* Gauge Section */}
          <div className="flex flex-col items-center justify-center p-6 border rounded-xl bg-slate-50 dark:bg-slate-900/50">
            <div
              aria-label="Risk probability"
              className="relative h-40 w-40 rounded-full transition-all duration-1000 ease-out"
              style={{ background: `conic-gradient(${donutColor} ${confidence * 3.6}deg, #e5e7eb 0deg)` }}
            >
              <div className="absolute inset-3 rounded-full bg-slate-50 dark:bg-slate-900/50 flex items-center justify-center shadow-inner">
                <div className="text-center">
                  <div className="text-3xl font-bold" style={{ color: donutColor }}>{confidence}%</div>
                  <div className="text-xs font-medium text-muted-foreground uppercase tracking-widest mt-1">Risk</div>
                </div>
              </div>
            </div>
          </div>
          {/* Summary / Info Section */}
          <div className="flex flex-col justify-center gap-4">
            <div className={cn("rounded-xl border px-5 py-4 shadow-sm", riskClassName)}>
              <div className="flex items-center gap-2 mb-1">
                <Activity className="h-5 w-5" />
                <h3 className="text-xl font-bold tracking-tight">Level: {result.riskLevel}</h3>
              </div>
              <p className="text-sm opacity-90 font-medium">Model Confidence: {confidence}%</p>
              <p className="mt-3 text-sm opacity-80 leading-relaxed">
                This screening utilizes multi-modal biomarker analysis but is not a clinical diagnosis. Always consult a neurologist for a comprehensive evaluation.
              </p>
            </div>
          </div>
        </div>

        {/* Middle Section: Profiles & Radar */}
        <div className="grid gap-6 lg:grid-cols-2 lg:items-start border-t pt-8">
          <div className="space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-widest text-muted-foreground border-b pb-2">
              Cognitive Profile (Radar)
            </h3>
            <div className="h-64 w-full flex items-center justify-center bg-slate-50 dark:bg-slate-900/30 rounded-xl border">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                  <PolarGrid strokeOpacity={0.4} />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: 'currentColor', fontSize: 12, fontWeight: 500 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                  <Radar name="Patient" dataKey="score" stroke={donutColor} fill={donutColor} fillOpacity={0.6} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-widest text-muted-foreground border-b pb-2">
              Key Biomarker Contributions
            </h3>
            <div className="space-y-4 bg-white dark:bg-card rounded-xl border p-4">
              {result.featureImportances.map((feature) => (
                <div key={feature.feature} className="space-y-1.5">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-semibold text-foreground">{formatFeatureName(feature.feature)}</span>
                    <span
                      className={cn(
                        "text-xs font-bold px-2 py-0.5 rounded-full bg-muted",
                        feature.direction === "positive" ? "text-red-600" : "text-green-600",
                      )}
                    >
                      {feature.direction === "positive" ? "↑" : "↓"} {Math.round(feature.contribution * 100)}%
                    </span>
                  </div>
                  <Progress value={Math.abs(feature.contribution) * 100} className="h-2" />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Comparative sub-scores */}
        <div className="space-y-4 border-t pt-8">
          <h3 className="text-sm font-bold uppercase tracking-widest text-muted-foreground border-b pb-2">Domain Analytics</h3>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-xl border p-4 bg-slate-50 dark:bg-slate-900/30 hover:shadow-sm transition-shadow">
              <p className="text-sm font-medium text-muted-foreground">Memory</p>
              <div className="space-y-2 mt-1">
                <div className="flex items-center justify-between text-xs"><span>Normal</span><span>100%</span></div>
                <Progress value={100} className="h-1.5" indicatorClassName="bg-primary" />
                <div className="flex items-center justify-between text-xs"><span>Patient</span><span>{Math.round(normalized.memory)}%</span></div>
                <Progress value={normalized.memory} className="h-1.5" indicatorClassName="bg-secondary" />
                <div className="flex items-center justify-between text-xs"><span>Dementia</span><span>{Math.round(dementiaPct.memory)}%</span></div>
                <Progress value={dementiaPct.memory} className="h-1.5" indicatorClassName="bg-destructive/70" />
              </div>
            </div>
            <div className="rounded-lg border p-3">
              <p className="text-sm font-medium text-muted-foreground">Attention</p>
              <div className="space-y-2 mt-1">
                <div className="flex items-center justify-between text-xs"><span>Normal</span><span>100%</span></div>
                <Progress value={100} className="h-1.5" indicatorClassName="bg-primary" />
                <div className="flex items-center justify-between text-xs"><span>Patient</span><span>{Math.round(normalized.attention)}%</span></div>
                <Progress value={normalized.attention} className="h-1.5" indicatorClassName="bg-secondary" />
                <div className="flex items-center justify-between text-xs"><span>Dementia</span><span>{Math.round(dementiaPct.attention)}%</span></div>
                <Progress value={dementiaPct.attention} className="h-1.5" indicatorClassName="bg-destructive/70" />
              </div>
            </div>
            <div className="rounded-lg border p-3">
              <p className="text-sm font-medium text-muted-foreground">Language</p>
              <div className="space-y-2 mt-1">
                <div className="flex items-center justify-between text-xs"><span>Normal</span><span>100%</span></div>
                <Progress value={100} className="h-1.5" indicatorClassName="bg-primary" />
                <div className="flex items-center justify-between text-xs"><span>Patient</span><span>{Math.round(normalized.language)}%</span></div>
                <Progress value={normalized.language} className="h-1.5" indicatorClassName="bg-secondary" />
                <div className="flex items-center justify-between text-xs"><span>Dementia</span><span>{Math.round(dementiaPct.language)}%</span></div>
                <Progress value={dementiaPct.language} className="h-1.5" indicatorClassName="bg-destructive/70" />
              </div>
            </div>
            <div className="rounded-xl border p-4 bg-slate-50 dark:bg-slate-900/30 hover:shadow-sm transition-shadow">
              <p className="font-bold text-muted-foreground mb-3 text-center">Executive</p>
              <div className="space-y-3">
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs font-semibold"><span className="text-primary">Normal</span><span>100%</span></div>
                  <Progress value={100} className="h-1.5" indicatorClassName="bg-primary" />
                </div>
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs font-semibold"><span className="text-secondary-foreground">Patient</span><span>{Math.round(normalized.executive)}%</span></div>
                  <Progress value={normalized.executive} className="h-2" indicatorClassName="bg-secondary" />
                </div>
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs font-semibold"><span className="text-destructive">Dementia</span><span>{Math.round(dementiaPct.executive)}%</span></div>
                  <Progress value={dementiaPct.executive} className="h-1.5" indicatorClassName="bg-destructive/70" />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Longitudinal History */}
        {history && history.length > 1 && (
          <div className="space-y-4 border-t pt-8">
            <h3 className="text-sm font-bold uppercase tracking-widest text-muted-foreground border-b pb-2">Longitudinal Trajectory</h3>
            <div className="bg-white dark:bg-card border rounded-xl shadow-sm p-4 w-full h-[300px]">
              <PatientTrendChart history={history} />
            </div>
          </div>
        )}

        {/* Recommendations & QR Code */}
        <div className="grid md:grid-cols-3 gap-6 border-t pt-8">
          <div className="md:col-span-2 space-y-3 p-5 rounded-xl border bg-muted/20">
            <h3 className="text-sm font-bold uppercase tracking-widest text-muted-foreground">
              Clinical Recommendations
            </h3>
            <ul className="list-disc space-y-2 pl-5 text-sm font-medium text-foreground">
              {result.recommendations.map((recommendation) => (
                <li key={recommendation} className="leading-relaxed">{recommendation}</li>
              ))}
            </ul>
          </div>

          <div className="flex flex-col items-center justify-center p-6 border rounded-xl bg-white dark:bg-card shadow-sm">
            <QRCode
              value={`${window.location.origin}/verify/${result.assessmentId}`}
              size={90}
              level="H"
              className="mb-3"
            />
            <div className="text-center space-y-1">
              <p className="text-xs font-bold uppercase tracking-wider text-muted-foreground flex items-center justify-center gap-1">
                <Share2 className="h-3 w-3" /> Verify Record
              </p>
              <p className="text-[10px] text-muted-foreground break-all">ID: {result.assessmentId.substring(0, 8).toUpperCase()}</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

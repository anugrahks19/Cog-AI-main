import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Search, TrendingUp, AlertCircle, FileJson, Users, LogOut, LayoutDashboard, Trash2 } from "lucide-react";
import PatientTrendChart from "@/components/dashboard/PatientTrendChart";
import { downloadFHIR } from "@/lib/fhir";
import { AssessmentResult } from "@/services/api";
import ClinicianLogin from "@/pages/ClinicianLogin";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";
import { loadGlobalReports, deleteGlobalReport } from "@/lib/firebase";
import { RiskResultCard } from "@/components/assessment/RiskResultCard";
import { useToast } from "@/hooks/use-toast";

const MOCK_PATIENTS = [
    { id: "P-101", name: "Ramesh Gupta", age: 68, risk: "High", lastVisit: "2026-12-18", trend: "Declining" },
    { id: "P-102", name: "Sarah Thomas", age: 72, risk: "Low", lastVisit: "2026-12-15", trend: "Stable" },
    { id: "P-103", name: "Lakshmi N", age: 65, risk: "Medium", lastVisit: "2026-12-10", trend: "Improving" },
    { id: "P-104", name: "John Doe", age: 70, risk: "Low", lastVisit: "2026-12-05", trend: "Stable" },
];

const MOCK_HISTORY: AssessmentResult[] = [
    {
        assessmentId: "a-1",
        riskLevel: "Low",
        probability: 0.15,
        generatedAt: "2026-09-01T10:00:00Z",
        recommendations: [],
        featureImportances: [],
        subScores: { memoryScore: 0.8, attentionScore: 0.9, languageScore: 0.85, executiveScore: 0.9 },
    },
    {
        assessmentId: "a-2",
        riskLevel: "Low",
        probability: 0.22,
        generatedAt: "2026-10-01T10:00:00Z",
        recommendations: [],
        featureImportances: [],
        subScores: { memoryScore: 0.75, attentionScore: 0.85, languageScore: 0.8, executiveScore: 0.88 },
    },
    {
        assessmentId: "a-3",
        riskLevel: "Medium",
        probability: 0.45,
        generatedAt: "2026-11-01T10:00:00Z",
        recommendations: [],
        featureImportances: [],
        subScores: { memoryScore: 0.6, attentionScore: 0.7, languageScore: 0.75, executiveScore: 0.7 },
    },
    {
        assessmentId: "a-4",
        riskLevel: "High",
        probability: 0.72,
        generatedAt: "2026-12-18T10:00:00Z",
        recommendations: ["Urgent Neurologist Consult"],
        featureImportances: [],
        subScores: { memoryScore: 0.4, attentionScore: 0.5, languageScore: 0.6, executiveScore: 0.5 },
    },
];

export default function ClinicianDashboard() {
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [selectedPatient, setSelectedPatient] = useState<string | null>(null);
    const [searchTerm, setSearchTerm] = useState("");
    const [globalReports, setGlobalReports] = useState<AssessmentResult[]>([]);
    const { toast } = useToast();
    const [hiddenMocks, setHiddenMocks] = useState<string[]>([]); // To allow "deleting" mocks from view

    useEffect(() => {
        if (isAuthenticated) {
            loadGlobalReports().then(setGlobalReports).catch(console.error);
        }
    }, [isAuthenticated]);

    if (!isAuthenticated) {
        return <ClinicianLogin onLogin={() => setIsAuthenticated(true)} />;
    }

    const uniquePatientsMap = new Map();
    globalReports.forEach(r => {
        const anyR = r as any;
        const name = anyR.patientSummary?.name || `Patient (${r.assessmentId.substring(0, 5)})`;
        const age = anyR.patientSummary?.age || "N/A";

        if (!uniquePatientsMap.has(name)) {
            uniquePatientsMap.set(name, {
                id: r.assessmentId,
                name,
                age,
                risk: r.riskLevel,
                lastVisit: new Date(r.generatedAt).toISOString().split('T')[0],
                trend: "New Data",
                history: [r]
            });
        } else {
            uniquePatientsMap.get(name).history.push(r);
            uniquePatientsMap.get(name).history.sort((a: any, b: any) => new Date(a.generatedAt).getTime() - new Date(b.generatedAt).getTime());

            const currentLastVisit = new Date(uniquePatientsMap.get(name).lastVisit).getTime();
            const thisVisit = new Date(r.generatedAt).getTime();
            if (thisVisit > currentLastVisit) {
                uniquePatientsMap.get(name).lastVisit = new Date(thisVisit).toISOString().split('T')[0];
                uniquePatientsMap.get(name).risk = r.riskLevel;
            }
        }
    });

    const realPatients = Array.from(uniquePatientsMap.values());
    const mockPatientsWithHistory = MOCK_PATIENTS.map(p => ({ ...p, history: MOCK_HISTORY }));
    const allPatients = [...realPatients, ...mockPatientsWithHistory].filter(p => !hiddenMocks.includes(p.id));

    const filteredPatients = allPatients.filter((p) =>
        p.name.toLowerCase().includes(searchTerm.toLowerCase()) || p.id.toLowerCase().includes(searchTerm.toLowerCase())
    );

    const patientDetails = allPatients.find((p) => p.id === selectedPatient);
    // Use the latest result in their history for FHIR export
    const latestResult = patientDetails ? patientDetails.history[patientDetails.history.length - 1] : MOCK_HISTORY[MOCK_HISTORY.length - 1];

    const handleDelete = async (patientId: string, e: React.MouseEvent) => {
        e.stopPropagation(); // prevent row click from selecting the patient
        if (window.confirm("Are you sure you want to delete this assessment record?")) {
            if (patientId.startsWith("P-")) {
                // It's a mock patient, simulate deletion
                setHiddenMocks(prev => [...prev, patientId]);
                toast({ title: "Deleted", description: "Mock record removed from view." });
            } else {
                // It's a real global report ID
                const success = await deleteGlobalReport(patientId);
                if (success) {
                    setGlobalReports(prev => prev.filter(r => r.assessmentId !== patientId));
                    toast({ title: "Deleted", description: "Assessment record permanently deleted." });
                } else {
                    toast({ title: "Error", description: "Network error occurred." });
                }
            }
            if (selectedPatient === patientId) {
                setSelectedPatient(null);
            }
        }
    };

    const handleExportFHIR = () => {
        if (!selectedPatient) return;
        downloadFHIR(latestResult, `${selectedPatient}-medical-record.json`);
    };

    const handleDownloadPDF = async () => {
        if (!selectedPatient || !patientDetails) return;

        const element = document.getElementById("patient-detail-view");
        if (!element) return;

        try {
            // Apply temporary styles for better PDF rendering
            const originalStyle = element.style.cssText;
            element.style.width = '1200px';
            element.style.height = 'auto';
            element.style.padding = '30px';
            element.style.backgroundColor = 'white';

            // Hide action buttons during export
            const actionButtons = document.getElementById("pdf-action-buttons");
            if (actionButtons) actionButtons.style.display = "none";

            const canvas = await html2canvas(element, {
                scale: 2,
                useCORS: true,
                logging: false,
                windowWidth: 1200
            });

            // Restore original styles
            element.style.cssText = originalStyle;
            if (actionButtons) actionButtons.style.display = "flex";

            const imgData = canvas.toDataURL("image/png");

            // A4 Landscape dimensions in mm (297 x 210)
            const pdf = new jsPDF({
                orientation: "landscape",
                unit: "mm",
                format: "a4"
            });

            // Calculate dimensions to fit the page while maintaining aspect ratio
            const pdfWidth = pdf.internal.pageSize.getWidth();
            const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

            pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
            pdf.save(`${patientDetails.name.replace(/\s+/g, '_')}_Scan_Report.pdf`);
        } catch (error) {
            console.error("Failed to generate PDF:", error);
            const actionButtons = document.getElementById("pdf-action-buttons");
            if (actionButtons) actionButtons.style.display = "flex";
        }
    };

    return (
        <div className="w-full">
            {/* Mobile Warning */}
            <div className="flex md:hidden min-h-[50vh] items-center justify-center p-6 text-center mt-10">
                <div className="bg-white dark:bg-card p-8 rounded-2xl border border-border/50 shadow-soft max-w-md space-y-4">
                    <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-2">
                        <LayoutDashboard className="h-8 w-8 text-primary" />
                    </div>
                    <h2 className="text-xl font-bold text-foreground">Desktop Support Only</h2>
                    <p className="text-muted-foreground text-sm leading-relaxed">
                        The Clinician Dashboard handles complex medical data and patient monitoring metrics that require a larger screen. Please access this portal from a desktop or laptop computer.
                    </p>
                    <Button variant="outline" className="mt-4 w-full" onClick={() => window.history.back()}>
                        Go Back
                    </Button>
                </div>
            </div>

            {/* Desktop Dashboard */}
            <div className="hidden md:block min-h-screen">
                {/* Top Bar */}
                <div className="bg-white dark:bg-card border-b border-border sticky top-16 z-30 px-6 py-4 flex justify-between items-center shadow-sm">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-primary/10 rounded-lg">
                            <LayoutDashboard className="h-6 w-6 text-primary" />
                        </div>
                        <div>
                            <h1 className="text-xl font-bold text-foreground">Clinician Portal</h1>
                            <p className="text-xs text-muted-foreground">Welcome, Dr. S. Iyer (Neurology)</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-4">
                        <Button variant="ghost" size="sm" onClick={() => setIsAuthenticated(false)} className="text-muted-foreground hover:text-destructive">
                            <LogOut className="h-4 w-4 mr-2" />
                            Sign Out
                        </Button>
                    </div>
                </div>

                <div className="p-6 max-w-7xl mx-auto space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        {/* Patient List */}
                        <Card className="md:col-span-1 h-[700px] flex flex-col border-border/60 shadow-lg">
                            <CardHeader className="bg-muted/30 border-b border-border/50 pb-4">
                                <div className="flex items-center justify-between mb-2">
                                    <CardTitle className="flex items-center gap-2">
                                        <Users className="h-5 w-5 text-primary" />
                                        My Patients
                                    </CardTitle>
                                    <Badge variant="secondary" className="px-2 py-0.5 text-xs font-normal">
                                        Live
                                    </Badge>
                                </div>
                                <div className="relative">
                                    <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                                    <Input
                                        placeholder="Search MRN or Name..."
                                        className="pl-9 bg-white dark:bg-card"
                                        value={searchTerm}
                                        onChange={(e) => setSearchTerm(e.target.value)}
                                    />
                                </div>
                            </CardHeader>
                            <CardContent className="flex-1 overflow-auto p-0">
                                <Table>
                                    <TableBody>
                                        {filteredPatients.map((patient) => (
                                            <TableRow
                                                key={patient.id}
                                                className={`cursor-pointer transition-colors ${selectedPatient === patient.id ? "bg-primary/5 border-l-4 border-l-primary" : "hover:bg-muted/50 border-l-4 border-l-transparent"}`}
                                                onClick={() => setSelectedPatient(patient.id)}
                                            >
                                                <TableCell className="py-4">
                                                    <div className="flex justify-between items-start">
                                                        <div>
                                                            <div className="font-semibold text-foreground">{patient.name}</div>
                                                            <div className="text-xs text-muted-foreground mt-0.5">ID: {patient.id} • {patient.age} yrs</div>
                                                        </div>
                                                        <div className="flex flex-col items-end gap-2">
                                                            <Badge
                                                                variant={patient.risk === "High" ? "destructive" : patient.risk === "Medium" ? "secondary" : "outline"}
                                                                className="shadow-sm"
                                                            >
                                                                {patient.risk} Risk
                                                            </Badge>
                                                            <Button
                                                                variant="ghost"
                                                                size="icon"
                                                                className="h-6 w-6 text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                                                                onClick={(e) => handleDelete(patient.id, e)}
                                                                title="Delete this record"
                                                            >
                                                                <Trash2 className="h-4 w-4" />
                                                            </Button>
                                                        </div>
                                                    </div>
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </CardContent>
                        </Card>

                        {/* Patient Detail View */}
                        <Card className="md:col-span-2 h-[700px] flex flex-col border-border/60 shadow-lg overflow-hidden">
                            {selectedPatient && patientDetails ? (
                                <div id="patient-detail-view" className="flex flex-col h-full bg-white dark:bg-card">
                                    <div className="bg-muted/30 p-6 border-b border-border/50 flex flex-wrap justify-between items-start gap-4">
                                        <div>
                                            <h2 className="text-3xl font-bold text-foreground mb-3">{patientDetails.name}</h2>
                                            <div className="flex items-center space-x-4 text-sm text-muted-foreground mt-2">
                                                <span className="flex items-center bg-white dark:bg-card px-2 py-1 rounded-md border shadow-sm">
                                                    <Users className="h-3 w-3 mr-1.5" /> {patientDetails.age} Years / Male
                                                </span>
                                                <span className="flex items-center bg-white dark:bg-card px-2 py-1 rounded-md border shadow-sm">
                                                    Last Visit: {patientDetails.lastVisit}
                                                </span>
                                            </div>
                                        </div>
                                        <div id="pdf-action-buttons" className="flex gap-2">
                                            <Button variant="outline" onClick={handleExportFHIR} className="bg-white dark:bg-card hover:bg-muted shadow-sm border-primary/20 hover:border-primary/50 text-primary">
                                                <FileJson className="mr-2 h-4 w-4" />
                                                Export FHIR JSON
                                            </Button>
                                            <Button onClick={handleDownloadPDF} className="bg-primary hover:bg-primary/90 shadow-md">
                                                Download PDF
                                            </Button>
                                        </div>
                                    </div>

                                    <CardContent className="space-y-6 overflow-auto p-0 bg-transparent flex-1 pt-6">
                                        <RiskResultCard
                                            result={patientDetails.history[patientDetails.history.length - 1]}
                                            history={patientDetails.history}
                                            patientName={patientDetails.name}
                                        />
                                    </CardContent>
                                </div>
                            ) : (
                                <div className="flex flex-col items-center justify-center h-full text-muted-foreground bg-muted/10">
                                    <div className="h-20 w-20 bg-muted/20 rounded-full flex items-center justify-center mb-6">
                                        <Search className="h-10 w-10 opacity-40" />
                                    </div>
                                    <h3 className="text-xl font-semibold mb-2">No Patient Selected</h3>
                                    <p>Select a patient from the list to assign tasks or view analytics.</p>
                                </div>
                            )}
                        </Card>
                    </div>
                </div>
            </div>
        </div>
    );
}

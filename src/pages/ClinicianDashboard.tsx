import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Search, TrendingUp, AlertCircle, FileJson, Users, LogOut, LayoutDashboard } from "lucide-react";
import PatientTrendChart from "@/components/dashboard/PatientTrendChart";
import { downloadFHIR } from "@/lib/fhir";
import { AssessmentResult } from "@/services/api";
import ClinicianLogin from "@/pages/ClinicianLogin";

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

    if (!isAuthenticated) {
        return <ClinicianLogin onLogin={() => setIsAuthenticated(true)} />;
    }

    const filteredPatients = MOCK_PATIENTS.filter((p) =>
        p.name.toLowerCase().includes(searchTerm.toLowerCase()) || p.id.toLowerCase().includes(searchTerm.toLowerCase())
    );

    const patientDetails = MOCK_PATIENTS.find((p) => p.id === selectedPatient);
    const latestResult = MOCK_HISTORY[MOCK_HISTORY.length - 1];

    const handleExportFHIR = () => {
        if (!selectedPatient) return;
        downloadFHIR(latestResult, `${selectedPatient}-medical-record.json`);
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
                                                        <Badge
                                                            variant={patient.risk === "High" ? "destructive" : patient.risk === "Medium" ? "secondary" : "outline"}
                                                            className="shadow-sm"
                                                        >
                                                            {patient.risk} Risk
                                                        </Badge>
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
                                <>
                                    <div className="bg-muted/30 p-6 border-b border-border/50 flex flex-wrap justify-between items-start gap-4">
                                        <div>
                                            <h2 className="text-3xl font-bold text-foreground mb-1">{patientDetails.name}</h2>
                                            <div className="flex items-center gap-4 text-sm text-muted-foreground">
                                                <span className="flex items-center gap-1.5 bg-white dark:bg-card px-2 py-1 rounded-md border shadow-sm">
                                                    <Users className="h-3 w-3" /> {patientDetails.age} Years / Male
                                                </span>
                                                <span className="flex items-center gap-1.5 bg-white dark:bg-card px-2 py-1 rounded-md border shadow-sm">
                                                    Last Visit: {patientDetails.lastVisit}
                                                </span>
                                            </div>
                                        </div>
                                        <div className="flex gap-2">
                                            <Button variant="outline" onClick={handleExportFHIR} className="bg-white dark:bg-card hover:bg-muted shadow-sm border-primary/20 hover:border-primary/50 text-primary">
                                                <FileJson className="mr-2 h-4 w-4" />
                                                Export FHIR JSON
                                            </Button>
                                            <Button className="bg-primary hover:bg-primary/90 shadow-md">
                                                Review Scan
                                            </Button>
                                        </div>
                                    </div>

                                    <CardContent className="space-y-6 overflow-auto p-6 bg-gradient-to-b from-white to-muted/20 flex-1">
                                        {/* Alert for High Risk */}
                                        {patientDetails.risk === "High" && (
                                            <div className="bg-red-50/80 border border-red-200 rounded-xl p-4 flex items-start gap-4 shadow-sm animate-in fade-in slide-in-from-top-2">
                                                <div className="p-2 bg-red-100 rounded-full">
                                                    <AlertCircle className="h-6 w-6 text-red-600" />
                                                </div>
                                                <div>
                                                    <h4 className="font-bold text-red-900 text-lg">Urgent Intervention Recommended</h4>
                                                    <p className="text-red-700 mt-1">
                                                        Patient's cognitive scores have dropped significantly (Probability: 72%).
                                                        Recent speech analysis indicates <span className="font-semibold">lexical retrieval deficits</span>.
                                                    </p>
                                                </div>
                                            </div>
                                        )}

                                        {/* The Chart */}
                                        <div className="bg-white dark:bg-card rounded-xl border border-border/50 p-4 shadow-sm">
                                            <div className="flex items-center justify-between mb-6">
                                                <h3 className="font-semibold flex items-center gap-2">
                                                    <TrendingUp className="h-5 w-5 text-primary" />
                                                    Longitudinal Cognitive Track (6 Months)
                                                </h3>
                                                <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                                                    AI Analysis: {patientDetails.trend}
                                                </Badge>
                                            </div>
                                            <div className="h-[300px] w-full">
                                                <PatientTrendChart key={selectedPatient} history={MOCK_HISTORY} />
                                            </div>
                                        </div>

                                        {/* Metrics Grid */}
                                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                                            <div className="p-4 bg-white dark:bg-card border border-border/50 rounded-xl shadow-sm text-center hover:shadow-md transition-shadow">
                                                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Memory</div>
                                                <div className="text-3xl font-bold text-primary">4<span className="text-lg text-muted-foreground/50">/10</span></div>
                                                <div className="h-1.5 w-full bg-muted mt-3 rounded-full overflow-hidden">
                                                    <div className="h-full bg-red-500 w-[40%]"></div>
                                                </div>
                                            </div>
                                            <div className="p-4 bg-white dark:bg-card border border-border/50 rounded-xl shadow-sm text-center hover:shadow-md transition-shadow">
                                                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Attention</div>
                                                <div className="text-3xl font-bold text-yellow-600">5<span className="text-lg text-muted-foreground/50">/10</span></div>
                                                <div className="h-1.5 w-full bg-muted mt-3 rounded-full overflow-hidden">
                                                    <div className="h-full bg-yellow-500 w-[50%]"></div>
                                                </div>
                                            </div>
                                            <div className="p-4 bg-white dark:bg-card border border-border/50 rounded-xl shadow-sm text-center hover:shadow-md transition-shadow">
                                                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Language</div>
                                                <div className="text-3xl font-bold text-green-600">6<span className="text-lg text-muted-foreground/50">/10</span></div>
                                                <div className="h-1.5 w-full bg-muted mt-3 rounded-full overflow-hidden">
                                                    <div className="h-full bg-green-500 w-[60%]"></div>
                                                </div>
                                            </div>
                                            <div className="p-4 bg-white dark:bg-card border border-border/50 rounded-xl shadow-sm text-center hover:shadow-md transition-shadow">
                                                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Executive</div>
                                                <div className="text-3xl font-bold text-blue-600">5<span className="text-lg text-muted-foreground/50">/10</span></div>
                                                <div className="h-1.5 w-full bg-muted mt-3 rounded-full overflow-hidden">
                                                    <div className="h-full bg-blue-500 w-[50%]"></div>
                                                </div>
                                            </div>
                                        </div>
                                    </CardContent>
                                </>
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

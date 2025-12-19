import { AssessmentResult } from "@/services/api";

/**
 * Converts an internal AssessmentResult into a bundle of HL7 FHIR resources.
 * This simulates a professional EMR (Electronic Medical Record) export.
 */
export function convertToFHIR(result: AssessmentResult, patientId: string = "patient-123"): Record<string, any> {
    const timestamp = new Date(result.generatedAt).toISOString();

    // 1. Patient Resource (Stub)
    const patientResource = {
        resourceType: "Patient",
        id: patientId,
        active: true,
    };

    // 2. Observation Resource (The Cognitive Score)
    const observationResource = {
        resourceType: "Observation",
        id: `obs-${result.assessmentId}`,
        status: "final",
        code: {
            coding: [
                {
                    system: "http://loinc.org",
                    code: "72172-8",
                    display: "Cognitive assessment score",
                },
            ],
            text: "Cog-AI Dementia Risk Assessment",
        },
        subject: {
            reference: `Patient/${patientId}`,
        },
        effectiveDateTime: timestamp,
        valueString: result.riskLevel, // High/Medium/Low
        component: [
            {
                code: { text: "Probability" },
                valueQuantity: {
                    value: Math.round(result.probability * 100),
                    unit: "%",
                    system: "http://unitsofmeasure.org",
                    code: "%",
                },
            },
            {
                code: { text: "Memory Score" },
                valueQuantity: { value: result.subScores.memoryScore },
            },
            {
                code: { text: "Attention Score" },
                valueQuantity: { value: result.subScores.attentionScore },
            },
            {
                code: { text: "Language Score" },
                valueQuantity: { value: result.subScores.languageScore },
            },
            {
                code: { text: "Executive Score" },
                valueQuantity: { value: result.subScores.executiveScore },
            },
        ],
    };

    // 3. DiagnosticReport Resource (The Wrapper)
    const diagnosticReport = {
        resourceType: "DiagnosticReport",
        id: `rpt-${result.assessmentId}`,
        status: "final",
        code: {
            coding: [
                {
                    system: "http://loinc.org",
                    code: "55751-2",
                    display: "Public health risk assessment",
                },
            ],
            text: "Cog-AI Rapid Screening Report",
        },
        subject: {
            reference: `Patient/${patientId}`,
        },
        effectiveDateTime: timestamp,
        issued: timestamp,
        result: [
            {
                reference: `Observation/${observationResource.id}`,
            },
        ],
        conclusion: `Risk Level: ${result.riskLevel}. Recommended Action: ${result.recommendations[0] || "Consult Neurologist"}.`,
    };

    // Bundle it all together
    return {
        resourceType: "Bundle",
        type: "transaction",
        entry: [
            {
                resource: patientResource,
                request: { method: "PUT", url: `Patient/${patientId}` },
            },
            {
                resource: observationResource,
                request: { method: "POST", url: "Observation" },
            },
            {
                resource: diagnosticReport,
                request: { method: "POST", url: "DiagnosticReport" },
            },
        ],
    };
}

export function downloadFHIR(result: AssessmentResult, filename: string = "medical-record.json") {
    const fhirData = convertToFHIR(result);
    const blob = new Blob([JSON.stringify(fhirData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

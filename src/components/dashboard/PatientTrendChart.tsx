import { AssessmentResult } from "@/services/api";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
} from "recharts";

interface Props {
    history: AssessmentResult[];
}

export default function PatientTrendChart({ history }: Props) {
    const data = history.map((h) => ({
        date: new Date(h.generatedAt).toLocaleDateString(),
        risk: Math.round(h.probability * 100),
        memory: Math.round((h.subScores.memoryScore / 2) * 100), // Norm to 100, max is 2
        attention: Math.round((h.subScores.attentionScore / 4) * 100), // Norm to 100, max is 4
    }));

    return (
        <div className="w-full h-full min-h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis dataKey="date" fontSize={12} tickMargin={10} />
                    <YAxis unit="%" domain={[0, 100]} fontSize={12} />
                    <Tooltip
                        contentStyle={{ borderRadius: "8px", border: "1px solid #e2e8f0" }}
                        labelStyle={{ fontWeight: "bold", color: "#64748b" }}
                    />
                    <Legend wrapperStyle={{ paddingTop: "10px" }} />

                    <Line
                        type="monotone"
                        dataKey="risk"
                        name="Risk Probability"
                        stroke="#ef4444" // Red
                        strokeWidth={3}
                        dot={{ r: 4 }}
                        activeDot={{ r: 6 }}
                    />
                    <Line
                        type="monotone"
                        dataKey="memory"
                        name="Memory"
                        stroke="#3b82f6" // Blue
                        strokeWidth={2}
                        strokeDasharray="5 5"
                    />
                    <Line
                        type="monotone"
                        dataKey="attention"
                        name="Attention"
                        stroke="#10b981" // Green
                        strokeWidth={2}
                        strokeDasharray="5 5"
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

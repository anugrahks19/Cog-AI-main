import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Check, Info } from "lucide-react";

export default function Pricing() {
    const plans = [
        {
            name: "Basic",
            price: "$0",
            period: "/month",
            description: "Essential cognitive screening for everyone.",
            features: [
                "1 Cognitive Assessment per Month",
                "3 Daily Brain Gym Exercises",
                "Basic Risk Score (Traffic Light)",
                "Local Data Storage"
            ],
            cta: "Current Plan",
            popular: false,
        },
        {
            name: "Cog-AI Plus",
            price: "$14.99",
            period: "/month",
            description: "Advanced tools for proactive brain health.",
            features: [
                "Unlimited Cognitive Assessments",
                "Full Brain Gym Access (10+ Games)",
                "Advanced Analytics & Trends",
                "Family Link (Up to 5 Caregivers)",
                "Priority Support"
            ],
            cta: "Start 7-Day Free Trial",
            popular: true,
            badge: "Most Popular"
        },
        {
            name: "Clinical Provider",
            price: "$299",
            period: "/month /provider",
            description: "For Neurologists & Geriatricians.",
            features: [
                "Clinician Dashboard Access",
                "Patient Population Management",
                "HL7 / FHIR EMR Integration",
                "RPM CPT Code Reporting",
                "Dedicated Success Manager"
            ],
            cta: "Contact Sales",
            popular: false,
        }
    ];

    return (
        <div className="container mx-auto py-12 px-4">
            <div className="text-center space-y-4 mb-12">
                <h1 className="text-4xl font-bold tracking-tight">Invest in Your Cognitive Future</h1>
                <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                    Choose the plan that fits your needs. Whether you are an individual tracking your health or a clinician managing patients.
                </p>
            </div>

            <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
                {plans.map((plan) => (
                    <Card
                        key={plan.name}
                        className={`flex flex-col relative ${plan.popular ? "border-primary shadow-lg scale-105 z-10" : "border-border"}`}
                    >
                        {plan.popular && (
                            <div className="absolute top-0 right-0 transform translate-x-2 -translate-y-2">
                                <Badge className="bg-primary text-primary-foreground hover:bg-primary text-sm px-3 py-1 uppercase">{plan.badge}</Badge>
                            </div>
                        )}
                        <CardHeader>
                            <CardTitle className="text-2xl">{plan.name}</CardTitle>
                            <div className="mt-4 flex items-baseline">
                                <span className="text-4xl font-extrabold tracking-tight">{plan.price}</span>
                                <span className="text-muted-foreground ml-1">{plan.period}</span>
                            </div>
                            <CardDescription className="mt-2">{plan.description}</CardDescription>
                        </CardHeader>
                        <CardContent className="flex-1">
                            <ul className="space-y-3">
                                {plan.features.map((feature) => (
                                    <li key={feature} className="flex items-start gap-2">
                                        <Check className="h-5 w-5 text-green-500 shrink-0" />
                                        <span className="text-sm">{feature}</span>
                                    </li>
                                ))}
                            </ul>
                        </CardContent>
                        <CardFooter>
                            <Button
                                variant={plan.popular ? "default" : "outline"}
                                className="w-full"
                                size="lg"
                            >
                                {plan.cta}
                            </Button>
                        </CardFooter>
                    </Card>
                ))}
            </div>

            <div className="mt-16 text-center bg-muted/30 p-8 rounded-xl max-w-3xl mx-auto space-y-4">
                <div className="flex justify-center items-center gap-2 text-muted-foreground">
                    <Info className="h-5 w-5" />
                    <span className="font-semibold">For Investors & Partners</span>
                </div>
                <p className="text-sm text-muted-foreground">
                    Our B2B model integrates with major EMR systems (Epic, Cerner) via FHIR standards, allowing for seamless reimbursement workflows under CPT codes 99453, 99454, and 99483.
                </p>
            </div>
        </div>
    );
}

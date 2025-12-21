import { useMemo } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { UserProfile } from "@/context/UserSessionContext";

const onboardingSchema = z
  .object({
    name: z
      .string()
      .min(2, "Name must be at least 2 characters")
      .max(120, "Name is too long"),
    age: z
      .coerce.number({ invalid_type_error: "Age is required" })
      .int("Age must be a whole number")
      .min(18, "Must be 18 or older")
      .max(120, "Please enter a valid age"),
    gender: z.coerce.number().int().min(0).max(1),
    education: z.coerce.number().int().min(0).max(3),
    // Medical History (0=No, 1=Yes)
    family_history: z.coerce.number().int().min(0).max(1).default(0),
    diabetes: z.coerce.number().int().min(0).max(1).default(0),
    hypertension: z.coerce.number().int().min(0).max(1).default(0),
    depression: z.coerce.number().int().min(0).max(1).default(0),
    head_injury: z.coerce.number().int().min(0).max(1).default(0),
    // Lifestyle
    sleep_quality: z.coerce.number().min(4).max(10).default(7), // 4-10 scale
    physical_activity: z.coerce.number().min(0).max(10).default(5), // 0-10 scale
    smoking: z.coerce.number().int().min(0).max(1).default(0), // 0=No, 1=Yes

    language: z.string({ required_error: "Select a language" }),
    consent: z
      .literal(true, {
        errorMap: () => ({ message: "Consent is required to proceed" }),
      })
      .or(z.boolean()),
  })
  .refine((data) => data.consent === true, {
    message: "Consent is required to proceed",
    path: ["consent"],
  });

export type OnboardingFormValues = z.infer<typeof onboardingSchema>;

interface OnboardingFormProps {
  user?: UserProfile | null;
  onSubmit: (values: OnboardingFormValues) => Promise<void>;
  isSubmitting?: boolean;
}

const DEFAULT_LANGUAGES: Array<{ value: string; label: string }> = [
  { value: "en", label: "English" },
  { value: "hi", label: "हिन्दी (Hindi)" },
  { value: "bn", label: "বাংলা (Bengali)" },
  { value: "ta", label: "தமிழ் (Tamil)" },
  { value: "te", label: "తెలుగు (Telugu)" },
  { value: "kn", label: "ಕನ್ನಡ (Kannada)" },
  { value: "ml", label: "മലയാളം (Malayalam)" },
  { value: "mr", label: "मराठी (Marathi)" },
  { value: "gu", label: "ગુજરાતી (Gujarati)" },
  { value: "pa", label: "ਪੰਜਾਬੀ (Punjabi)" },
];

export const OnboardingForm = ({
  user,
  onSubmit,
  isSubmitting,
}: OnboardingFormProps) => {
  const defaultValues = useMemo(
    () => ({
      name: user?.name ?? "",
      age: user?.age ?? ("" as unknown as number),
      gender: user?.gender ?? 0,
      education: user?.education ?? 1,
      family_history: user?.family_history ?? 0,
      diabetes: user?.diabetes ?? 0,
      hypertension: user?.hypertension ?? 0,
      depression: user?.depression ?? 0,
      head_injury: user?.head_injury ?? 0,
      sleep_quality: user?.sleep_quality ?? 7,
      physical_activity: user?.physical_activity ?? 5,
      smoking: user?.smoking ?? 0,
      language: user?.language ?? "en",
      consent: user?.consent ?? false,
    }),
    [user],
  );

  const form = useForm<OnboardingFormValues>({
    resolver: zodResolver(onboardingSchema),
    defaultValues,
  });

  const handleSubmit = async (values: OnboardingFormValues) => {
    await onSubmit(values);
  };

  return (
    <Form {...form}>
      <form
        onSubmit={form.handleSubmit(handleSubmit)}
        className="space-y-6"
        noValidate
      >
        <div className="grid gap-6 md:grid-cols-2">
          <FormField
            control={form.control}
            name="name"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Full Name</FormLabel>
                <FormControl>
                  <Input placeholder="Enter your full name" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="age"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Age</FormLabel>
                <FormControl>
                  <Input
                    type="number"
                    min={18}
                    max={120}
                    placeholder="Your age"
                    {...field}
                    onChange={(event) =>
                      field.onChange(Number(event.target.value))
                    }
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="gender"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Gender</FormLabel>
                <Select
                  onValueChange={(val) => field.onChange(Number(val))}
                  defaultValue={String(field.value)}
                >
                  <FormControl>
                    <SelectTrigger>
                      <SelectValue placeholder="Select Gender" />
                    </SelectTrigger>
                  </FormControl>
                  <SelectContent>
                    <SelectItem value="0">Male</SelectItem>
                    <SelectItem value="1">Female</SelectItem>
                  </SelectContent>
                </Select>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="education"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Education Level</FormLabel>
                <Select
                  onValueChange={(val) => field.onChange(Number(val))}
                  defaultValue={String(field.value)}
                >
                  <FormControl>
                    <SelectTrigger>
                      <SelectValue placeholder="Select Education" />
                    </SelectTrigger>
                  </FormControl>
                  <SelectContent>
                    <SelectItem value="0">None</SelectItem>
                    <SelectItem value="1">High School</SelectItem>
                    <SelectItem value="2">Bachelor's</SelectItem>
                    <SelectItem value="3">Higher Degree</SelectItem>
                  </SelectContent>
                </Select>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>

        <div className="space-y-4 rounded-lg border p-4">
          <h3 className="text-lg font-medium">Health & Lifestyle</h3>
          <div className="grid gap-6 md:grid-cols-2">

            {/* Medical History Checklist-style Selects */}
            <FormField
              control={form.control}
              name="family_history"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Family History of Alzheimer's?</FormLabel>
                  <Select onValueChange={(v) => field.onChange(Number(v))} defaultValue={String(field.value)}>
                    <FormControl><SelectTrigger><SelectValue /></SelectTrigger></FormControl>
                    <SelectContent><SelectItem value="0">No</SelectItem><SelectItem value="1">Yes</SelectItem></SelectContent>
                  </Select>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="diabetes"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Do you have Diabetes?</FormLabel>
                  <Select onValueChange={(v) => field.onChange(Number(v))} defaultValue={String(field.value)}>
                    <FormControl><SelectTrigger><SelectValue /></SelectTrigger></FormControl>
                    <SelectContent><SelectItem value="0">No</SelectItem><SelectItem value="1">Yes</SelectItem></SelectContent>
                  </Select>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="hypertension"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>High Blood Pressure?</FormLabel>
                  <Select onValueChange={(v) => field.onChange(Number(v))} defaultValue={String(field.value)}>
                    <FormControl><SelectTrigger><SelectValue /></SelectTrigger></FormControl>
                    <SelectContent><SelectItem value="0">No</SelectItem><SelectItem value="1">Yes</SelectItem></SelectContent>
                  </Select>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="smoking"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Do you Smoke?</FormLabel>
                  <Select onValueChange={(v) => field.onChange(Number(v))} defaultValue={String(field.value)}>
                    <FormControl><SelectTrigger><SelectValue /></SelectTrigger></FormControl>
                    <SelectContent><SelectItem value="0">No</SelectItem><SelectItem value="1">Yes</SelectItem></SelectContent>
                  </Select>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="sleep_quality"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Sleep Quality (1-10)</FormLabel>
                  <Select onValueChange={(v) => field.onChange(Number(v))} defaultValue={String(field.value)}>
                    <FormControl><SelectTrigger><SelectValue /></SelectTrigger></FormControl>
                    <SelectContent>
                      {[...Array(11).keys()].slice(4).map(i => <SelectItem key={i} value={String(i)}>{i}</SelectItem>)}
                    </SelectContent>
                  </Select>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="physical_activity"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Activity Level (0-10)</FormLabel>
                  <Select onValueChange={(v) => field.onChange(Number(v))} defaultValue={String(field.value)}>
                    <FormControl><SelectTrigger><SelectValue /></SelectTrigger></FormControl>
                    <SelectContent>
                      {[...Array(11).keys()].map(i => <SelectItem key={i} value={String(i)}>{i}</SelectItem>)}
                    </SelectContent>
                  </Select>
                  <FormMessage />
                </FormItem>
              )}
            />
          </div>
        </div>

        <FormField
          control={form.control}
          name="language"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Preferred Language</FormLabel>
              <Select onValueChange={field.onChange} value={field.value}>
                <FormControl>
                  <SelectTrigger>
                    <SelectValue placeholder="Choose your language" />
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  {DEFAULT_LANGUAGES.map((language) => (
                    <SelectItem key={language.value} value={language.value}>
                      {language.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <FormMessage />
            </FormItem>
          )}
        />

        <FormField
          control={form.control}
          name="consent"
          render={({ field }) => (
            <FormItem className="space-y-3">
              <div className="flex items-start space-x-3 rounded-lg border p-4">
                <FormControl>
                  <Checkbox
                    checked={Boolean(field.value)}
                    onCheckedChange={(value) => field.onChange(Boolean(value))}
                  />
                </FormControl>
                <div className="space-y-1 text-sm">
                  <FormLabel className="text-base font-semibold">
                    Consent & Privacy Agreement
                  </FormLabel>
                  <p className="text-muted-foreground">
                    I confirm that I am providing my informed consent to record
                    speech samples, store and analyze my cognitive assessment
                    data, and process them securely for early dementia screening.
                  </p>
                </div>
              </div>
              <FormMessage />
            </FormItem>
          )}
        />

        <Button type="submit" size="lg" className="w-full" disabled={isSubmitting}>
          {isSubmitting ? "Registering..." : "Agree & Continue"}
        </Button>
      </form>
    </Form>
  );
};

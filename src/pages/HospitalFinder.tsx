import { useState } from "react";
import { MapPin, Navigation, Shield, RefreshCw, PhoneCall, HeartPulse, Compass, Globe2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const HospitalFinder = () => {
  const [status, setStatus] = useState<"idle" | "locating" | "denied" | "unsupported" | "error">("idle");
  const [message, setMessage] = useState<string>("");

  const openGeneralSearch = () => {
    window.open("https://www.google.com/maps/search/hospitals+near+me", "_blank", "noopener,noreferrer");
  };

  const handleLocate = () => {
    if (typeof window === "undefined" || !("geolocation" in navigator)) {
      setStatus("unsupported");
      setMessage("Location access is not available in this browser. Use the general search instead.");
      return;
    }

    setStatus("locating");
    setMessage("Getting your location…");

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        const mapsUrl = `https://www.google.com/maps/search/hospitals/@${latitude},${longitude},15z`;
        window.open(mapsUrl, "_blank", "noopener,noreferrer");
        setStatus("idle");
        setMessage("Opened nearby hospital results in Google Maps.");
      },
      (error) => {
        if (error.code === error.PERMISSION_DENIED) {
          setStatus("denied");
          setMessage("Permission denied. Please enable location access or use the general hospital search link.");
        } else {
          setStatus("error");
          setMessage("Could not fetch your location. Try again or use the general hospital search link.");
        }
      },
      { enableHighAccuracy: true, timeout: 12_000, maximumAge: 0 },
    );
  };

  const highlights = [
    {
      icon: Compass,
      title: "Location aware",
      description: "Quickly open a trusted hospital search near where you are without sharing the data with our servers.",
    },
    {
      icon: HeartPulse,
      title: "Care-first guidance",
      description: "We pair physical hospital search with mental-health helplines so you always know who to call.",
    },
    {
      icon: Globe2,
      title: "Works anywhere",
      description: "If location access is blocked, use our curated maps link to browse hospitals manually in seconds.",
    },
  ];

  const helplines = [
    {
      label: "Tele-MANAS",
      number: "14416 / 1-800-891-4416",
      time: "24×7",
      url: "https://telemanas.mohfw.gov.in",
    },
    {
      label: "Vandrevala Foundation",
      number: "9999 666 555",
      time: "24×7",
      url: "https://www.vandrevalafoundation.com",
    },
    {
      label: "AASRA",
      number: "+91 9820466726",
      time: "24×7",
      url: "http://www.aasra.info/helpline.html",
    },
    {
      label: "iCall (TISS)",
      number: "+91 9152987821",
      time: "Mon–Sat, 10 AM – 8 PM",
      url: "https://icallhelpline.org",
    },
  ];

  return (
    <div className="min-h-screen relative">
      <section className="relative overflow-hidden py-24 px-4 sm:px-6 lg:px-10">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/25 via-primary/10 to-secondary/25 opacity-90" />
        <div className="relative max-w-5xl mx-auto text-center space-y-6">
          <div className="inline-flex items-center gap-3 px-5 py-2 rounded-full bg-[#7c3aed] text-white shadow-lg shadow-primary/40">
            <span className="text-sm font-semibold tracking-wide uppercase">Need urgent care?</span>
            <Button
              variant="secondary"
              size="sm"
              className="h-8 bg-white dark:bg-card text-[#5b21b6] dark:text-primary font-semibold hover:bg-white/90 dark:hover:bg-card/90 border-none"
              onClick={openGeneralSearch}
            >
              Open hospital map
            </Button>
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold text-foreground">
            Hospital & Emergency Support Finder
          </h1>
          <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
            Get directions to nearby hospitals instantly, or reach a trusted mental-health helpline. Powered by Google Maps and
            curated national resources—no data stored on our servers.
          </p>
        </div>
      </section>

      <section className="-mt-16 pb-20 px-4 sm:px-6 lg:px-10">
        <div className="max-w-5xl mx-auto space-y-12">
          <div className="relative z-10 grid grid-cols-1 md:grid-cols-3 gap-6">
            {highlights.map((item) => (
              <Card
                key={item.title}
                className="bg-white dark:bg-card border border-white/50 dark:border-border shadow-[0_15px_30px_-18px_rgba(79,70,229,0.35)] dark:shadow-none transition-transform duration-200 hover:-translate-y-1.5 hover:shadow-[0_25px_45px_-20px_rgba(79,70,229,0.45)]"
              >
                <CardHeader className="space-y-4">
                  <div className="p-3 w-fit rounded-xl bg-[#ede9fe] text-[#5b21b6] dark:bg-primary/20 dark:text-primary">
                    <item.icon className="h-6 w-6" />
                  </div>
                  <CardTitle className="text-xl text-foreground">{item.title}</CardTitle>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    {item.description}
                  </p>
                </CardHeader>
              </Card>
            ))}
          </div>

          <Card className="shadow-card border-border/70 bg-card/80">
            <CardHeader className="space-y-2">
              <CardTitle className="flex flex-wrap items-center gap-2 text-2xl">
                <MapPin className="h-6 w-6 text-primary" />
                Quick Hospital Locator
              </CardTitle>
              <CardDescription>
                Allow location access to open Google Maps with hospitals near you. We never capture or store your coordinates.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <div className="flex flex-wrap gap-3">
                <Button onClick={handleLocate} disabled={status === "locating"} className="flex items-center gap-2">
                  {status === "locating" ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Navigation className="h-4 w-4" />}
                  Locate me & open map
                </Button>
                <Button variant="outline" onClick={openGeneralSearch} className="flex items-center gap-2">
                  <MapPin className="h-4 w-4" />
                  Open hospital search
                </Button>
              </div>
              {message && (
                <div className="text-sm text-muted-foreground border border-border rounded-md p-3 bg-muted/40">
                  {message}
                </div>
              )}
              {status === "denied" && (
                <p className="text-xs text-muted-foreground">
                  Permission denied. Clear the location block in your browser settings and try again, or use the manual search link above.
                </p>
              )}
              {status === "unsupported" && (
                <p className="text-xs text-muted-foreground">
                  Your device or browser does not support geolocation. Please use the hospital search link instead.
                </p>
              )}
            </CardContent>
          </Card>

          <Card className="shadow-card border-border/70 bg-card/85">
            <CardHeader className="space-y-2">
              <CardTitle className="flex flex-wrap items-center gap-2 text-2xl">
                <Shield className="h-6 w-6 text-primary" />
                Mental Health Helplines (24×7)
              </CardTitle>
              <CardDescription>
                When emotions feel heavy, counsellors are just a call away. All services listed below are free and confidential.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {helplines.map((item) => (
                  <div key={item.label} className="rounded-lg border border-border/70 bg-muted/40 p-4 space-y-2">
                    <div className="flex items-center gap-2">
                      <PhoneCall className="h-4 w-4 text-primary" />
                      <p className="font-semibold text-foreground">{item.label}</p>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      📞 {item.number}
                      <br />
                      🕒 {item.time}
                    </p>
                    <a
                      href={item.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-xs text-primary hover:underline"
                    >
                      Visit website
                    </a>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </section>
    </div>
  );
};

export default HospitalFinder;

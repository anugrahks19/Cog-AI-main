import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate, useLocation } from "react-router-dom";
import Navigation from "./components/Navigation";
import Footer from "./components/Footer";
import Index from "./pages/Index";
import About from "./pages/About";
import HowItWorks from "./pages/HowItWorks";
import Features from "./pages/Features";
import Resources from "./pages/Resources";
import Contact from "./pages/Contact";
import NotFound from "./pages/NotFound";
import Assessment from "./pages/Assessment";
import Chatbot from "./components/Chatbot";
import ParticleBackground from "./components/ParticleBackground";
import ScrollToTop from "./components/ScrollToTop";
import HospitalFinder from "./pages/HospitalFinder";
import ClinicianDashboard from "./pages/ClinicianDashboard";
import BrainGym from "./pages/BrainGym";
import Pricing from "./pages/Pricing";

const queryClient = new QueryClient();

const AppContent = () => {
  const location = useLocation();
  const isHome = location.pathname === "/";
  const particleCount = isHome ? 90 : 30;

  return (
    <>
      <ScrollToTop />
      <div className="min-h-screen flex flex-col">
        <ParticleBackground count={particleCount} />
        <Navigation />
        <main className="flex-1 pt-16">
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/about" element={<About />} />
            <Route path="/how-it-works" element={<HowItWorks />} />
            <Route path="/features" element={<Features />} />
            <Route path="/demo" element={<Navigate to="/assessment" replace />} />
            <Route path="/assessment" element={<Assessment />} />
            <Route path="/resources" element={<Resources />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/pricing" element={<Pricing />} /> {/* Added Pricing route */}
            <Route path="/hospital-finder" element={<HospitalFinder />} />
            <Route path="/clinician" element={<ClinicianDashboard />} />
            <Route path="/brain-gym" element={<BrainGym />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </main>
        <Footer />
        {/* Floating chatbot */}
        <Chatbot />
      </div>
    </>
  );
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <AppContent />
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;

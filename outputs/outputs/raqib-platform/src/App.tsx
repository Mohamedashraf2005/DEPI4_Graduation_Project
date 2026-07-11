import { useEffect } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { I18nProvider } from "./i18n/I18nContext";
import { GuideProvider } from "./components/Onboarding";
import { MinistryLayout } from "./components/MinistryLayout";
import { CitizenLayout } from "./components/CitizenLayout";
import { Landing } from "./pages/Landing";
import { Analyze } from "./pages/ministry/Analyze";
import { Reports } from "./pages/ministry/Reports";
import { CaseReview } from "./pages/ministry/CaseReview";
import { Risk } from "./pages/ministry/Risk";
import { Heatmap } from "./pages/ministry/Heatmap";
import { Chatbot } from "./pages/ministry/chatbot";
import { RoadRiskAnalyzer } from "./pages/ministry/RoadRiskAnalyzer.tsx";
//outputs\outputs\raqib-platform\src\pages\ministry\RoadRiskAnalyzer.tsx
import { CitizenHome } from "./pages/citizen/CitizenHome";
import { CitizenReport } from "./pages/citizen/CitizenReport";
import { CitizenTrack } from "./pages/citizen/CitizenTrack";
import { CitizenRisk } from "./pages/citizen/CitizenRisk";
// --- Fixed ScrollToTop to guarantee no async Promise return values ---
   
   // Inside <Route path="/app" element={<MinistryLayout />}>
function ScrollToTop() {
  const { pathname } = useLocation();
  
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  
  return null;
}
export default function App() {
  console.log("ENV:", import.meta.env);
  console.log("POTHOLE:", import.meta.env.VITE_MODEL_POTHOLE_URL);
  console.log("ACCIDENT:", import.meta.env.VITE_MODEL_ACCIDENT_URL);
  console.log("MOCK:", import.meta.env.VITE_USE_MOCK);
  return (
    <I18nProvider>
      <BrowserRouter>
        <GuideProvider>
          <ScrollToTop />
          <Routes>
            {/* Landing */}
            <Route path="/" element={<Landing />} />
            
            {/* Ministry Routes under Layout */}
            <Route path="/app" element={<MinistryLayout />}>
              <Route index element={<Analyze />} />
              <Route path="reports" element={<Reports />} />
              <Route path="reports/:ref" element={<CaseReview />} />
              <Route path="risk" element={<Risk />} />
              <Route path="heatmap" element={<Heatmap />} />
              <Route path="chatbot" element={<Chatbot />} />
              <Route path="roadriskanalyzer" element={<RoadRiskAnalyzer />} />
            </Route>
            
            {/* Citizen Routes under Layout */}
            <Route path="/citizen" element={<CitizenLayout />}>
              <Route index element={<CitizenHome />} />
              <Route path="report" element={<CitizenReport />} />
              <Route path="track" element={<CitizenTrack />} />
              <Route path="risk" element={<CitizenRisk />} />
            </Route>
            
            {/* Fallback Catch-All Routing */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </GuideProvider>
      </BrowserRouter>
    </I18nProvider>
  );
}

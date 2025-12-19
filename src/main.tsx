import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";
import { UserSessionProvider } from "./context/UserSessionContext";

createRoot(document.getElementById("root")!).render(
  <UserSessionProvider>
    <App />
  </UserSessionProvider>,
);

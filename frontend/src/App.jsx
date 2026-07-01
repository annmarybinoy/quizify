/**
 * App.jsx - Root Component
 *
 * Sets up routing for the entire application.
 * Each route maps a URL to a page component.
 */

import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import Home from "./pages/Home";
import CreateQuiz from "./pages/CreateQuiz";
import QuizPreview from "./pages/QuizPreview";
import JoinRoom from "./pages/JoinRoom";
import PlayQuiz from "./pages/PlayQuiz";
import Results from "./pages/Results";

// ── React Query client ─────────────────────────────────────────────────────────
// Manages all API call states — loading, error, success, caching
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,             // retry failed requests once
      staleTime: 1000 * 60, // cache data for 1 minute
    },
  },
});


export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/create" element={<CreateQuiz />} />
          <Route path="/quiz/:quizId" element={<QuizPreview />} />
          <Route path="/join" element={<JoinRoom />} />
          <Route path="/play/:roomCode" element={<PlayQuiz />} />
          <Route path="/results/:roomCode" element={<Results />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
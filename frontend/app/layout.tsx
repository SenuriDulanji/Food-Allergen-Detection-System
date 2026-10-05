import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata: Metadata = {
  title: "AllerScan — Sri Lankan Dish Allergen Detector",
  description:
    "AI-powered food allergen detection for Sri Lankan dishes. Scan any dish photo to get instant allergen analysis powered by Gemini Vision + ML risk prediction.",
  keywords: ["allergen", "food safety", "Sri Lankan cuisine", "AI", "deep learning"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="noise mesh-bg min-h-screen">
        <Navbar />
        <main className="relative z-10">{children}</main>
        <footer className="relative z-10 border-t border-white/5 mt-24 py-8 text-center text-sm text-[var(--text-muted)]">
          <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-2">
            <span>
              <span className="gradient-text font-semibold">AllerScan</span> — AI-Based Food Allergen Detection for Sri Lankan Dishes
            </span>
            <span>Powered by Gemini Vision ·  logistic regression · Apriori Rules</span>
          </div>
        </footer>
      </body>
    </html>
  );
}

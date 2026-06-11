import type { Metadata } from "next";
import { Inter, Outfit } from "next/font/google";
import "./globals.css";

const outfit = Outfit({
  subsets: ["latin"],
  weight: ["600", "700", "800"],
  variable: "--font-outfit",
  display: "swap",
});

const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Text Models of arena.ai - ELO history, open-source race, price-performance",
  description:
    "Daily snapshots of the arena.ai text leaderboard, kept as history: ELO trajectories of the top models, the open-source race, and price versus performance.",
  openGraph: {
    title: "Text Models of arena.ai",
    description:
      "Daily snapshots of the arena.ai text leaderboard, kept as history: ELO trajectories of the top models, the open-source race, and price versus performance.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${outfit.variable} ${inter.variable}`}>
      <body>{children}</body>
    </html>
  );
}

import type { Metadata } from "next";
import {
  Geist_Mono,
  Instrument_Serif,
  Montserrat,
} from "next/font/google";

import { DevTools } from "@/app/dev-tools";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

const montserrat = Montserrat({
  variable: "--font-montserrat",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const instrumentSerif = Instrument_Serif({
  variable: "--font-instrument-serif",
  subsets: ["latin"],
  weight: "400",
});

export const metadata: Metadata = {
  title: "RAG Intelligence",
  description: "Análise de partidas CS:GO com inteligência artificial",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="pt-BR" className="dark">
      <body
        className={`${montserrat.variable} ${geistMono.variable} ${instrumentSerif.variable} antialiased`}
      >
        <DevTools />
        <TooltipProvider>{children}</TooltipProvider>
      </body>
    </html>
  );
}

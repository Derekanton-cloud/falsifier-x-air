import type { Metadata } from "next";
import "./globals.css";
import "./refinement.css";
import "./graph.css";

export const metadata: Metadata = {
  title: "FALSIFIER-X AIR | Research Console",
  description: "Experimental falsification, identifiability and model repair research console.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}

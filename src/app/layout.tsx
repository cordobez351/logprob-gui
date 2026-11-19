import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Logprob GUI",
  description:
    "Retro terminal-style chat that shows how steady or wavering the model feels as it writes—so you can watch the attitude behind each piece of text, not just the final answer.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

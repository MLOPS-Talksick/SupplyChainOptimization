import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import { ThemeToggle } from "@/components/theme-toggle";
import Link from "next/link";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Supply Chain Optimization",
  description: "Optimize your supply chain with data-driven insights",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <div className="flex min-h-screen flex-col">
            <header className="border-b">
              <div className="flex h-16 items-center justify-between px-8">
                <Link href="/" className="text-2xl font-bold">
                  Supply Chain Optimization
                </Link>
                <nav className="flex gap-6 mx-6 flex-1 justify-center">
                  <Link
                    href="/dashboard"
                    className="font-medium hover:text-primary transition-colors"
                  >
                    Dashboard
                  </Link>
                  <Link
                    href="/upload"
                    className="font-medium hover:text-primary transition-colors"
                  >
                    Upload Data
                  </Link>
                  <Link
                    href="/forecast"
                    className="font-medium hover:text-primary transition-colors"
                  >
                    Forecast
                  </Link>
                </nav>
                <ThemeToggle />
              </div>
            </header>
            <main className="flex-1 space-y-4 p-8 pt-6">{children}</main>
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "motion/react";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";
import { MenuIcon } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";

function NavLink({
  href,
  children,
}: {
  href: string;
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const isActive = pathname === href;

  return (
    <Link
      href={href}
      className={cn(
        "relative px-4 py-2 font-medium transition-colors",
        isActive ? "text-primary" : "hover:text-primary"
      )}
    >
      {isActive && (
        <motion.div
          layoutId="activeRoute"
          className="absolute inset-0 h-full w-full rounded-full bg-primary/10"
          transition={{
            type: "spring",
            stiffness: 500,
            damping: 30,
          }}
        />
      )}
      <span className="relative z-10">{children}</span>
    </Link>
  );
}

export default function NavigationMenu() {
  const [isMobile, setIsMobile] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    // Check if screen is mobile size
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 640);
    };

    // Initial check
    checkMobile();

    // Add event listener
    window.addEventListener("resize", checkMobile);

    // Clean up
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  const navItems = [
    { href: "/dashboard", label: "Dashboard" },
    { href: "/upload", label: "Upload Data" },
    { href: "/forecast", label: "Forecast" },
  ];

  return (
    <nav className="flex gap-2 mx-6 flex-1 justify-center">
      {isMobile ? (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button className="flex items-center gap-2 px-4 py-2 rounded-md hover:bg-primary/10">
              <MenuIcon className="h-5 w-5" />
              <span className="font-medium">Menu</span>
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48">
            {navItems.map((item) => (
              <DropdownMenuItem key={item.href} asChild>
                <Link
                  href={item.href}
                  className={cn(
                    "w-full",
                    pathname === item.href && "bg-primary/10 font-medium"
                  )}
                >
                  {item.label}
                </Link>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      ) : (
        <>
          <NavLink href="/dashboard">Dashboard</NavLink>
          <NavLink href="/upload">Upload Data</NavLink>
          <NavLink href="/forecast">Forecast</NavLink>
        </>
      )}
    </nav>
  );
}

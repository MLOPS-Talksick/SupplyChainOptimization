"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "motion/react";
import { cn } from "@/lib/utils";

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
  return (
    <nav className="flex gap-2 mx-6 flex-1 justify-center">
      <NavLink href="/dashboard">Dashboard</NavLink>
      <NavLink href="/upload">Upload Data</NavLink>
      <NavLink href="/forecast">Forecast</NavLink>
    </nav>
  );
}

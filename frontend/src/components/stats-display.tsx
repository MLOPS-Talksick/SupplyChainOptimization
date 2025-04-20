"use client";

import { useEffect, useState } from "react";
import { Loader2, BarChart2, ShoppingCart, Calendar } from "lucide-react";
import { GlowingEffect } from "@/components/ui/glowing-effect";

interface Stats {
  start_date: string;
  end_date: string;
  total_entries: number;
  total_products: number;
}

export default function StatsDisplay() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      setLoading(true);
      try {
        const response = await fetch(`/api/proxy?endpoint=get-stats`, {
          method: "GET",
          headers: {
            token: "backendapi1234567890",
          },
        });

        if (!response.ok) {
          throw new Error(`Error fetching stats: ${response.status}`);
        }

        const data = await response.json();
        setStats(data);
      } catch (err) {
        console.error("Failed to fetch stats:", err);
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  const formatDateRange = () => {
    if (!stats) return "--";

    // Format dates for display
    const formatDate = (dateStr: string) => {
      const date = new Date(dateStr);
      return date.toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
    };

    return `${formatDate(stats.start_date)} - ${formatDate(stats.end_date)}`;
  };

  if (loading) {
    return (
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="flex items-center justify-center p-6 border rounded-lg"
          >
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="p-6 border rounded-lg bg-destructive/10 text-destructive"
          >
            Error loading stats
          </div>
        ))}
      </div>
    );
  }

  return (
    <ul className="grid grid-cols-1 gap-2 md:grid-cols-12 mb-4">
      <StatsItem
        area="md:[grid-area:1/1/2/5]"
        icon={
          <ShoppingCart className="h-4 w-4 text-black dark:text-neutral-400" />
        }
        title="Total Products"
        description={
          <>
            <span className="text-xl font-bold">
              {stats?.total_products || "--"}
            </span>
            <p className="text-xs text-muted-foreground">
              Unique products in database
            </p>
          </>
        }
      />

      <StatsItem
        area="md:[grid-area:1/5/2/9]"
        icon={
          <BarChart2 className="h-4 w-4 text-black dark:text-neutral-400" />
        }
        title="Total Records"
        description={
          <>
            <span className="text-xl font-bold">
              {stats?.total_entries?.toLocaleString() || "--"}
            </span>
            <p className="text-xs text-muted-foreground">
              Sales records in database
            </p>
          </>
        }
      />

      <StatsItem
        area="md:[grid-area:1/9/2/13]"
        icon={<Calendar className="h-4 w-4 text-black dark:text-neutral-400" />}
        title="Date Range"
        description={
          <>
            <span className="text-xl font-bold">{formatDateRange()}</span>
            <p className="text-xs text-muted-foreground">
              Covered by available data
            </p>
          </>
        }
      />
    </ul>
  );
}

interface StatsItemProps {
  area: string;
  icon: React.ReactNode;
  title: string;
  description: React.ReactNode;
}

const StatsItem = ({ area, icon, title, description }: StatsItemProps) => {
  return (
    <li className={`min-h-[5rem] list-none ${area}`}>
      <div className="relative h-full rounded-xl border p-1 md:rounded-2xl md:p-2">
        <GlowingEffect
          spread={40}
          glow={true}
          disabled={false}
          proximity={64}
          inactiveZone={0.01}
        />
        <div className="border-0.75 relative flex h-full flex-col justify-between gap-1 overflow-hidden rounded-lg p-2 md:p-3 dark:shadow-[0px_0px_27px_0px_#2D2D2D]">
          <div className="relative flex flex-1 flex-col justify-between gap-2">
            <div className="w-fit rounded-lg border border-gray-600 p-1.5">
              {icon}
            </div>
            <div className="space-y-2">
              <h3 className="font-sans text-base font-semibold text-balance text-black dark:text-white">
                {title}
              </h3>
              <div className="font-sans text-sm/[1.125rem] text-black dark:text-neutral-400">
                {description}
              </div>
            </div>
          </div>
        </div>
      </div>
    </li>
  );
};

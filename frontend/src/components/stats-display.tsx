"use client";

import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

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
        const response = await fetch("/api/get-stats", {
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
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      <div className="p-6 border rounded-lg">
        <h3 className="text-sm font-medium text-muted-foreground mb-2">
          Total Products
        </h3>
        <div className="text-2xl font-bold">
          {stats?.total_products || "--"}
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Unique products in database
        </p>
      </div>

      <div className="p-6 border rounded-lg">
        <h3 className="text-sm font-medium text-muted-foreground mb-2">
          Total Records
        </h3>
        <div className="text-2xl font-bold">
          {stats?.total_entries?.toLocaleString() || "--"}
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Sales records in database
        </p>
      </div>

      <div className="p-6 border rounded-lg">
        <h3 className="text-sm font-medium text-muted-foreground mb-2">
          Date Range
        </h3>
        <div className="text-2xl font-bold">{formatDateRange()}</div>
        <p className="text-xs text-muted-foreground mt-1">
          Covered by available data
        </p>
      </div>
    </div>
  );
}

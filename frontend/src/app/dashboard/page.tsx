import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import DataTable from "@/components/data-table";
import StatsDisplay from "@/components/stats-display";
import { GlowingEffect } from "@/components/ui/glowing-effect";

export default function DashboardPage() {
  return (
    <div className="space-y-4">
      <StatsDisplay />
      <div className="grid gap-4 md:grid-cols-1">
        <Card className="col-span-1 relative">
          <GlowingEffect
            spread={40}
            glow={true}
            disabled={false}
            proximity={64}
            inactiveZone={0.01}
          />
          <div className="relative">
            <CardHeader>
              <CardTitle>Recent Data</CardTitle>
              <CardDescription>
                Latest sales records from database
              </CardDescription>
            </CardHeader>
            <CardContent>
              <DataTable />
            </CardContent>
          </div>
        </Card>
      </div>
    </div>
  );
}

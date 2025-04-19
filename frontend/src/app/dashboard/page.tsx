import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import DataTable from "@/components/data-table";
import StatsDisplay from "@/components/stats-display";

export default function DashboardPage() {
  return (
    <div className="space-y-4">
      <StatsDisplay />
      <div className="grid gap-4 md:grid-cols-1">
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle>Recent Data</CardTitle>
            <CardDescription>
              Latest sales records from database
            </CardDescription>
          </CardHeader>
          <CardContent>
            <DataTable />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

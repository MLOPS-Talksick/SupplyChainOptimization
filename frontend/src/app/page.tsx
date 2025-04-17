import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import UploadForm from "@/components/upload-form";
import DataTable from "@/components/data-table";
import ForecastSection from "@/components/forecast-section";
import { ThemeToggle } from "@/components/theme-toggle";
import StatsDisplay from "@/components/stats-display";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <div className="flex-1 space-y-4 p-8 pt-6">
        <div className="flex items-center justify-between">
          <h2 className="text-3xl font-bold tracking-tight">
            Supply Chain Optimization
          </h2>
          <ThemeToggle />
        </div>
        <Tabs defaultValue="dashboard" className="space-y-4">
          <TabsList>
            <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
            <TabsTrigger value="upload">Upload Data</TabsTrigger>
            <TabsTrigger value="forecast">Forecast</TabsTrigger>
          </TabsList>
          <TabsContent value="dashboard" className="space-y-4">
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
          </TabsContent>
          <TabsContent value="upload" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Upload Data</CardTitle>
                <CardDescription>
                  Upload Excel files with sales data to the database
                </CardDescription>
              </CardHeader>
              <CardContent>
                <UploadForm />
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="forecast" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Demand Forecasting</CardTitle>
                <CardDescription>
                  Predict future demand for products
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ForecastSection />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

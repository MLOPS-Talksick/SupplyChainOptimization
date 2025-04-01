"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { DatePicker } from "@/components/ui/date-picker";
import { Card, CardContent } from "@/components/ui/card";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { AlertCircle } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface ForecastResult {
  dates: string[];
  predictions: number[];
}

export default function ForecastSection() {
  const [productId, setProductId] = useState("");
  const [forecastPeriod, setForecastPeriod] = useState("30");
  const [startDate, setStartDate] = useState<Date | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [forecastData, setForecastData] = useState<ForecastResult | null>(null);

  const handleForecast = async () => {
    if (!productId) {
      setError("Please enter a product ID");
      return;
    }

    if (!startDate) {
      setError("Please select a start date");
      return;
    }

    setLoading(true);
    setError(null);

    // Format date to YYYY-MM-DD
    const formattedDate = startDate.toISOString().split("T")[0];

    try {
      // This is a placeholder - we need to implement a forecast endpoint on the backend
      const response = await fetch("http://localhost:3000/forecast", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          product_id: productId,
          start_date: formattedDate,
          periods: parseInt(forecastPeriod, 10),
        }),
      });

      const result = await response.json();

      if (response.ok) {
        setForecastData(result);
      } else {
        setError(result.error || "Failed to generate forecast");
      }
    } catch {
      setError("Error connecting to the server. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // Transform data for recharts
  const chartData = forecastData
    ? forecastData.dates.map((date, index) => ({
        date: new Date(date).toLocaleDateString(),
        demand: forecastData.predictions[index],
      }))
    : [];

  return (
    <div className="space-y-8">
      <div className="grid gap-6 md:grid-cols-2">
        <div>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="product-id">Product ID</Label>
              <Input
                id="product-id"
                placeholder="Enter product ID"
                value={productId}
                onChange={(e) => setProductId(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label>Forecast Start Date</Label>
              <DatePicker selected={startDate} onSelect={setStartDate} />
            </div>

            <div className="space-y-2">
              <Label htmlFor="forecast-period">Forecast Period (Days)</Label>
              <Select value={forecastPeriod} onValueChange={setForecastPeriod}>
                <SelectTrigger id="forecast-period">
                  <SelectValue placeholder="Select period" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="7">7 days</SelectItem>
                  <SelectItem value="14">14 days</SelectItem>
                  <SelectItem value="30">30 days</SelectItem>
                  <SelectItem value="90">90 days</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Button
              onClick={handleForecast}
              disabled={loading}
              className="w-full"
            >
              {loading ? "Generating Forecast..." : "Generate Forecast"}
            </Button>

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </div>
        </div>

        <Card>
          <CardContent className="pt-6">
            {forecastData ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar
                    dataKey="demand"
                    fill="#3b82f6"
                    name="Predicted Demand"
                  />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-[300px] text-center text-muted-foreground">
                Generate a forecast to see predictions
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {forecastData && (
        <div className="rounded-md border overflow-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Date
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Predicted Demand
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {forecastData.dates.map((date, index) => (
                <tr key={date}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {new Date(date).toLocaleDateString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {forecastData.predictions[index].toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

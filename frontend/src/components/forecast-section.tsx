"use client";

import { useState, useEffect, useCallback } from "react";
import { TrendingUp } from "lucide-react";
import { AlertCircle } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface ForecastData {
  dates: string[];
  quantities: number[];
  product: string;
}

export default function ForecastSection() {
  const [selectedProduct, setSelectedProduct] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [forecastData, setForecastData] = useState<ForecastData | null>(null);
  const [productNames, setProductNames] = useState<string[]>([]);

  // Memoize the fetchProductNames function to prevent unnecessary recreations
  const fetchProductNames = useCallback(async () => {
    if (productNames.length > 0) return; // Don't fetch if we already have data

    setLoading(true);
    setError(null);

    try {
      // Add cache busting parameter to prevent browser from caching the response
      const response = await fetch(`/api/forecast?t=${Date.now()}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          token: "backendapi1234567890",
        },
        body: JSON.stringify({
          days: 7,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        // Extract unique product names from the predictions
        const products = Object.values(data.predictions.product_name);
        const uniqueProducts = Array.from(new Set(products)) as string[];
        setProductNames(uniqueProducts);
      } else {
        setError("Failed to fetch product data");
      }
    } catch {
      setError("Error connecting to the server. Please try again.");
    } finally {
      setLoading(false);
    }
  }, [productNames.length]);

  // Fetch product list on component mount
  useEffect(() => {
    // Only fetch if needed
    if (productNames.length === 0) {
      fetchProductNames();
    }
  }, [fetchProductNames, productNames.length]);

  const handleForecast = async () => {
    if (!selectedProduct) {
      setError("Please select a product");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Add cache busting parameter
      const response = await fetch(`/api/forecast?t=${Date.now()}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          token: "backendapi1234567890",
        },
        body: JSON.stringify({
          days: 7,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        // Process data for the selected product
        const indices = Object.keys(data.predictions.product_name).filter(
          (key) => data.predictions.product_name[key] === selectedProduct
        );

        // Extract dates for the selected product
        const dates = indices.map((idx) => data.predictions.sale_date[idx]);

        // Get unique dates (we might have multiple entries per day)
        const uniqueDates = Array.from(new Set(dates));

        // Calculate total quantity per day
        const dailyQuantities = uniqueDates.map((date) => {
          const dateIndices = indices.filter(
            (idx) => data.predictions.sale_date[idx] === date
          );
          return dateIndices.reduce(
            (sum, idx) => sum + Number(data.predictions.total_quantity[idx]),
            0
          );
        });

        setForecastData({
          dates: uniqueDates,
          quantities: dailyQuantities,
          product: selectedProduct,
        });
      } else {
        setError(data.error || "Failed to generate forecast");
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
        date: new Date(date).toLocaleDateString("en-US", {
          month: "short",
          day: "numeric",
        }),
        quantity: forecastData.quantities[index],
      }))
    : [];

  // Calculate trend if we have data
  const calculateTrend = () => {
    if (!forecastData || forecastData.quantities.length < 2) return 0;

    const firstVal = forecastData.quantities[0];
    const lastVal = forecastData.quantities[forecastData.quantities.length - 1];

    if (firstVal === 0) return 0;
    return (((lastVal - firstVal) / firstVal) * 100).toFixed(1);
  };

  const trend = calculateTrend();
  const trendingUp = Number(trend) >= 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Demand Forecast</CardTitle>
        <CardDescription>7-Day Product Demand Prediction</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <Select value={selectedProduct} onValueChange={setSelectedProduct}>
            <SelectTrigger>
              <SelectValue placeholder="Select a product" />
            </SelectTrigger>
            <SelectContent>
              {productNames.map((product) => (
                <SelectItem key={product} value={product}>
                  {product}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {!forecastData && (
            <button
              onClick={handleForecast}
              disabled={loading || !selectedProduct}
              className="mt-4 w-full px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50"
            >
              {loading ? "Generating Forecast..." : "Generate Forecast"}
            </button>
          )}

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </div>

        {forecastData ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="date"
                tickLine={false}
                tickMargin={10}
                axisLine={false}
              />
              <YAxis />
              <Tooltip />
              <Bar
                dataKey="quantity"
                fill="var(--chart-1)"
                name="Predicted Demand"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex items-center justify-center h-[300px] text-center text-muted-foreground">
            Select a product and generate a forecast to see predictions
          </div>
        )}
      </CardContent>

      {forecastData && (
        <CardFooter className="flex-col items-start gap-2 text-sm">
          <div className="flex gap-2 font-medium leading-none">
            {trendingUp ? (
              <>
                Trending up by {trend}% over the forecast period{" "}
                <TrendingUp className="h-4 w-4" />
              </>
            ) : (
              <>
                Trending down by {Math.abs(Number(trend))}% over the forecast
                period <TrendingUp className="h-4 w-4 rotate-180" />
              </>
            )}
          </div>
          <div className="leading-none text-muted-foreground">
            Showing predicted demand for {selectedProduct}
          </div>
        </CardFooter>
      )}
    </Card>
  );
}

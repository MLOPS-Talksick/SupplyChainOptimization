"use client";

import { useState, useEffect, useCallback } from "react";
import { TrendingUp } from "lucide-react";
import { AlertCircle, RefreshCw } from "lucide-react";
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
  ResponsiveContainer,
  Tooltip as ChartTooltip,
  TooltipProps,
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
import { ChartConfig, ChartContainer } from "@/components/ui/chart";
import { Loader2 } from "lucide-react";
import { GlowingEffect } from "@/components/ui/glowing-effect";
import { Button } from "@/components/ui/button";

interface ForecastData {
  dates: string[];
  quantities: number[];
  product: string;
}

// Custom tooltip component for better styling
interface CustomTooltipProps extends TooltipProps<number, string> {
  active?: boolean;
  payload?: Array<{
    value: number;
    dataKey: string;
    name?: string;
  }>;
  label?: string;
}

const CustomTooltip = ({ active, payload, label }: CustomTooltipProps) => {
  if (active && payload && payload.length) {
    return (
      <div className="rounded-lg border bg-background p-3 shadow-md">
        <p className="mb-2 font-medium">{label}</p>
        <div className="flex items-center justify-between gap-8">
          <span className="text-sm text-muted-foreground">
            Predicted Demand:
          </span>
          <span className="font-medium">{payload[0].value}</span>
        </div>
      </div>
    );
  }
  return null;
};

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
      const response = await fetch(
        `/api/proxy?endpoint=data&n=7&t=${Date.now()}`,
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            token: "backendapi1234567890",
            predictions: "True",
          },
        }
      );

      const result = await response.json();

      if (response.ok && result.records) {
        try {
          // Check if result.records is already an object or an empty object
          let parsedData;
          if (typeof result.records === "string") {
            // Parse it if it's a string
            parsedData = JSON.parse(result.records);
          } else {
            // Use it directly if it's already an object
            parsedData = result.records;
          }

          // Check if we have any data
          if (
            parsedData &&
            parsedData.product_name &&
            Object.keys(parsedData.product_name).length > 0
          ) {
            // Extract unique product names from the predictions
            const products = Object.values(parsedData.product_name);
            const uniqueProducts = Array.from(new Set(products)) as string[];
            setProductNames(uniqueProducts);
          } else {
            // Handle empty data scenario - this is a valid state, not an error
            setProductNames([]);
          }
        } catch (parseError) {
          console.error("Error parsing data:", parseError);
          setError("Error parsing data from server");
        }
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

  // Generate forecast whenever a product is selected
  useEffect(() => {
    if (selectedProduct) {
      handleForecast();
    }
  }, [selectedProduct]);

  const handleForecast = async () => {
    if (!selectedProduct) {
      setError("Please select a product");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Add cache busting parameter
      const response = await fetch(
        `/api/proxy?endpoint=data&n=7&t=${Date.now()}`,
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            token: "backendapi1234567890",
            predictions: "True",
          },
        }
      );

      const result = await response.json();

      if (response.ok && result.records) {
        try {
          // Check if result.records is already an object or an empty object
          let parsedData;
          if (typeof result.records === "string") {
            // Parse it if it's a string
            parsedData = JSON.parse(result.records);
          } else {
            // Use it directly if it's already an object
            parsedData = result.records;
          }

          // Check if we have any data for the selected product
          if (
            parsedData &&
            parsedData.product_name &&
            Object.keys(parsedData.product_name).length > 0
          ) {
            // Process data for the selected product
            const indices = Object.keys(parsedData.product_name).filter(
              (key) => parsedData.product_name[key] === selectedProduct
            );

            if (indices.length > 0) {
              // Extract dates for the selected product
              const dates = indices.map((idx) => parsedData.sale_date[idx]);

              // Get unique dates (we might have multiple entries per day)
              const uniqueDates = Array.from(new Set(dates));

              // Calculate total quantity per day
              const dailyQuantities = uniqueDates.map((date) => {
                const dateIndices = indices.filter(
                  (idx) => parsedData.sale_date[idx] === date
                );
                return dateIndices.reduce(
                  (sum, idx) => sum + Number(parsedData.total_quantity[idx]),
                  0
                );
              });

              setForecastData({
                dates: uniqueDates,
                quantities: dailyQuantities,
                product: selectedProduct,
              });
            } else {
              setError(`No data available for ${selectedProduct}`);
              setForecastData(null);
            }
          } else {
            // Handle empty data scenario
            setError("No forecast data available");
            setForecastData(null);
          }
        } catch (parseError) {
          console.error("Error parsing data:", parseError);
          setError("Error parsing data from server");
        }
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
        date: new Date(Number(date)).toLocaleDateString("en-US", {
          month: "short",
          day: "numeric",
        }),
        quantity: forecastData.quantities[index],
      }))
    : [];

  // Sort chart data by date and limit to the last 7 days
  const sortedChartData = [...chartData]
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
    .slice(-7);

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

  // Chart configuration
  const chartConfig: ChartConfig = {
    quantity: {
      label: "Predicted Demand ",
      color: "hsl(142 88% 28%)",
    },
  };

  return (
    <Card className="w-full relative">
      <GlowingEffect
        spread={40}
        glow={true}
        disabled={false}
        proximity={64}
        inactiveZone={0.01}
      />
      <div className="relative">
        <CardHeader>
          <CardTitle>Demand Forecast</CardTitle>
          <CardDescription>7-Day Product Demand Prediction</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
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

            {loading && (
              <div className="flex justify-center py-4">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            )}

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {!loading && !error && productNames.length === 0 && (
              <div className="py-4 text-center">
                <p className="text-muted-foreground mb-2">
                  No products available
                </p>
                <Button
                  onClick={fetchProductNames}
                  variant="outline"
                  className="mt-2"
                  disabled={loading}
                >
                  {loading ? (
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  ) : (
                    <RefreshCw className="h-4 w-4 mr-2" />
                  )}
                  Refresh Data
                </Button>
              </div>
            )}

            {!loading && !error && productNames.length > 0 && !forecastData && (
              <div className="flex items-center justify-center h-[300px] text-center text-muted-foreground">
                Select a product and generate a forecast to see predictions
              </div>
            )}

            {forecastData ? (
              <div className="mt-6">
                <ChartContainer
                  config={chartConfig}
                  className="aspect-auto h-[300px] w-full"
                >
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={sortedChartData}>
                      <defs>
                        <linearGradient
                          id="fillQuantity"
                          x1="0"
                          y1="0"
                          x2="0"
                          y2="1"
                        >
                          <stop
                            offset="5%"
                            stopColor="hsl(142 88% 28%)"
                            stopOpacity={0.4}
                          />
                          <stop
                            offset="95%"
                            stopColor="hsl(142 88% 28%)"
                            stopOpacity={0.1}
                          />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis
                        dataKey="date"
                        tickLine={false}
                        axisLine={false}
                        tickMargin={10}
                        tick={{ fill: "var(--foreground)" }}
                      />
                      <YAxis
                        tickLine={false}
                        axisLine={false}
                        tickMargin={8}
                        tick={{ fill: "var(--foreground)" }}
                      />
                      <ChartTooltip
                        cursor={false}
                        content={<CustomTooltip />}
                      />
                      <Bar
                        dataKey="quantity"
                        fill="url(#fillQuantity)"
                        stroke="hsl(142 88% 28%)"
                        strokeWidth={2}
                        name="Predicted Demand"
                        radius={[4, 4, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartContainer>
              </div>
            ) : null}
          </div>
        </CardContent>

        {forecastData && (
          <CardFooter className="flex flex-col items-start gap-2 text-sm border-t pt-4">
            <div className="flex items-center gap-2 font-medium">
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
            <div className="text-muted-foreground">
              Showing predicted demand for {selectedProduct}
            </div>
          </CardFooter>
        )}
      </div>
    </Card>
  );
}

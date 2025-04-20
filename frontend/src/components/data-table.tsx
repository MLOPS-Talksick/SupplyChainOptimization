"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { AlertCircle, RefreshCw, Loader2, X } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
// Add chart imports
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";

interface SalesRecord {
  sale_date: Date;
  product_name: string;
  total_quantity: number;
}

interface BackendResponse {
  records: string;
  count: number;
}

// API configuration
const API_CONFIG = {
  DEFAULT_RECORDS: "50",
};

// Store the user's preference in localStorage
const savePreferences = (records: string, custom: string) => {
  if (typeof window !== "undefined") {
    localStorage.setItem("recordsToFetch", records);
    localStorage.setItem("customCount", custom);
  }
};

// Load the user's preferences from localStorage
const loadPreferences = () => {
  if (typeof window !== "undefined") {
    return {
      recordsToFetch:
        localStorage.getItem("recordsToFetch") || API_CONFIG.DEFAULT_RECORDS,
      customCount: localStorage.getItem("customCount") || "",
    };
  }
  return {
    recordsToFetch: API_CONFIG.DEFAULT_RECORDS,
    customCount: "",
  };
};

export default function DataTable() {
  // Load saved preferences
  const prefs = loadPreferences();

  const [data, setData] = useState<SalesRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recordsToFetch, setRecordsToFetch] = useState(prefs.recordsToFetch);
  const [customCount, setCustomCount] = useState(prefs.customCount);
  const [selectedProduct, setSelectedProduct] = useState<string>("");
  const [productOptions, setProductOptions] = useState<string[]>([]);

  // Add a ref to store the current values
  const currentValuesRef = useRef({
    recordsToFetch: prefs.recordsToFetch,
    customCount: prefs.customCount,
  });

  // Update the ref when the values change
  useEffect(() => {
    currentValuesRef.current = {
      recordsToFetch,
      customCount,
    };
  }, [recordsToFetch, customCount]);

  // Memoize the loadData function to prevent unnecessary recreations
  const loadData = useCallback(async () => {
    setLoading(true);
    setRefreshing(true);
    setError(null);

    // Use the current values from the ref
    const { recordsToFetch, customCount } = currentValuesRef.current;
    const countToUse = customCount || recordsToFetch;

    try {
      const response = await fetch(`/api/proxy?endpoint=data&n=${countToUse}`, {
        method: "GET",
        headers: {
          token: "backendapi1234567890",
        },
      });
      const result: BackendResponse = await response.json();

      if (result && result.records) {
        try {
          // The backend returns a JSON string, we need to parse it
          const parsedData = JSON.parse(result.records);
          const totalCount = result.count || 0;

          // Convert the parsed data to an array of records
          const records = Object.keys(parsedData.sale_date).map((index) => ({
            sale_date: new Date(parseInt(parsedData.sale_date[index])), // Convert timestamp to Date
            product_name: parsedData.product_name[index],
            total_quantity: parsedData.total_quantity[index],
          }));

          console.log(
            `Received ${records.length} records (total available: ${totalCount})`
          );
          setData(records);

          // Extract unique product names for the dropdown
          const uniqueProducts = Array.from(
            new Set(records.map((record) => record.product_name))
          ).sort();

          setProductOptions(uniqueProducts);

          // Save preferences after successful data load
          savePreferences(recordsToFetch, customCount);
        } catch (parseError) {
          console.error("Error parsing data:", parseError);
          setError("Error parsing data from server");
          setData([]);
        }
      } else {
        setData([]);
      }
    } catch (err) {
      console.error("Data fetch error:", err);
      setError(err instanceof Error ? err.message : "Failed to fetch data");
      setData([]);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []); // No dependencies needed since we're using the ref

  // Filter data based on selected product
  const filteredData = useMemo(() => {
    if (!selectedProduct) return data;
    return data.filter((record) => record.product_name === selectedProduct);
  }, [data, selectedProduct]);

  // Handle product selection
  const handleProductChange = useCallback((value: string) => {
    setSelectedProduct(value);
  }, []);

  // Clear product selection
  const clearProductSelection = useCallback(() => {
    setSelectedProduct("");
  }, []);

  useEffect(() => {
    const fetchDataWithRetry = async (retries = 3, delay = 1000) => {
      for (let i = 0; i < retries; i++) {
        try {
          await loadData();
          return; // Success, exit the retry loop
        } catch (error) {
          console.error(`Attempt ${i + 1} failed:`, error);
          if (i < retries - 1) {
            console.log(`Retrying in ${delay}ms...`);
            await new Promise((resolve) => setTimeout(resolve, delay));
            // Increase delay for next retry (exponential backoff)
            delay *= 2;
          }
        }
      }
    };

    // Initial data load
    fetchDataWithRetry();

    // Set up auto-refresh every 30 seconds
    const intervalId = setInterval(() => {
      fetchDataWithRetry(2, 1000); // Less retries for auto-refresh
    }, 30000);

    return () => {
      clearInterval(intervalId);
    };
  }, []); // Remove loadData dependency to prevent automatic refreshing

  const handleRefresh = useCallback(() => {
    loadData();
  }, [loadData]);

  const handleRecordCountChange = useCallback((value: string) => {
    setRecordsToFetch(value);
    setCustomCount(""); // Clear custom count when selecting from dropdown
    // Don't save preferences or load data here - wait for explicit refresh
  }, []);

  const handleCustomCountChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = e.target.value;
      // Only allow numeric input
      if (value === "" || /^\d+$/.test(value)) {
        setCustomCount(value);
        // Clear dropdown selection when entering custom count
        if (value) {
          setRecordsToFetch("custom");
        } else {
          setRecordsToFetch(API_CONFIG.DEFAULT_RECORDS);
        }
      }
    },
    []
  );

  // Add a key down handler specifically for Enter key
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter") {
        e.preventDefault();
        loadData();
      }
    },
    [loadData]
  );

  // Add a form submission handler
  const handleFormSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault(); // Prevent default form submission
      loadData(); // Refresh data
    },
    [loadData]
  );

  // Prepare chart data - memoized to improve performance
  const chartData = useMemo(() => {
    if (!selectedProduct || filteredData.length === 0) return [];

    // Sort by date (ascending)
    return [...filteredData]
      .sort((a, b) => a.sale_date.getTime() - b.sale_date.getTime())
      .map((record) => ({
        date: record.sale_date.toISOString().split("T")[0], // Format as YYYY-MM-DD
        quantity: record.total_quantity,
      }));
  }, [filteredData, selectedProduct]);

  // Chart configuration
  const chartConfig: ChartConfig = {
    quantity: {
      label: "Quantity",
      color: "hsl(142 88% 28%)",
    },
  };

  if (loading && data.length === 0) {
    return <div className="text-center py-8">Loading data...</div>;
  }

  if (error && data.length === 0) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-3">
          <Select
            value={recordsToFetch}
            onValueChange={handleRecordCountChange}
          >
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Number of records" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="5">Last 5 records</SelectItem>
              <SelectItem value="10">Last 10 records</SelectItem>
              <SelectItem value="50">Last 50 records</SelectItem>
              <SelectItem value="100">Last 100 records</SelectItem>
              <SelectItem value="1000">Last 1000 records</SelectItem>
              <SelectItem value="custom">Custom count</SelectItem>
            </SelectContent>
          </Select>

          {(recordsToFetch === "custom" || customCount) && (
            <form onSubmit={handleFormSubmit} className="m-0 p-0">
              <Input
                type="text"
                placeholder="Enter record count"
                className="w-[150px]"
                value={customCount}
                onChange={handleCustomCountChange}
                onKeyDown={handleKeyDown}
                aria-label="Custom number of records to fetch"
              />
              <button type="submit" hidden></button>
            </form>
          )}

          <Button
            onClick={handleRefresh}
            size="icon"
            variant="outline"
            disabled={refreshing}
          >
            {refreshing ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4" />
            )}
          </Button>
        </div>

        <div className="ml-auto relative">
          <div className="flex items-center gap-2">
            <Select value={selectedProduct} onValueChange={handleProductChange}>
              <SelectTrigger className="w-[250px]">
                <SelectValue placeholder="Select a product" />
              </SelectTrigger>
              <SelectContent className="max-h-[300px]">
                {productOptions.map((product) => (
                  <SelectItem key={product} value={product}>
                    {product}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedProduct && (
              <Button
                onClick={clearProductSelection}
                size="icon"
                variant="ghost"
                className="h-9 w-9"
                aria-label="Clear product selection"
              >
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </div>

      {filteredData.length === 0 && !loading ? (
        <div className="text-center py-8">No data available</div>
      ) : (
        <>
          <div className="flex items-center justify-between w-full">
            <div className="text-sm text-muted-foreground flex items-center gap-2">
              <span className="flex items-center gap-1">
                {selectedProduct && filteredData.length !== data.length ? (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary font-medium mr-1">
                    Filtered: &ldquo;{selectedProduct}&rdquo;
                  </span>
                ) : null}
                <span>Showing:</span>
                <span className="font-bold text-foreground">
                  {filteredData.length}
                </span>
                {filteredData.length !== data.length && (
                  <>
                    <span>of</span>
                    <span className="font-bold text-foreground">
                      {data.length}
                    </span>
                  </>
                )}
                <span>record{filteredData.length !== 1 ? "s" : ""}</span>
              </span>

              {refreshing && (
                <span className="ml-2 flex items-center text-xs text-muted-foreground">
                  <Loader2 className="h-3 w-3 animate-spin mr-1" />
                  <span>Refreshing...</span>
                </span>
              )}
            </div>
          </div>

          <div className="rounded-md border">
            <div className="h-[50vh] max-h-[500px] min-h-[300px] overflow-auto">
              <Table>
                <TableHeader className="sticky top-0 bg-background z-10 shadow-sm">
                  <TableRow>
                    <TableHead className="sticky top-0 bg-background z-10">
                      Date
                    </TableHead>
                    <TableHead className="sticky top-0 bg-background z-10">
                      Product Name
                    </TableHead>
                    <TableHead className="sticky top-0 bg-background z-10">
                      Quantity
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {loading && data.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={3} className="h-24 text-center">
                        <Loader2 className="h-6 w-6 animate-spin mx-auto" />
                        <span className="mt-2 block">Loading records...</span>
                      </TableCell>
                    </TableRow>
                  ) : (
                    filteredData.map((record, index) => (
                      <TableRow key={index}>
                        <TableCell>
                          {record.sale_date.toLocaleDateString()}
                        </TableCell>
                        <TableCell>{record.product_name}</TableCell>
                        <TableCell>{record.total_quantity}</TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </div>
          </div>

          {/* Chart visualization */}
          {selectedProduct && chartData.length > 0 && (
            <Card className="@container/card mt-8">
              <CardHeader className="relative">
                <CardTitle>
                  Quantity Trend for &ldquo;{selectedProduct}&rdquo;
                </CardTitle>
                <CardDescription>
                  <span className="@[540px]/card:block hidden">
                    Historical data for {filteredData.length} records
                  </span>
                  <span className="@[540px]/card:hidden">
                    {filteredData.length} records
                  </span>
                </CardDescription>
              </CardHeader>
              <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
                <ChartContainer
                  config={chartConfig}
                  className="aspect-auto h-[250px] w-full"
                >
                  <AreaChart data={chartData}>
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
                    <CartesianGrid vertical={false} />
                    <XAxis
                      dataKey="date"
                      tickLine={false}
                      axisLine={false}
                      tickMargin={8}
                      minTickGap={32}
                      tick={{ fill: "var(--foreground)" }}
                      tickFormatter={(value) => {
                        const date = new Date(value);
                        return date.toLocaleDateString("en-US", {
                          month: "short",
                          day: "numeric",
                        });
                      }}
                    />
                    <YAxis
                      tickLine={false}
                      axisLine={false}
                      tickMargin={8}
                      tick={{ fill: "var(--foreground)" }}
                    />
                    <ChartTooltip
                      cursor={false}
                      content={
                        <ChartTooltipContent
                          labelFormatter={(value) => {
                            return new Date(value).toLocaleDateString("en-US", {
                              month: "short",
                              day: "numeric",
                            });
                          }}
                          indicator="dot"
                        />
                      }
                    />
                    <Area
                      dataKey="quantity"
                      type="natural"
                      fill="url(#fillQuantity)"
                      stroke="hsl(142 88% 28%)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ChartContainer>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}

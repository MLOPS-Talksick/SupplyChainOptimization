"use client";

import { useState, useEffect, useCallback } from "react";
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
import { fetchData } from "@/lib/api";
import { API_CONFIG } from "@/lib/config";

interface SalesRecord {
  sale_date: Date;
  product_name: string;
  total_quantity: number;
}

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
  const [searchTerm, setSearchTerm] = useState("");
  const [activeSearchTerm, setActiveSearchTerm] = useState("");

  const loadData = async () => {
    setLoading(true);
    setRefreshing(true);
    setError(null);

    // If a custom count is entered, use it instead of the dropdown selection
    const countToUse = customCount || recordsToFetch;

    try {
      const result = await fetchData({
        n: countToUse,
      });

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
  };

  // Filter data based on search term - memoized to improve performance
  const filteredData = useCallback(() => {
    if (!activeSearchTerm) return data;
    const searchLower = activeSearchTerm.toLowerCase();
    return data.filter((record) =>
      record.product_name.toLowerCase().includes(searchLower)
    );
  }, [data, activeSearchTerm])();

  // Simple search input change handler - only updates the input value
  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  };

  // Clear search handler
  const clearSearch = () => {
    setSearchTerm("");
    setActiveSearchTerm("");
  };

  // Search form submission handler - applies the filter
  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setActiveSearchTerm(searchTerm);
  };

  // Search input key down handler - applies filter on Enter
  const handleSearchKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      setActiveSearchTerm(searchTerm);
    }
  };

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

    fetchDataWithRetry();

    // Set up auto-refresh every 30 seconds
    const intervalId = setInterval(() => {
      fetchDataWithRetry(2, 1000); // Less retries for auto-refresh
    }, 30000);

    return () => clearInterval(intervalId);
  }, []);

  const handleRefresh = () => {
    loadData();
  };

  const handleRecordCountChange = (value: string) => {
    setRecordsToFetch(value);
    setCustomCount(""); // Clear custom count when selecting from dropdown
    savePreferences(value, "");
  };

  const handleCustomCountChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    // Only allow numeric input
    if (value === "" || /^\d+$/.test(value)) {
      setCustomCount(value);
      // Clear dropdown selection when entering custom count
      if (value) {
        setRecordsToFetch("custom");
        savePreferences("custom", value);
      } else {
        setRecordsToFetch(API_CONFIG.DEFAULT_RECORDS);
        savePreferences(API_CONFIG.DEFAULT_RECORDS, "");
      }
    }
  };

  // Add a key down handler specifically for Enter key
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      loadData();
    }
  };

  // Add a form submission handler
  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault(); // Prevent default form submission
    loadData(); // Refresh data
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
          <form onSubmit={handleSearchSubmit} className="m-0 p-0">
            <Input
              placeholder="Search products..."
              className={`w-[250px] ${
                activeSearchTerm
                  ? "border-primary"
                  : searchTerm
                  ? "border-muted"
                  : ""
              } pr-9`}
              value={searchTerm}
              onChange={handleSearchChange}
              onKeyDown={handleSearchKeyDown}
            />
            <button type="submit" hidden></button>
          </form>
          {searchTerm && (
            <button
              onClick={clearSearch}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground focus:outline-none"
              aria-label="Clear search"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {filteredData.length === 0 && !loading ? (
        <div className="text-center py-8">No data available</div>
      ) : (
        <>
          <div className="flex items-center justify-between w-full">
            <div className="text-sm text-muted-foreground flex items-center gap-2">
              <span className="flex items-center gap-1">
                {activeSearchTerm && filteredData.length !== data.length ? (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary font-medium mr-1">
                    Filtered: &ldquo;{activeSearchTerm}&rdquo;
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
        </>
      )}
    </div>
  );
}

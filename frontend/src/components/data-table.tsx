"use client";

import { useState, useEffect } from "react";
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
import { AlertCircle } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface SalesRecord {
  id: number;
  product_id: string;
  sale_date: string;
  quantity: number;
  price: number;
}

export default function DataTable() {
  const [data, setData] = useState<SalesRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [recordsToFetch, setRecordsToFetch] = useState("50");

  const fetchData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `http://localhost:3000/data?n=${recordsToFetch}`
      );
      const result = await response.json();

      if (response.ok) {
        setData(result.data || []);
      } else {
        setError(result.error || "Failed to fetch data");
      }
    } catch {
      setError("Error connecting to the server. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleRefresh = () => {
    fetchData();
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
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Select value={recordsToFetch} onValueChange={setRecordsToFetch}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Number of records" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="10">Last 10 records</SelectItem>
              <SelectItem value="50">Last 50 records</SelectItem>
              <SelectItem value="100">Last 100 records</SelectItem>
              <SelectItem value="500">Last 500 records</SelectItem>
            </SelectContent>
          </Select>
          <Button onClick={handleRefresh}>Refresh</Button>
        </div>
        <div>
          <Input placeholder="Search products..." className="w-[250px]" />
        </div>
      </div>

      {data.length === 0 ? (
        <div className="text-center py-8">No data available</div>
      ) : (
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ID</TableHead>
                <TableHead>Product ID</TableHead>
                <TableHead>Date</TableHead>
                <TableHead>Quantity</TableHead>
                <TableHead>Price</TableHead>
                <TableHead>Total</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.map((record) => (
                <TableRow key={record.id}>
                  <TableCell>{record.id}</TableCell>
                  <TableCell>{record.product_id}</TableCell>
                  <TableCell>
                    {new Date(record.sale_date).toLocaleDateString()}
                  </TableCell>
                  <TableCell>{record.quantity}</TableCell>
                  <TableCell>${record.price.toFixed(2)}</TableCell>
                  <TableCell>
                    ${(record.quantity * record.price).toFixed(2)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}

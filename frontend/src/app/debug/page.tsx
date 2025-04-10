"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { API_CONFIG } from "@/lib/config";

interface ApiStatusResponse {
  status: string;
  timestamp: string;
  environment: string;
  backend: {
    url: string;
    status: string;
    statusCode?: number;
  };
  error?: string;
}

export default function DebugPage() {
  const [apiStatus, setApiStatus] = useState<ApiStatusResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checkApiStatus = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/health");
      const data = await response.json();
      setApiStatus(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Error checking API status"
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkApiStatus();
  }, []);

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">API Debug Information</h1>

      <Card className="mb-4">
        <CardHeader>
          <CardTitle>API Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="bg-secondary p-4 rounded overflow-auto">
            {JSON.stringify(API_CONFIG, null, 2)}
          </pre>
        </CardContent>
      </Card>

      <Card className="mb-4">
        <CardHeader>
          <CardTitle>API Health Check</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <p>Loading API status...</p>
          ) : error ? (
            <div className="text-destructive">{error}</div>
          ) : (
            <pre className="bg-secondary p-4 rounded overflow-auto">
              {JSON.stringify(apiStatus, null, 2)}
            </pre>
          )}
          <Button onClick={checkApiStatus} disabled={loading} className="mt-4">
            Refresh Status
          </Button>
        </CardContent>
      </Card>

      <div className="mt-4">
        <p className="text-sm text-muted-foreground">
          Use this information to diagnose connectivity issues between the
          frontend and backend API.
        </p>
        <Button
          onClick={() => (window.location.href = "/")}
          variant="outline"
          className="mt-2"
        >
          Return to Dashboard
        </Button>
      </div>
    </div>
  );
}

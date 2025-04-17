"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle, CheckCircle2, UploadCloud } from "lucide-react";

export default function UploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{
    success: boolean;
    message: string;
  } | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setUploadStatus(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!file) {
      setUploadStatus({
        success: false,
        message: "Please select a file to upload",
      });
      return;
    }

    // Check if file is an Excel file
    const fileExt = file.name.split(".").pop()?.toLowerCase();
    if (fileExt !== "xls" && fileExt !== "xlsx") {
      setUploadStatus({
        success: false,
        message: "Only Excel files (.xls or .xlsx) are allowed",
      });
      return;
    }

    setIsUploading(true);
    setUploadStatus(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:3000/upload", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        setUploadStatus({
          success: true,
          message: "File uploaded successfully",
        });
        setFile(null);
        // Clear the file input
        const fileInput = document.getElementById(
          "file-upload"
        ) as HTMLInputElement;
        if (fileInput) fileInput.value = "";
      } else {
        setUploadStatus({
          success: false,
          message: result.error || "Failed to upload file",
        });
      }
    } catch {
      setUploadStatus({
        success: false,
        message: "Error uploading file. Please try again.",
      });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="grid w-full max-w-sm items-center gap-1.5">
        <label
          htmlFor="file-upload"
          className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
        >
          Upload Excel File
        </label>
        <Input
          id="file-upload"
          type="file"
          accept=".xls,.xlsx"
          onChange={handleFileChange}
          className="cursor-pointer"
        />
        <p className="text-sm text-muted-foreground">
          Upload .xls or .xlsx files containing sales data
        </p>
      </div>

      {uploadStatus && (
        <Alert variant={uploadStatus.success ? "default" : "destructive"}>
          {uploadStatus.success ? (
            <CheckCircle2 className="h-4 w-4" />
          ) : (
            <AlertCircle className="h-4 w-4" />
          )}
          <AlertTitle>{uploadStatus.success ? "Success" : "Error"}</AlertTitle>
          <AlertDescription>{uploadStatus.message}</AlertDescription>
        </Alert>
      )}

      <Button type="submit" disabled={isUploading}>
        {isUploading ? (
          <>Uploading...</>
        ) : (
          <>
            <UploadCloud className="mr-2 h-4 w-4" />
            Upload File
          </>
        )}
      </Button>
    </form>
  );
}

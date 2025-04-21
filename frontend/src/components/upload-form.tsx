"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  AlertCircle,
  CheckCircle2,
  Trash2,
  Edit,
  UploadCloud,
  X,
  CheckCheck,
} from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { useDropzone } from "react-dropzone";

interface NewProduct {
  name: string;
  action: "keep" | "rename" | "deny";
  newName?: string;
}

export default function UploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [validationComplete, setValidationComplete] = useState(false);
  const [newProducts, setNewProducts] = useState<NewProduct[]>([]);
  const [originalProducts, setOriginalProducts] = useState<NewProduct[]>([]);
  const [uploadStatus, setUploadStatus] = useState<{
    success: boolean;
    message: string;
  } | null>(null);
  const [editingProduct, setEditingProduct] = useState<{
    name: string;
    newName: string;
  } | null>(null);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setFile(acceptedFiles[0]);
        setUploadStatus(null);
        setValidationComplete(false);
        setNewProducts([]);
      }
    },
    accept: {
      "application/vnd.ms-excel": [".xls"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [
        ".xlsx",
      ],
    },
    maxFiles: 1,
    disabled: validationComplete,
  });

  const validateFile = async () => {
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

    setIsValidating(true);
    setUploadStatus(null);

    // Create form data for the request
    const formData = new FormData();

    // Explicitly create a new File object from the selected file
    // This ensures the file is properly serialized for FormData
    const fileBlob = new Blob([await file.arrayBuffer()], { type: file.type });
    const newFile = new File([fileBlob], file.name, { type: file.type });

    // Add the file to the form data
    formData.append("file", newFile);

    console.log(
      `Validating Excel file: ${file.name} (${file.size} bytes, type: ${file.type})`
    );

    // Debug logging
    for (const [key, value] of formData.entries()) {
      if (value instanceof File) {
        console.log(
          `FormData contains file: ${key} = ${value.name} (${value.size} bytes)`
        );
      } else {
        console.log(`FormData contains: ${key} = ${value}`);
      }
    }

    try {
      // Use the proxy endpoint with a cache-busting parameter
      const response = await fetch(
        `/api/proxy?endpoint=validate_excel&t=${Date.now()}`,
        {
          method: "POST",
          headers: {
            token: "backendapi1234567890",
            // Note: Do NOT set Content-Type for FormData - browser will set it with boundary
          },
          body: formData,
        }
      );

      // Handle the response
      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Validation error (${response.status}): ${errorText}`);
        throw new Error(
          `Server error (${response.status}): ${response.statusText}`
        );
      }

      const result = await response.json();

      if (result.new_products) {
        // Transform new products into our internal format
        const productList: NewProduct[] = result.new_products.map(
          (name: string) => ({
            name,
            action: "keep", // Default action is to keep
          })
        );

        setNewProducts(productList);
        // Save the original products for reset functionality
        setOriginalProducts([...productList]);
        setValidationComplete(true);

        if (productList.length === 0) {
          // If no new products, proceed directly to upload
          await handleUpload();
        }
      } else {
        setUploadStatus({
          success: false,
          message:
            result.error ||
            "Failed to validate file. No new products returned.",
        });
      }
    } catch (error) {
      console.error("Validation error:", error);

      let errorMessage = "Error validating file. Please try again.";

      if (error instanceof Error) {
        errorMessage = error.message;
      }

      setUploadStatus({
        success: false,
        message: errorMessage,
      });
    } finally {
      setIsValidating(false);
    }
  };

  const handleProductAction = (
    index: number,
    action: "keep" | "rename" | "deny"
  ) => {
    const updatedProducts = [...newProducts];
    updatedProducts[index].action = action;

    // If switching from rename to another action, clear the new name
    if (action !== "rename" && updatedProducts[index].newName) {
      delete updatedProducts[index].newName;
    }

    setNewProducts(updatedProducts);
  };

  const openRenameDialog = (name: string, currentNewName: string = "") => {
    setEditingProduct({ name, newName: currentNewName });
  };

  const saveNewName = () => {
    if (!editingProduct) return;

    const updatedProducts = newProducts.map((product) => {
      if (product.name === editingProduct.name) {
        return {
          ...product,
          action: "rename" as const,
          newName: editingProduct.newName,
        };
      }
      return product;
    });

    setNewProducts(updatedProducts);
    setEditingProduct(null);
  };

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    setUploadStatus(null);

    // Prepare data for upload
    const formData = new FormData();

    // Explicitly create a new File object from the selected file
    // This ensures the file is properly serialized for FormData
    const fileBlob = new Blob([await file.arrayBuffer()], { type: file.type });
    const newFile = new File([fileBlob], file.name, { type: file.type });

    // Add the file to the form data
    formData.append("file", newFile);

    // Create deny list (comma-separated)
    const denyList = newProducts
      .filter((p) => p.action === "deny")
      .map((p) => p.name)
      .join(",");

    // Create rename dictionary as JSON string
    const renameDict = JSON.stringify(
      newProducts
        .filter((p) => p.action === "rename" && p.newName)
        .reduce(
          (acc, curr) => ({
            ...acc,
            [curr.name]: curr.newName,
          }),
          {}
        )
    );

    if (denyList) {
      formData.append("deny_list", denyList);
      console.log(`Deny list: ${denyList}`);
    }

    if (renameDict && renameDict !== "{}") {
      formData.append("rename_dict", renameDict);
      console.log(`Rename dict: ${renameDict}`);
    }

    console.log(
      `Uploading Excel file: ${file.name} (${file.size} bytes, type: ${file.type})`
    );

    // Debug logging
    for (const [key, value] of formData.entries()) {
      if (value instanceof File) {
        console.log(
          `FormData contains file: ${key} = ${value.name} (${value.size} bytes)`
        );
      } else {
        console.log(`FormData contains: ${key} = ${value}`);
      }
    }

    try {
      // Use the proxy endpoint with a cache-busting parameter
      const response = await fetch(
        `/api/proxy?endpoint=upload&t=${Date.now()}`,
        {
          method: "POST",
          headers: {
            token: "backendapi1234567890",
            // Note: Do NOT set Content-Type for FormData - browser will set it with boundary
          },
          body: formData,
        }
      );

      // Handle the response
      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Upload error (${response.status}): ${errorText}`);
        throw new Error(
          `Server error (${response.status}): ${response.statusText}`
        );
      }

      const result = await response.json();

      if (result.error) {
        setUploadStatus({
          success: false,
          message: result.error,
        });
        return;
      }

      setUploadStatus({
        success: true,
        message: "File uploaded successfully",
      });
      setFile(null);
      setValidationComplete(false);
      setNewProducts([]);
    } catch (error) {
      console.error("Upload error:", error);

      let errorMessage = "Error uploading file. Please try again.";

      if (error instanceof Error) {
        errorMessage = error.message;
      }

      setUploadStatus({
        success: false,
        message: errorMessage,
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validationComplete) {
      await validateFile();
    } else {
      await handleUpload();
    }
  };

  const resetForm = () => {
    setFile(null);
    setUploadStatus(null);
    setValidationComplete(false);
    setNewProducts([]);
    setOriginalProducts([]);
  };

  const resetValidation = () => {
    // Reset product actions to their original state (all "keep")
    const resetProducts = originalProducts.map((product) => ({
      name: product.name,
      action: "keep" as const,
    }));
    setNewProducts(resetProducts);
  };

  const getProductCounts = () => {
    const keep = newProducts.filter((p) => p.action === "keep").length;
    const rename = newProducts.filter((p) => p.action === "rename").length;
    const deny = newProducts.filter((p) => p.action === "deny").length;

    return { keep, rename, deny };
  };

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid w-full items-center gap-1.5">
          <label className="text-sm font-medium leading-none">
            Upload Excel File
          </label>

          {validationComplete ? (
            <div className="flex items-center gap-2 mt-2">
              <div className="flex-1 p-3 border rounded-md">
                <div className="flex items-center">
                  <CheckCircle2 className="h-5 w-5 text-primary mr-2" />
                  <span className="font-medium">{file?.name}</span>
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  {file?.size ? (file.size / (1024 * 1024)).toFixed(2) : 0} MB
                </p>
              </div>
              <Button size="sm" variant="outline" onClick={resetForm}>
                <X className="h-4 w-4" />
              </Button>
            </div>
          ) : file ? (
            <div className="mt-2">
              <div className="flex items-center justify-between p-3 border border-primary/20 bg-primary/5 rounded-md">
                <div className="flex items-center">
                  <UploadCloud className="h-5 w-5 text-primary mr-2" />
                  <div>
                    <span className="font-medium">{file.name}</span>
                    <p className="text-xs text-muted-foreground">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB · Click
                      Validate to proceed
                    </p>
                  </div>
                </div>
                <Button size="sm" variant="ghost" onClick={resetForm}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <div
                {...getRootProps()}
                className="mt-2 text-center text-sm text-primary cursor-pointer hover:underline"
              >
                <input {...getInputProps()} />
                Select a different file
              </div>
            </div>
          ) : (
            <div
              {...getRootProps()}
              className={cn(
                "border-2 border-dashed rounded-md p-8 transition-colors cursor-pointer hover:border-primary/50",
                isDragActive
                  ? "border-primary bg-primary/5"
                  : "border-muted-foreground/25"
              )}
            >
              <input {...getInputProps()} />
              <div className="flex flex-col items-center justify-center gap-1 text-center">
                <UploadCloud className="h-10 w-10 text-muted-foreground mb-2" />
                <p className="text-sm font-medium">
                  {isDragActive
                    ? "Drop the file here"
                    : "Drag and drop your Excel file here"}
                </p>
                <p className="text-xs text-muted-foreground">
                  or click to select file (.xls or .xlsx)
                </p>
              </div>
            </div>
          )}

          {uploadStatus && (
            <Alert
              variant={uploadStatus.success ? "default" : "destructive"}
              className="mt-4"
            >
              {uploadStatus.success ? (
                <CheckCircle2 className="h-4 w-4" />
              ) : (
                <AlertCircle className="h-4 w-4" />
              )}
              <AlertTitle>
                {uploadStatus.success ? "Success" : "Error"}
              </AlertTitle>
              <AlertDescription className="flex flex-col gap-2">
                <p>{uploadStatus.message}</p>

                {!uploadStatus.success && (
                  <div className="mt-1">
                    <Button
                      size="sm"
                      variant="outline"
                      className="w-fit"
                      onClick={validateFile}
                    >
                      Try Again
                    </Button>

                    {uploadStatus.message.includes("500") ||
                    uploadStatus.message.includes("Server error") ? (
                      <div className="text-xs mt-2 space-y-1">
                        <p>Troubleshooting suggestions:</p>
                        <ul className="list-disc pl-4">
                          <li>Check that your Excel file is not corrupted</li>
                          <li>Try with a smaller file if possible</li>
                          <li>
                            Make sure the Excel file follows the expected format
                          </li>
                          <li>
                            The server might be temporarily unavailable, try
                            again later
                          </li>
                        </ul>
                      </div>
                    ) : null}
                  </div>
                )}
              </AlertDescription>
            </Alert>
          )}

          <Button
            type="submit"
            disabled={
              isUploading || isValidating || (!file && !validationComplete)
            }
            className="mt-4"
          >
            {isValidating ? (
              <>Validating...</>
            ) : isUploading ? (
              <>Uploading...</>
            ) : !validationComplete ? (
              <>
                <UploadCloud className="mr-2 h-4 w-4" />
                Validate File
              </>
            ) : (
              <>
                <CheckCheck className="mr-2 h-4 w-4" />
                Upload File
              </>
            )}
          </Button>
        </div>
      </form>

      {/* Product validation section */}
      {validationComplete && newProducts.length > 0 && (
        <Card className="p-6">
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium mb-2">
                New Products Detected
              </h3>
              <p className="text-sm text-muted-foreground">
                These products are not in the database. You can keep them as is,
                rename them, or exclude them from the upload.
              </p>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex flex-wrap gap-2">
                <Badge
                  variant="outline"
                  className="dark:bg-green-950 bg-green-50 text-green-700 dark:text-green-400 border-green-200 dark:border-green-800"
                >
                  Keep: {getProductCounts().keep}
                </Badge>
                <Badge
                  variant="outline"
                  className="dark:bg-blue-950 bg-blue-50 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-800"
                >
                  Rename: {getProductCounts().rename}
                </Badge>
                <Badge
                  variant="outline"
                  className="dark:bg-red-950 bg-red-50 text-red-700 dark:text-red-400 border-red-200 dark:border-red-800"
                >
                  Exclude: {getProductCounts().deny}
                </Badge>
              </div>

              <Button variant="outline" size="sm" onClick={resetValidation}>
                <X className="mr-2 h-4 w-4" />
                Reset
              </Button>
            </div>

            <Separator />

            <div className="max-h-[400px] overflow-y-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Product Name</TableHead>
                    <TableHead>Action</TableHead>
                    <TableHead className="w-[150px]">New Name</TableHead>
                    <TableHead className="text-right">Options</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {newProducts.map((product, index) => (
                    <TableRow key={product.name}>
                      <TableCell className="font-medium">
                        {product.name}
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant={
                            product.action === "keep"
                              ? "default"
                              : product.action === "rename"
                              ? "secondary"
                              : "destructive"
                          }
                        >
                          {product.action === "keep"
                            ? "Keep"
                            : product.action === "rename"
                            ? "Rename"
                            : "Exclude"}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        {product.action === "rename" && product.newName
                          ? product.newName
                          : "—"}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex justify-end gap-2">
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() =>
                              handleProductAction(
                                index,
                                product.action === "deny" ? "keep" : "deny"
                              )
                            }
                            className={cn(
                              product.action === "keep"
                                ? "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-400 dark:border-green-800 hover:bg-green-200 dark:hover:bg-green-900/60"
                                : product.action === "deny"
                                ? "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-400 dark:border-red-800 hover:bg-red-200 dark:hover:bg-red-900/60"
                                : "hover:bg-red-100 hover:text-red-700 dark:hover:bg-red-900/30 dark:hover:text-red-400"
                            )}
                          >
                            {product.action === "deny" ? (
                              <Trash2 className="h-4 w-4" />
                            ) : (
                              <CheckCircle2 className="h-4 w-4" />
                            )}
                            <span className="sr-only">
                              {product.action === "deny" ? "Exclude" : "Keep"}
                            </span>
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => {
                              if (product.action !== "rename") {
                                openRenameDialog(product.name);
                              } else {
                                openRenameDialog(product.name, product.newName);
                              }
                            }}
                            className={cn(
                              "hover:bg-blue-100 hover:text-blue-700 dark:hover:bg-blue-900/30 dark:hover:text-blue-400",
                              product.action === "rename"
                                ? "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-400 dark:border-blue-800"
                                : ""
                            )}
                          >
                            <Edit className="h-4 w-4" />
                            <span className="sr-only">Rename</span>
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
        </Card>
      )}

      {/* Rename Dialog */}
      <Dialog
        open={!!editingProduct}
        onOpenChange={(open) => !open && setEditingProduct(null)}
      >
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Rename Product</DialogTitle>
            <DialogDescription>
              Enter a new name for &ldquo;{editingProduct?.name}&rdquo;
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <Input
              placeholder="New product name"
              value={editingProduct?.newName || ""}
              onChange={(e) =>
                setEditingProduct((prev) =>
                  prev ? { ...prev, newName: e.target.value } : null
                )
              }
              className="col-span-3"
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditingProduct(null)}>
              Cancel
            </Button>
            <Button onClick={saveNewName} disabled={!editingProduct?.newName}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

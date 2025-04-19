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

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/validate_excel", {
        method: "POST",
        headers: {
          token: "backendapi1234567890",
        },
        body: formData,
      });

      const result = await response.json();

      if (response.ok && result.new_products) {
        // Transform new products into our internal format
        const productList: NewProduct[] = result.new_products.map(
          (name: string) => ({
            name,
            action: "keep", // Default action is to keep
          })
        );

        setNewProducts(productList);
        setValidationComplete(true);

        if (productList.length === 0) {
          // If no new products, proceed directly to upload
          await handleUpload();
        }
      } else {
        setUploadStatus({
          success: false,
          message: result.error || "Failed to validate file",
        });
      }
    } catch (error) {
      console.error("Validation error:", error);
      setUploadStatus({
        success: false,
        message: "Error validating file. Please try again.",
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
    formData.append("file", file);

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
    }

    if (renameDict && renameDict !== "{}") {
      formData.append("rename_dict", renameDict);
    }

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        headers: {
          token: "backendapi1234567890",
        },
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        setUploadStatus({
          success: true,
          message: "File uploaded successfully",
        });
        setFile(null);
        setValidationComplete(false);
        setNewProducts([]);
      } else {
        setUploadStatus({
          success: false,
          message: result.error || "Failed to upload file",
        });
      }
    } catch (error) {
      console.error("Upload error:", error);
      setUploadStatus({
        success: false,
        message: "Error uploading file. Please try again.",
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
              <AlertDescription>{uploadStatus.message}</AlertDescription>
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
                <Badge variant="outline" className="bg-green-50 text-green-700">
                  Keep: {getProductCounts().keep}
                </Badge>
                <Badge variant="outline" className="bg-blue-50 text-blue-700">
                  Rename: {getProductCounts().rename}
                </Badge>
                <Badge variant="outline" className="bg-red-50 text-red-700">
                  Exclude: {getProductCounts().deny}
                </Badge>
              </div>

              <Button variant="outline" size="sm" onClick={resetForm}>
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
                            onClick={() => handleProductAction(index, "keep")}
                            className={
                              product.action === "keep" ? "bg-green-100" : ""
                            }
                          >
                            <CheckCircle2 className="h-4 w-4" />
                            <span className="sr-only">Keep</span>
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
                            className={
                              product.action === "rename" ? "bg-blue-100" : ""
                            }
                          >
                            <Edit className="h-4 w-4" />
                            <span className="sr-only">Rename</span>
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => handleProductAction(index, "deny")}
                            className={
                              product.action === "deny" ? "bg-red-100" : ""
                            }
                          >
                            <Trash2 className="h-4 w-4" />
                            <span className="sr-only">Exclude</span>
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

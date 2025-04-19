import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import UploadForm from "@/components/upload-form";

export default function UploadPage() {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Upload Data</CardTitle>
          <CardDescription>
            Upload Excel files with sales data to the database
          </CardDescription>
        </CardHeader>
        <CardContent>
          <UploadForm />
        </CardContent>
      </Card>
    </div>
  );
}

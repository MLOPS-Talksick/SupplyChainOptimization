import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import UploadForm from "@/components/upload-form";
import { GlowingEffect } from "@/components/ui/glowing-effect";

export default function UploadPage() {
  return (
    <div className="space-y-4">
      <Card className="relative">
        <GlowingEffect
          spread={40}
          glow={true}
          disabled={false}
          proximity={64}
          inactiveZone={0.01}
        />
        <div className="relative">
          <CardHeader>
            <CardTitle>Upload Data</CardTitle>
            <CardDescription>
              Upload Excel files with sales data to the database
            </CardDescription>
          </CardHeader>
          <CardContent>
            <UploadForm />
          </CardContent>
        </div>
      </Card>
    </div>
  );
}

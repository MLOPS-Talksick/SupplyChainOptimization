import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import ForecastSection from "@/components/forecast-section";

export default function ForecastPage() {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Demand Forecasting</CardTitle>
          <CardDescription>Predict future demand for products</CardDescription>
        </CardHeader>
        <CardContent>
          <ForecastSection />
        </CardContent>
      </Card>
    </div>
  );
}

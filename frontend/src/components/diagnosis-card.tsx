import { AlertTriangle, BellRing, CheckCircle2, Calendar, Activity } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Diagnosis } from "@/lib/types";

const STYLES: Record<
  Diagnosis["action_band"],
  { wrap: string; chip: string; chipText: string; icon: React.ElementType; label: string }
> = {
  urgent: {
    wrap: "border-red-500 bg-red-50/60",
    chip: "bg-red-500",
    chipText: "text-red-700",
    icon: BellRing,
    label: "URGENT",
  },
  soon: {
    wrap: "border-amber-500 bg-amber-50/60",
    chip: "bg-amber-500",
    chipText: "text-amber-700",
    icon: AlertTriangle,
    label: "SOON",
  },
  routine: {
    wrap: "border-sky-500 bg-sky-50/60",
    chip: "bg-sky-500",
    chipText: "text-sky-700",
    icon: Calendar,
    label: "ROUTINE",
  },
  none: {
    wrap: "border-emerald-500 bg-emerald-50/60",
    chip: "bg-emerald-500",
    chipText: "text-emerald-700",
    icon: CheckCircle2,
    label: "OK",
  },
};

const FACTOR_STYLES: Record<"low" | "medium" | "high", string> = {
  low: "bg-sky-50 text-sky-700 border-sky-200",
  medium: "bg-amber-50 text-amber-700 border-amber-200",
  high: "bg-red-50 text-red-700 border-red-200",
};

export function DiagnosisCard({ d }: { d: Diagnosis }) {
  const s = STYLES[d.action_band];
  const Icon = s.icon;

  return (
    <Card className={cn("border-l-4 shadow-md", s.wrap)}>
      <CardContent className="space-y-5 p-6">
        {/* Top row: action title + band chip */}
        <div className="flex items-start gap-4">
          <div className={cn("grid h-12 w-12 flex-shrink-0 place-items-center rounded-xl", s.chip)}>
            <Icon className="h-6 w-6 text-white" />
          </div>
          <div className="flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className={cn("text-xl font-bold leading-tight md:text-2xl", s.chipText)}>
                {d.action_title}
              </h2>
              <Badge className={cn("uppercase tracking-wider", s.chip, "text-white hover:opacity-90")}>
                {s.label}
              </Badge>
            </div>
            <p className="mt-2 text-sm text-foreground/90 md:text-[15px]">{d.summary}</p>
          </div>
        </div>

        {/* Action recommendation */}
        <div className="rounded-lg border bg-card p-4">
          <div className="mb-1.5 flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
            <Activity className="h-3 w-3" /> Khuyến nghị lâm sàng
          </div>
          <p className="text-sm text-foreground/90">{d.action}</p>
        </div>

        {/* Buckets + max metrics */}
        <div className="grid gap-3 sm:grid-cols-5">
          <Bucket label="Cần can thiệp" value={d.nodule_buckets.actionable} color="red" />
          <Bucket label="Theo dõi" value={d.nodule_buckets.monitor} color="amber" />
          <Bucket label="Bỏ qua" value={d.nodule_buckets.incidental} color="emerald" />
          <Bucket
            label="Nốt lớn nhất"
            value={d.max_diameter_mm > 0 ? `${d.max_diameter_mm.toFixed(1)} mm` : "—"}
          />
          <Bucket
            label="Brock cao nhất"
            value={d.max_brock_pct > 0 ? `${d.max_brock_pct.toFixed(1)}%` : "—"}
          />
        </div>

        {/* Index nodule reason */}
        {d.index_nodule_id != null && d.index_reason && (
          <div className="rounded-lg border bg-card p-4">
            <div className="mb-1 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
              Nốt cần ưu tiên — #{d.index_nodule_id}
            </div>
            <p className="text-sm text-foreground/90">{d.index_reason}</p>
          </div>
        )}

        {/* Risk factors */}
        {d.risk_factors.length > 0 && (
          <div>
            <div className="mb-2 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
              Yếu tố nguy cơ ({d.risk_factors.length})
            </div>
            <div className="flex flex-wrap gap-2">
              {d.risk_factors.map((rf, i) => (
                <span
                  key={i}
                  className={cn(
                    "inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium",
                    FACTOR_STYLES[rf.weight]
                  )}
                >
                  {rf.factor}
                </span>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function Bucket({
  label,
  value,
  color,
}: {
  label: string;
  value: number | string;
  color?: "red" | "amber" | "emerald";
}) {
  const cc = {
    red: "text-red-600",
    amber: "text-amber-600",
    emerald: "text-emerald-600",
  }[color ?? "red"];
  return (
    <div className="rounded-lg border bg-card p-3 text-center">
      <div className={cn("font-mono text-2xl font-bold leading-none tabular-nums", color && cc)}>
        {value}
      </div>
      <div className="mt-1.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
    </div>
  );
}

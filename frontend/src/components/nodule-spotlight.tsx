import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Nodule } from "@/lib/types";
import { riskBadgeClass } from "@/lib/risk";

export function NoduleSpotlight({ n }: { n: Nodule }) {
  const risk = n.risk_combined ?? "low";
  const borderClass = {
    high: "border-red-300",
    medium: "border-amber-300",
    low: "border-emerald-300",
    review: "border-sky-300",
  }[risk];

  const stats: { k: string; v: string }[] = [
    { k: "AI score", v: n.ai_class != null ? `${n.ai_class}/5` : "—" },
    { k: "P(suspicious)", v: n.ai_susp_prob != null ? `${(n.ai_susp_prob * 100).toFixed(0)}%` : "—" },
    { k: "Brock 4-yr", v: n.brock_prob != null ? `${(n.brock_prob * 100).toFixed(1)}%` : "—" },
    { k: "Volume", v: `${n.volume_mm3.toFixed(0)} mm³` },
  ];

  return (
    <div className={cn("grid gap-5 rounded-xl border-2 bg-card p-5 md:grid-cols-2", borderClass)}>
      <div>
        <div className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
          Index nodule (lớn nhất)
        </div>
        <div className="mt-1 font-mono text-3xl font-bold tracking-tight">#{n.id}</div>
        <p className="mt-3 text-sm text-foreground/80">
          <strong className="font-mono tabular-nums">{n.diameter_mm.toFixed(1)} mm</strong>
          <span className="px-1">·</span> {n.nodule_type ?? "solid"}
          <span className="px-1">·</span> {n.upper_lobe ? "thuỳ trên" : "thuỳ dưới"}
          {n.lung_rads && (
            <>
              <span className="px-1">·</span>
              <Badge variant="secondary" className="font-mono text-[11px]">
                LR {n.lung_rads.category}
              </Badge>
            </>
          )}
        </p>
        <Badge className={cn("mt-3 font-bold uppercase", riskBadgeClass(risk))}>{risk}</Badge>
      </div>
      <div className="grid grid-cols-2 gap-3">
        {stats.map((s) => (
          <div key={s.k} className="rounded-md bg-muted/50 p-3">
            <div className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
              {s.k}
            </div>
            <div className="mt-1 font-mono text-xl font-bold tabular-nums">{s.v}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

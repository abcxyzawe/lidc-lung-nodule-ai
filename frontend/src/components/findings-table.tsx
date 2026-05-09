import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import type { Nodule } from "@/lib/types";
import { riskBadgeClass, rowTintClass } from "@/lib/risk";
import { NoduleThumb } from "@/components/nodule-thumb";

export function FindingsTable({ nodules }: { nodules: Nodule[] }) {
  const sorted = [...nodules].sort((a, b) => b.diameter_mm - a.diameter_mm);

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>#</TableHead>
            <TableHead>Slice</TableHead>
            <TableHead>Đường kính</TableHead>
            <TableHead>Loại</TableHead>
            <TableHead>Vị trí</TableHead>
            <TableHead>Lung-RADS</TableHead>
            <TableHead>AI</TableHead>
            <TableHead>P(susp)</TableHead>
            <TableHead>Brock 4-yr</TableHead>
            <TableHead>Tổng</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sorted.map((n, i) => {
            const risk = n.risk_combined ?? "low";
            return (
              <TableRow
                key={n.id}
                className={cn(
                  rowTintClass(risk),
                  i === 0 && "border-l-[3px] border-l-red-500"
                )}
              >
                <TableCell className="font-mono font-bold tabular-nums">{n.id}</TableCell>
                <TableCell>
                  <NoduleThumb n={n} />
                </TableCell>
                <TableCell>
                  <span className="font-mono font-bold tabular-nums">
                    {n.diameter_mm.toFixed(1)}
                  </span>{" "}
                  mm
                </TableCell>
                <TableCell className="text-muted-foreground">
                  {n.nodule_type ?? "solid"}
                </TableCell>
                <TableCell className="text-muted-foreground">
                  {n.upper_lobe ? "thuỳ trên" : "thuỳ dưới"}
                </TableCell>
                <TableCell>
                  {n.lung_rads ? (
                    <>
                      <Badge variant="secondary" className="font-mono text-[10px]">
                        LR {n.lung_rads.category}
                      </Badge>
                      <span className="mt-0.5 block text-[11px] text-muted-foreground">
                        {n.lung_rads.label}
                      </span>
                    </>
                  ) : (
                    "—"
                  )}
                </TableCell>
                <TableCell>
                  {n.ai_class != null ? (
                    <>
                      <span className="font-mono font-bold tabular-nums">{n.ai_class}/5</span>
                      <span className="mt-0.5 block text-[11px] text-muted-foreground">
                        EV {n.ai_expected?.toFixed(2)}
                      </span>
                    </>
                  ) : (
                    "—"
                  )}
                </TableCell>
                <TableCell className="font-mono tabular-nums">
                  {n.ai_susp_prob != null ? `${(n.ai_susp_prob * 100).toFixed(0)}%` : "—"}
                </TableCell>
                <TableCell>
                  {n.brock_prob != null && n.brock_band ? (
                    <Badge className={cn("font-mono", riskBadgeClass(n.brock_band))}>
                      {(n.brock_prob * 100).toFixed(1)}%
                    </Badge>
                  ) : (
                    "—"
                  )}
                </TableCell>
                <TableCell>
                  <Badge className={cn("font-bold uppercase", riskBadgeClass(risk))}>
                    {risk}
                  </Badge>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

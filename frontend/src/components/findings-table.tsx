"use client";

import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import type { Nodule, ModelKey } from "@/lib/types";
import { NoduleThumb } from "@/components/nodule-thumb";
import { Check, X } from "lucide-react";

type Verdict = "accept" | "reject" | "review" | undefined;

function lungRadsClass(category?: string): string {
  if (!category) return "";
  if (category === "2") return "bg-emerald-100 text-emerald-700";
  if (category === "3") return "bg-amber-100 text-amber-700";
  return "bg-red-100 text-red-700"; // 4A/4B/4X
}

type ServerVerdict = { verdict: Verdict; reason?: string; ts?: string };

export function FindingsTable({
  nodules,
  caseId,
  model,
}: {
  nodules: Nodule[];
  caseId: string;
  model?: ModelKey;
}) {
  const [verdicts, setVerdicts] = useState<Record<number, Verdict>>({});

  useEffect(() => {
    fetch(`http://127.0.0.1:8081/api/case/${caseId}`)
      .then((r) => r.json())
      .then((d) => {
        const sv = (d.verdicts ?? {}) as Record<string, ServerVerdict>;
        const out: Record<number, Verdict> = {};
        Object.entries(sv).forEach(([k, v]) => { out[Number(k)] = v.verdict; });
        setVerdicts(out);
      })
      .catch((err) => console.error("Verdict load failed:", err));
  }, [caseId]);

  function setVerdict(id: number, v: Verdict) {
    setVerdicts((prev) => ({ ...prev, [id]: v }));
    fetch(`http://127.0.0.1:8081/api/case/${caseId}/verdict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nodule_id: id, verdict: v ?? null }),
    }).catch((err) => console.error("Verdict save failed:", err));
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-12">#</TableHead>
            <TableHead>Slice</TableHead>
            <TableHead>Đường kính</TableHead>
            <TableHead>Loại</TableHead>
            <TableHead>Vị trí</TableHead>
            <TableHead>Lung-RADS</TableHead>
            {model === "gt" ? (
              <TableHead>Độ tin cậy GT</TableHead>
            ) : (
              <>
                <TableHead>Confidence</TableHead>
                <TableHead>Review</TableHead>
              </>
            )}
          </TableRow>
        </TableHeader>
        <TableBody>
          {nodules.map((n) => {
            const v = verdicts[n.id];
            return (
              <TableRow
                key={n.id}
                className={cn(
                  v === "accept" && "bg-emerald-50/60",
                  v === "reject" && "bg-muted/40 opacity-60"
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
                  {n.diameter_full_mm && Math.abs(n.diameter_full_mm - n.diameter_mm) > 1 && (
                    <span className="ml-1 text-[10px] text-muted-foreground">
                      (mask {n.diameter_full_mm.toFixed(1)}mm)
                    </span>
                  )}
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
                      <Badge className={cn("font-mono text-[10px]", lungRadsClass(n.lung_rads.category))}>
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
                {model === "gt" ? (
                  <TableCell>
                    <GtTierBadges n={n} />
                  </TableCell>
                ) : (
                  <>
                    <TableCell>
                      {n.confidence != null ? (
                        <div className="flex items-center gap-2">
                          <div className="h-2 w-16 overflow-hidden rounded-full bg-muted">
                            <div
                              className={cn(
                                "h-full",
                                n.confidence >= 0.85 ? "bg-emerald-500"
                                : n.confidence >= 0.7 ? "bg-amber-500"
                                : "bg-red-400"
                              )}
                              style={{ width: `${n.confidence * 100}%` }}
                            />
                          </div>
                          <span className="font-mono text-xs tabular-nums">
                            {(n.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      ) : "—"}
                    </TableCell>
                    <TableCell>
                      <div className="flex gap-1">
                        <Button
                          size="sm"
                          variant={v === "accept" ? "default" : "outline"}
                          className={cn(
                            "h-7 px-2",
                            v === "accept" && "bg-emerald-600 hover:bg-emerald-700"
                          )}
                          onClick={() => setVerdict(n.id, v === "accept" ? undefined : "accept")}
                          title="Confirm là nodule thật"
                        >
                          <Check className="h-3 w-3" />
                        </Button>
                        <Button
                          size="sm"
                          variant={v === "reject" ? "default" : "outline"}
                          className={cn(
                            "h-7 px-2",
                            v === "reject" && "bg-red-600 hover:bg-red-700"
                          )}
                          onClick={() => setVerdict(n.id, v === "reject" ? undefined : "reject")}
                          title="Reject là false positive"
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      </div>
                    </TableCell>
                  </>
                )}
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

function GtTierBadges({ n }: { n: Nodule }) {
  const readers = n.n_readers ?? 0;
  const readerBadge =
    readers >= 3 ? (
      <Badge className="bg-emerald-100 text-emerald-700 hover:bg-emerald-100 font-medium text-[10px]">
        ≥3 bác sĩ đọc
      </Badge>
    ) : (
      <Badge className="bg-muted text-muted-foreground hover:bg-muted font-medium text-[10px]">
        {readers}/4 bác sĩ
      </Badge>
    );

  const malignantBadge =
    n.malignant === true ? (
      <Badge className="bg-red-100 text-red-700 hover:bg-red-100 font-medium text-[10px]">
        Nghi ác tính
        {n.malignancy_score != null
          ? ` (${n.malignancy_score.toFixed(1)}/5)`
          : ""}
      </Badge>
    ) : null;

  return (
    <div className="flex flex-col gap-1">
      {readerBadge}
      {malignantBadge}
    </div>
  );
}

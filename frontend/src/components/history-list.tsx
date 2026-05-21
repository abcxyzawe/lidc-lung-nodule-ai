"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { listCases } from "@/lib/api";
import type { CaseListItem } from "@/lib/types";
import { MODEL_META } from "@/lib/model-meta";

export function HistoryList() {
  const [cases, setCases] = useState<CaseListItem[] | null>(null);

  useEffect(() => {
    listCases().then(setCases).catch(() => setCases([]));
  }, []);

  if (!cases || cases.length === 0) return null;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Lịch sử phân tích</CardTitle>
        <span className="text-xs text-muted-foreground">{cases.length} ca</span>
      </CardHeader>
      <CardContent>
        <ul className="divide-y">
          {cases.map((c) => {
            const mm = MODEL_META[c.model ?? "mine"];
            return (
              <li key={c.id} className="flex items-center justify-between gap-3 py-3">
                <div className="flex items-center gap-2 min-w-0">
                  <Link
                    href={`/case/${c.id}`}
                    className="font-semibold text-foreground hover:text-primary hover:underline truncate"
                  >
                    {c.name}
                  </Link>
                  <span className={`shrink-0 rounded border px-2 py-0.5 text-[10px] font-medium ${mm.badgeClass}`}>
                    {mm.label}
                  </span>
                </div>
                <span className="text-xs text-muted-foreground shrink-0">
                  {c.n_nodules} nodule · {c.n_slices} slice · {c.ts}
                </span>
              </li>
            );
          })}
        </ul>
      </CardContent>
    </Card>
  );
}

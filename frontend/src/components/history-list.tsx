"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { listCases } from "@/lib/api";
import type { CaseListItem } from "@/lib/types";

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
          {cases.map((c) => (
            <li key={c.id} className="flex items-center justify-between py-3">
              <Link
                href={`/case/${c.id}`}
                className="font-semibold text-foreground hover:text-primary hover:underline"
              >
                {c.name}
              </Link>
              <span className="text-xs text-muted-foreground">
                {c.n_nodules} nodule · {c.n_slices} slice · {c.ts}
              </span>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

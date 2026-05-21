"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceDot,
  ResponsiveContainer,
} from "recharts";
import { Topbar } from "@/components/topbar";
import { getTrainingMetrics } from "@/lib/api";
import type { TrainingMetrics } from "@/lib/types";

// ────────────────────────────────────────────────────────────────────────────
// Summary cards
// ────────────────────────────────────────────────────────────────────────────

function SummaryCard({
  label,
  value,
  sub,
  highlight,
}: {
  label: string;
  value: string;
  sub?: string;
  highlight?: boolean;
}) {
  return (
    <div
      className={`rounded-xl border p-4 ${highlight ? "border-blue-500 bg-blue-50 dark:bg-blue-950/30" : "bg-card"}`}
    >
      <p className="text-xs text-muted-foreground">{label}</p>
      <p
        className={`mt-1 text-2xl font-bold tabular-nums ${highlight ? "text-blue-700 dark:text-blue-300" : ""}`}
      >
        {value}
      </p>
      {sub && <p className="mt-0.5 text-[11px] text-muted-foreground">{sub}</p>}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

function fmt2(n: number) {
  return n.toFixed(2);
}
function fmt4(n: number | null | undefined) {
  return n == null ? "—" : n.toFixed(4);
}
function pct(n: number | null | undefined) {
  return n == null ? "—" : `${(n * 100).toFixed(1)}%`;
}

// ────────────────────────────────────────────────────────────────────────────
// Stage2 chart
// ────────────────────────────────────────────────────────────────────────────

function Stage2Chart({
  data,
  bestEpoch,
}: {
  data: TrainingMetrics["stage2"];
  bestEpoch: number;
}) {
  const bestRow = data.find((r) => r.epoch === bestEpoch);

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart
        data={data}
        margin={{ top: 8, right: 20, bottom: 8, left: 0 }}
      >
        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
        <XAxis
          dataKey="epoch"
          label={{ value: "Epoch", position: "insideBottom", offset: -2, fontSize: 12 }}
          tick={{ fontSize: 11 }}
        />
        <YAxis tick={{ fontSize: 11 }} width={52} />
        <Tooltip
          contentStyle={{ fontSize: 12 }}
          formatter={(value: unknown, name: unknown) => [
            typeof value === "number" ? fmt4(value) : String(value ?? ""),
            String(name ?? ""),
          ]}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line
          type="monotone"
          dataKey="loss"
          name="Loss"
          stroke="#ef4444"
          dot={false}
          strokeWidth={1.5}
        />
        <Line
          type="monotone"
          dataKey="val_dice"
          name="Val Dice"
          stroke="#3b82f6"
          dot={false}
          strokeWidth={1.5}
        />
        {bestRow && (
          <>
            <ReferenceDot
              x={bestRow.epoch}
              y={bestRow.loss}
              r={6}
              fill="#ef4444"
              stroke="white"
              strokeWidth={2}
              label={{ value: `e${bestEpoch}`, position: "top", fontSize: 11, fill: "#ef4444" }}
            />
            <ReferenceDot
              x={bestRow.epoch}
              y={bestRow.val_dice}
              r={6}
              fill="#3b82f6"
              stroke="white"
              strokeWidth={2}
            />
          </>
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// FPR chart
// ────────────────────────────────────────────────────────────────────────────

function FprChart({
  data,
  bestEpoch,
}: {
  data: TrainingMetrics["fpr"];
  bestEpoch: number;
}) {
  const bestRow = data.find((r) => r.epoch === bestEpoch);

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart
        data={data}
        margin={{ top: 8, right: 20, bottom: 8, left: 0 }}
      >
        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
        <XAxis
          dataKey="epoch"
          label={{ value: "Epoch", position: "insideBottom", offset: -2, fontSize: 12 }}
          tick={{ fontSize: 11 }}
        />
        <YAxis tick={{ fontSize: 11 }} width={52} />
        <Tooltip
          contentStyle={{ fontSize: 12 }}
          formatter={(value: unknown, name: unknown) => [
            typeof value === "number" ? fmt4(value) : String(value ?? ""),
            String(name ?? ""),
          ]}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line
          type="monotone"
          dataKey="loss"
          name="Loss"
          stroke="#ef4444"
          dot={false}
          strokeWidth={1.5}
        />
        <Line
          type="monotone"
          dataKey="val_acc"
          name="Val Acc"
          stroke="#3b82f6"
          dot={false}
          strokeWidth={1.5}
        />
        <Line
          type="monotone"
          dataKey="val_bal_acc"
          name="Val Bal Acc"
          stroke="#10b981"
          dot={false}
          strokeWidth={1.5}
        />
        <Line
          type="monotone"
          dataKey="val_susp_f1"
          name="Val Susp F1"
          stroke="#f59e0b"
          dot={false}
          strokeWidth={1.5}
        />
        {bestRow && (
          <ReferenceDot
            x={bestRow.epoch}
            y={bestRow.val_susp_f1}
            r={6}
            fill="#f59e0b"
            stroke="white"
            strokeWidth={2}
            label={{ value: `e${bestEpoch}`, position: "top", fontSize: 11, fill: "#f59e0b" }}
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Test panel table
// ────────────────────────────────────────────────────────────────────────────

function TestPanelTable({ s }: { s: TrainingMetrics["summary"] }) {
  const rows: { metric: string; value: string; note: string }[] = [
    { metric: "Test Accuracy", value: pct(s.test_acc), note: "Overall correct predictions" },
    { metric: "Test Balanced Accuracy", value: pct(s.test_bal_acc), note: "Macro avg across classes" },
    { metric: "Test Susp F1", value: fmt4(s.test_susp_f1), note: "F1 for suspicious class" },
    { metric: "Mine F1 (test panel)", value: fmt4(s.mine_f1_test_panel), note: "Full pipeline F1 — primary KPI" },
    { metric: "FPR AUC", value: fmt4(s.fpr_auc), note: "Area under ROC curve" },
    { metric: "FPR Best Threshold", value: fmt2(s.fpr_best_thr), note: "Optimal decision threshold" },
  ];

  return (
    <div className="overflow-x-auto rounded-xl border">
      <table className="w-full text-sm">
        <thead className="bg-muted/50">
          <tr>
            <th className="px-4 py-2 text-left font-medium">Metric</th>
            <th className="px-4 py-2 text-right font-medium">Value</th>
            <th className="px-4 py-2 text-left font-medium text-muted-foreground">Note</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.metric} className="border-t transition-colors hover:bg-muted/30">
              <td className="px-4 py-2 font-medium">{r.metric}</td>
              <td className="px-4 py-2 text-right tabular-nums font-mono">{r.value}</td>
              <td className="px-4 py-2 text-muted-foreground">{r.note}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Page
// ────────────────────────────────────────────────────────────────────────────

export default function TrainingPage() {
  const [data, setData] = useState<TrainingMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getTrainingMetrics()
      .then(setData)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, []);

  const s = data?.summary;

  return (
    <>
      <Topbar subtitle="Chỉ số huấn luyện mô hình" />
      <main className="mx-auto max-w-5xl space-y-8 px-4 py-8 md:px-6 md:py-10">
        {/* Header */}
        <header>
          <h1 className="text-2xl font-bold tracking-tight md:text-3xl">
            Chỉ số huấn luyện
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Training metrics cho Stage 2 (segmentation) và FPR classifier (DenseNet121-3D).
            AI hỗ trợ đọc thứ hai — không phải chẩn đoán độc lập.
          </p>
        </header>

        {/* Loading */}
        {loading && (
          <div className="flex items-center gap-3 rounded-xl border bg-muted/30 px-6 py-10 text-muted-foreground">
            <svg className="h-5 w-5 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
            </svg>
            Đang tải dữ liệu huấn luyện...
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="rounded-xl border border-destructive/40 bg-destructive/10 px-6 py-5 text-sm text-destructive">
            <strong>Lỗi:</strong> {error}
            <p className="mt-1 text-xs text-muted-foreground">
              Backend endpoint <code>/api/training-metrics</code> chưa sẵn sàng hoặc server chưa chạy.
            </p>
          </div>
        )}

        {data && s && (
          <>
            {/* Summary cards */}
            <section>
              <h2 className="mb-3 text-lg font-semibold">Tóm tắt kết quả</h2>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <SummaryCard
                  label="Mine F1 (test panel)"
                  value={fmt4(s.mine_f1_test_panel)}
                  sub="Toàn pipeline — KPI chính"
                  highlight
                />
                <SummaryCard
                  label="Stage 2 Best Val Dice"
                  value={fmt4(s.stage2_best_val_dice)}
                  sub={`Epoch ${s.stage2_best_epoch} / ${s.stage2_total_epochs}`}
                />
                <SummaryCard
                  label="FPR AUC"
                  value={fmt4(s.fpr_auc)}
                  sub={`Best thr = ${fmt2(s.fpr_best_thr)}`}
                />
                <SummaryCard
                  label="FPR Best Threshold"
                  value={fmt2(s.fpr_best_thr)}
                  sub={`${s.fpr_total_epochs} epochs`}
                />
              </div>
            </section>

            {/* Stage 2 chart */}
            <section>
              <h2 className="mb-1 text-lg font-semibold">Stage 2 — Segmentation</h2>
              <p className="mb-3 text-sm text-muted-foreground">
                Loss (red) vs Val Dice (blue). Dấu chấm = epoch tốt nhất (epoch {s.stage2_best_epoch}).
              </p>
              <div className="rounded-xl border bg-card p-4">
                <Stage2Chart data={data.stage2} bestEpoch={s.stage2_best_epoch} />
              </div>
            </section>

            {/* FPR chart */}
            <section>
              <h2 className="mb-1 text-lg font-semibold">FPR Classifier — DenseNet121-3D</h2>
              <p className="mb-3 text-sm text-muted-foreground">
                Loss, Val Acc, Val Balanced Acc, Val Susp F1. Dấu chấm vàng = best epoch theo Susp F1.
              </p>
              <div className="rounded-xl border bg-card p-4">
                <FprChart data={data.fpr} bestEpoch={s.fpr_best_epoch} />
              </div>
            </section>

            {/* Test panel table */}
            <section>
              <h2 className="mb-3 text-lg font-semibold">Test panel — Final metrics</h2>
              <TestPanelTable s={s} />
            </section>

            {/* Disclaimer */}
            <p className="text-xs text-muted-foreground">
              * Kết quả huấn luyện trên tập nội bộ LIDC-IDRI. Mô hình là công cụ
              hỗ trợ đọc thứ hai (second reader assist) — không thay thế đánh giá lâm sàng
              của bác sĩ chuyên khoa.
            </p>
          </>
        )}

        {/* Back link */}
        <div>
          <Link href="/" className="text-sm text-blue-600 hover:underline dark:text-blue-400">
            Quay lại trang chính
          </Link>
        </div>
      </main>
    </>
  );
}

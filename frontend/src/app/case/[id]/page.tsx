"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { ArrowLeft, Trash2, Stethoscope } from "lucide-react";

import { Topbar } from "@/components/topbar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Table, TableBody, TableCell, TableRow,
} from "@/components/ui/table";
import { FindingsTable } from "@/components/findings-table";
import { Plot3D } from "@/components/plot-3d";
import { cn } from "@/lib/utils";
import { deleteCase, getCase } from "@/lib/api";
import { toast } from "sonner";
import type { CaseDetail } from "@/lib/types";

export default function CasePage() {
  const router = useRouter();
  const { id } = useParams<{ id: string }>();
  const [data, setData] = useState<CaseDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    getCase(id).then(setData).catch((e) => setErr(String(e?.message ?? e)));
  }, [id]);

  if (err) {
    return (
      <>
        <Topbar />
        <main className="mx-auto max-w-3xl px-4 py-10">
          <p className="text-red-600">Không tải được case: {err}</p>
        </main>
      </>
    );
  }
  if (!data) {
    return (
      <>
        <Topbar />
        <main className="mx-auto max-w-3xl px-4 py-10 text-muted-foreground">
          Đang tải case…
        </main>
      </>
    );
  }

  const { meta, plot_html } = data;
  const total = meta.n_nodules;
  const nLR4 = meta.nodules.filter(n => {
    const c = n.lung_rads?.category;
    return c === "4A" || c === "4B" || c === "4X";
  }).length;
  const nLR3 = meta.nodules.filter(n => n.lung_rads?.category === "3").length;
  const nLR2 = meta.nodules.filter(n => n.lung_rads?.category === "2").length;
  const minutesSaved = Math.max(2, Math.round(meta.n_slices / 30));

  async function onDelete() {
    if (!confirm("Xoá case này?")) return;
    try {
      await deleteCase(id);
      toast.success("Đã xoá case");
      router.push("/");
    } catch (e) {
      toast.error(`Xoá lỗi: ${e instanceof Error ? e.message : e}`);
    }
  }

  return (
    <>
      <Topbar subtitle={`Báo cáo · ${meta.timestamp}`} />
      <main className="mx-auto max-w-7xl space-y-4 px-4 py-8 md:px-6 md:py-10">
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-primary"
        >
          <ArrowLeft className="h-4 w-4" /> Trang chủ
        </Link>

        <header>
          <h1 className="text-2xl font-bold tracking-tight md:text-3xl">{meta.name}</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            AI hỗ trợ tầm soát · {total} candidate · ~{minutesSaved} phút tiết kiệm so với scroll thủ công
          </p>
        </header>

        {/* Detection summary banner */}
        <div className={cn(
          "flex items-start gap-4 rounded-xl border-l-4 p-5",
          nLR4 > 0 ? "border-red-500 bg-red-50/60"
          : nLR3 > 0 ? "border-amber-500 bg-amber-50/60"
          : "border-emerald-500 bg-emerald-50/60"
        )}>
          <Stethoscope className={cn(
            "mt-1 h-6 w-6 flex-shrink-0",
            nLR4 > 0 ? "text-red-600" : nLR3 > 0 ? "text-amber-600" : "text-emerald-600"
          )} />
          <div className="flex-1">
            <div className="text-lg font-bold">
              {total === 0
                ? "Không phát hiện nodule"
                : `Phát hiện ${total} nodule cần bác sĩ review`}
            </div>
            <p className="mt-1 text-sm text-foreground/80">
              {total === 0
                ? "AI không tìm thấy nodule nào ≥ 6.5mm trong vùng phổi."
                : <>
                  <span className="font-mono font-semibold">{nLR4}</span> Lung-RADS 4 ·{" "}
                  <span className="font-mono font-semibold">{nLR3}</span> Lung-RADS 3 ·{" "}
                  <span className="font-mono font-semibold">{nLR2}</span> Lung-RADS 2 (bỏ qua).{" "}
                  <strong>Bác sĩ confirm/reject từng nodule trong table.</strong>
                </>
              }
            </p>
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-5">
          <Stat label="slices CT" value={meta.n_slices} />
          <Stat label="nodule" value={total} />
          <Stat label="LR-4 (đáng nghi)" value={nLR4} color="red" />
          <Stat label="LR-3 (theo dõi)" value={nLR3} color="amber" />
          <Stat label="AI infer" value={`${meta.timing_sec.ai_predict.toFixed(1)}s`} />
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">3D phổi + nodule</CardTitle>
            <p className="text-sm text-muted-foreground">
              Vỏ phổi xanh trong suốt, nodule màu đặc. Click legend để bật/tắt từng cái.
            </p>
          </CardHeader>
          <CardContent>
            <Plot3D html={plot_html} />
          </CardContent>
        </Card>

        {meta.nodules.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                Danh sách nodule (sort theo confidence — cao nhất trước)
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                Click ảnh để xem slice phóng to. Mark <strong>Accept</strong> nếu là nodule thật,
                <strong> Reject</strong> nếu là false positive.
              </p>
            </CardHeader>
            <CardContent>
              <FindingsTable nodules={meta.nodules} caseId={id} />
              <p className="mt-4 text-sm text-muted-foreground">
                <strong>Confidence:</strong> trung bình xác suất AI trong vùng nodule.
                <strong> Lung-RADS:</strong> chuẩn ACR v2022 dựa trên đường kính (rule, không phải AI).
                AI <strong>KHÔNG chẩn đoán ung thư</strong> — chỉ flag chỗ nghi để bác sĩ xem.
              </p>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Thông số kỹ thuật</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableBody>
                <TechRow k="Voxel spacing (z, y, x)" v={meta.voxel_spacing_zyx_mm.map((x) => x.toFixed(2)).join(", ") + " mm"} />
                <TechRow k="Voxel phổi (lungmask R231)" v={meta.lung_voxels.toLocaleString()} />
                <TechRow k="Voxel pred (∩ phổi)" v={meta.pred_voxels.toLocaleString()} />
                <TechRow k="Đọc DICOM" v={`${meta.timing_sec.read_dicom.toFixed(2)} s`} />
                <TechRow k="Lung segmentation" v={`${meta.timing_sec.lung_seg.toFixed(2)} s`} />
                <TechRow k="Nodule segmentation (B5 + TTA + ensemble)" v={`${meta.timing_sec.ai_predict.toFixed(2)} s`} />
                <TechRow k="Connected components + filters" v={`${meta.timing_sec.postprocess.toFixed(2)} s`} />
                <TechRow k="Render Plotly Mesh3d" v={`${meta.timing_sec.render_3d.toFixed(2)} s`} />
              </TableBody>
            </Table>
            <Button onClick={onDelete} variant="outline" className="mt-4 border-red-300 text-red-600 hover:bg-red-50 hover:text-red-700">
              <Trash2 className="mr-2 h-4 w-4" /> Xoá case này
            </Button>
          </CardContent>
        </Card>

        <footer className="pt-8 text-center text-xs text-muted-foreground">
          AI hỗ trợ tầm soát — KHÔNG thay thế chẩn đoán bác sĩ ·
          UNet++ B5 ensemble + Lungmask R231 · F1 detection 0.66 · Trained on LIDC-IDRI
        </footer>
      </main>
    </>
  );
}

function Stat({
  label,
  value,
  color,
}: {
  label: string;
  value: string | number;
  color?: "emerald" | "amber" | "red";
}) {
  const cc = {
    emerald: "text-emerald-600",
    amber: "text-amber-600",
    red: "text-red-600",
  }[color ?? "emerald"];
  return (
    <div className="rounded-lg border bg-card p-4">
      <div className={cn("font-mono text-2xl font-bold leading-tight tabular-nums", color && cc)}>
        {value}
      </div>
      <div className="mt-1 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
    </div>
  );
}

function TechRow({ k, v }: { k: string; v: string }) {
  return (
    <TableRow>
      <TableCell className="text-muted-foreground">{k}</TableCell>
      <TableCell className="text-right font-mono tabular-nums">{v}</TableCell>
    </TableRow>
  );
}

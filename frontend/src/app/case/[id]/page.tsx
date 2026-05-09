"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { ArrowLeft, Trash2 } from "lucide-react";

import { Topbar } from "@/components/topbar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table, TableBody, TableCell, TableRow,
} from "@/components/ui/table";
import { VerdictBanner } from "@/components/verdict-banner";
import { DiagnosisCard } from "@/components/diagnosis-card";
import { PatientContextBar } from "@/components/patient-context-bar";
import { NoduleSpotlight } from "@/components/nodule-spotlight";
import { FindingsTable } from "@/components/findings-table";
import { Plot3D } from "@/components/plot-3d";
import { riskBadgeClass } from "@/lib/risk";
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
  const nHigh = meta.nodules.filter((n) => n.risk_combined === "high").length;
  const nMed = meta.nodules.filter((n) => n.risk_combined === "medium").length;
  const nLow = meta.nodules.filter((n) => n.risk_combined === "low").length;

  const verdict: "low" | "medium" | "high" =
    nHigh > 0 ? "high" : nMed > 0 ? "medium" : "low";

  const verdictText = (() => {
    if (nHigh > 0)
      return {
        title: "Nguy cơ CAO — cần khám chuyên khoa",
        text: `${nHigh} nodule có nguy cơ cao. Khuyến nghị PET-CT hoặc sinh thiết theo Lung-RADS.`,
      };
    if (nMed > 0)
      return {
        title: "Cần theo dõi định kỳ",
        text: `${nMed} nodule nguy cơ trung bình. Chụp lại CT phổi sau 3-6 tháng.`,
      };
    if (meta.n_nodules > 0)
      return {
        title: "Không phát hiện nodule đáng lo",
        text: `Có ${nLow} nốt nhỏ < 6mm — Lung-RADS phân loại bỏ qua, không cần follow-up.`,
      };
    return {
      title: "Phổi sạch",
      text: "AI không tìm thấy nodule nào ≥ 4.5mm trong vùng phổi.",
    };
  })();

  const indexNodule = [...meta.nodules].sort(
    (a, b) => b.diameter_mm - a.diameter_mm
  )[0];

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
        </header>

        {meta.clinical?.diagnosis ? (
          <DiagnosisCard d={meta.clinical.diagnosis} />
        ) : (
          <VerdictBanner
            verdict={verdict}
            title={verdictText.title}
            text={verdictText.text}
          />
        )}

        {meta.clinical && <PatientContextBar cli={meta.clinical} />}

        {meta.clinical && meta.clinical.symptoms.count > 0 && (
          <Card
            className={cn(
              "border-l-4",
              meta.clinical.symptoms.level === "high" && "border-l-red-500",
              meta.clinical.symptoms.level === "medium" && "border-l-amber-500",
              meta.clinical.symptoms.level === "low" && "border-l-sky-500"
            )}
          >
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-base">
                Triệu chứng ({meta.clinical.symptoms.count})
              </CardTitle>
              <Badge className={cn("font-bold uppercase", riskBadgeClass(meta.clinical.symptoms.level))}>
                {meta.clinical.symptoms.level}
              </Badge>
            </CardHeader>
            <CardContent>
              <p className="mb-3 text-sm text-muted-foreground">
                {meta.clinical.symptoms.message}
              </p>
              <div className="flex flex-wrap gap-2">
                {meta.clinical.symptoms.active.map((s) => (
                  <Badge
                    key={s.key}
                    variant="outline"
                    className={cn("font-medium", riskBadgeClass(s.weight))}
                  >
                    {s.label}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {indexNodule && <NoduleSpotlight n={indexNodule} />}

        <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-6">
          <Stat label="slices CT" value={meta.n_slices} />
          <Stat label="nodule" value={meta.n_nodules} />
          <Stat label="low" value={nLow} color="emerald" />
          <Stat label="medium" value={nMed} color="amber" />
          <Stat label="high" value={nHigh} color="red" />
          <Stat label="AI infer" value={`${meta.timing_sec.ai_predict.toFixed(1)}s`} />
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">3D phổi + nodule</CardTitle>
            <p className="text-sm text-muted-foreground">
              Vỏ phổi xanh trong suốt + nodule màu đặc. Kéo chuột xoay, scroll zoom.
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
                Tất cả nodule ({meta.nodules.length})
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                Sắp xếp theo kích thước. Index nodule có viền đỏ bên trái.
              </p>
            </CardHeader>
            <CardContent>
              <FindingsTable nodules={meta.nodules} />
              <p className="mt-4 text-sm text-muted-foreground">
                <strong className="text-foreground">AI score 1-5</strong> — DenseNet121-3D phân loại malignancy ·
                <strong className="ml-1 text-foreground">Brock 4-yr</strong> — xác suất ung thư 4 năm (McWilliams NEJM 2013) ·
                <strong className="ml-1 text-foreground">Lung-RADS</strong> ACR v2022. Brock band: &lt;5% theo dõi
                thường, 5-10% chụp lại 3 tháng, &gt;10% PET / sinh thiết.
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
                <TechRow k="Nodule segmentation (B5 + TTA)" v={`${meta.timing_sec.ai_predict.toFixed(2)} s`} />
                <TechRow k="Connected components" v={`${meta.timing_sec.postprocess.toFixed(2)} s`} />
                <TechRow k="Malignancy classifier" v={`${meta.timing_sec.malignancy.toFixed(2)} s`} />
                {meta.timing_sec.clinical != null && (
                  <TechRow k="Brock + USPSTF" v={`${meta.timing_sec.clinical.toFixed(2)} s`} />
                )}
                <TechRow k="Render Plotly Mesh3d" v={`${meta.timing_sec.render_3d.toFixed(2)} s`} />
              </TableBody>
            </Table>
            <Button onClick={onDelete} variant="outline" className="mt-4 border-red-300 text-red-600 hover:bg-red-50 hover:text-red-700">
              <Trash2 className="mr-2 h-4 w-4" /> Xoá case này
            </Button>
          </CardContent>
        </Card>

        <footer className="pt-8 text-center text-xs text-muted-foreground">
          Không thay thế chẩn đoán của bác sĩ · UNet++ B5 + DenseNet121-3D + Lungmask R231 + Brock 2013
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
  const colorClass = {
    emerald: "text-emerald-600",
    amber: "text-amber-600",
    red: "text-red-600",
  }[color ?? "emerald"];
  return (
    <div className="rounded-lg border bg-card p-4">
      <div className={cn("font-mono text-2xl font-bold leading-tight tabular-nums", color && colorClass)}>
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

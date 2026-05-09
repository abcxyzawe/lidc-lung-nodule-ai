"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
import { Folder, FileArchive, Zap, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { streamProgress, uploadAndAnalyze } from "@/lib/api";
import type { PatientForm } from "@/lib/types";

const SYMPTOMS = [
  { key: "sym_hemoptysis", label: "Ho ra máu", weight: "high" as const, tag: "Red flag" },
  { key: "sym_weight_loss", label: "Sụt cân không rõ nguyên nhân", weight: "medium" as const, tag: "Đáng kể" },
  { key: "sym_clubbing", label: "Ngón tay dùi trống", weight: "medium" as const, tag: "Đáng kể" },
  { key: "sym_hoarseness", label: "Khàn tiếng kéo dài", weight: "medium" as const, tag: "Đáng kể" },
  { key: "sym_cough", label: "Ho mạn tính > 3 tuần", weight: "low" as const, tag: "Nhẹ" },
  { key: "sym_chest_pain", label: "Đau ngực", weight: "low" as const, tag: "Nhẹ" },
  { key: "sym_dyspnea", label: "Khó thở mới hoặc xấu đi", weight: "low" as const, tag: "Nhẹ" },
  { key: "sym_recurrent_infection", label: "Nhiễm trùng ngực tái phát", weight: "low" as const, tag: "Nhẹ" },
];

const SYM_STYLES: Record<"high" | "medium" | "low", string> = {
  high: "border-red-200 bg-red-50/40 text-red-700 hover:border-red-300 data-[state=checked]:border-red-500 data-[state=checked]:bg-red-50",
  medium: "border-amber-200 bg-amber-50/40 text-amber-700 hover:border-amber-300 data-[state=checked]:border-amber-500 data-[state=checked]:bg-amber-50",
  low: "border-sky-200 bg-sky-50/40 text-sky-700 hover:border-sky-300 data-[state=checked]:border-sky-500 data-[state=checked]:bg-sky-50",
};
const TAG_STYLES: Record<"high" | "medium" | "low", string> = {
  high: "bg-red-100 text-red-700",
  medium: "bg-amber-100 text-amber-700",
  low: "bg-sky-100 text-sky-700",
};

export function UploadForm() {
  const router = useRouter();
  const dirRef = useRef<HTMLInputElement>(null);
  const zipRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<FileList | null>(null);
  const [fileLabel, setFileLabel] = useState("");

  const [cigsPerDay, setCigsPerDay] = useState(0);
  const [yearsSmoked, setYearsSmoked] = useState(0);
  const packYears = (cigsPerDay / 20) * yearsSmoked;

  const [patient, setPatient] = useState<PatientForm>({
    age: 60,
    sex: "male",
    pack_years: 0,
    currently_smoking: false,
    years_since_quit: 0,
    family_hx: false,
    emphysema: false,
    sym_hemoptysis: false,
    sym_weight_loss: false,
    sym_clubbing: false,
    sym_hoarseness: false,
    sym_cough: false,
    sym_chest_pain: false,
    sym_dyspnea: false,
    sym_recurrent_infection: false,
  });

  const [submitting, setSubmitting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState("");
  const [log, setLog] = useState<{ msg: string; ok?: boolean; err?: boolean }[]>([]);

  function setField<K extends keyof PatientForm>(k: K, v: PatientForm[K]) {
    setPatient((p) => ({ ...p, [k]: v }));
  }

  function pickFiles(fl: FileList | null, label: string) {
    if (!fl || fl.length === 0) return;
    setFiles(fl);
    setFileLabel(label);
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!files) {
      toast.error("Vui lòng chọn folder DICOM hoặc file .zip");
      return;
    }
    setSubmitting(true);
    setLog([]);
    setProgress(0);
    setProgressText("Đang upload…");

    // Compose final patient form: derive pack_years from the user-friendly inputs
    const submitPatient: PatientForm = { ...patient, pack_years: Number(packYears.toFixed(2)) };

    try {
      const r = await uploadAndAnalyze(files, submitPatient, (loaded, total) => {
        const upPct = (loaded / total) * 100;
        setProgress(Math.round(upPct * 0.3));
        setProgressText(`Upload ${(loaded / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB`);
      });

      setLog((l) => [...l, { msg: `Upload xong (${r.n_dicoms} DICOM). Bắt đầu AI…`, ok: true }]);
      setProgress(30);
      setProgressText("Server nhận file, đang khởi động AI…");

      streamProgress(
        r.case_id,
        (pct, msg) => {
          setLog((l) => [...l, { msg: `${pct >= 0 ? pct + "%" : "ERR"} — ${msg}`, err: pct < 0 }]);
          if (pct >= 0) {
            setProgress(pct);
            setProgressText(msg);
          }
        },
        (finalPct) => {
          if (finalPct === 100) {
            setTimeout(() => router.push(`/case/${r.case_id}`), 600);
          } else {
            setSubmitting(false);
            toast.error("AI gặp lỗi — kiểm tra log bên dưới");
          }
        }
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      toast.error(`Lỗi upload: ${msg}`);
      setLog((l) => [...l, { msg, err: true }]);
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={onSubmit}>
      <Card>
        <CardContent className="space-y-8 pt-6">
          {/* === STEP 1: Files === */}
          <section>
            <div className="mb-4 flex items-center gap-3">
              <span className="grid h-6 w-6 place-items-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                1
              </span>
              <h3 className="text-base font-semibold">Ảnh CT (DICOM)</h3>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <DropZone
                icon={<Folder className="h-5 w-5 text-primary" />}
                title="Chọn folder DICOM"
                sub="Cả thư mục .dcm"
                onClick={() => dirRef.current?.click()}
              />
              <DropZone
                icon={<FileArchive className="h-5 w-5 text-primary" />}
                title="Chọn file .zip"
                sub="Folder DICOM đã nén"
                onClick={() => zipRef.current?.click()}
              />
              <input
                ref={dirRef}
                type="file"
                multiple
                // @ts-expect-error directory upload non-standard but supported in Chrome/Edge/Safari
                webkitdirectory=""
                directory=""
                hidden
                onChange={(e) => pickFiles(e.target.files, `${e.target.files?.length ?? 0} file từ folder`)}
              />
              <input
                ref={zipRef}
                type="file"
                accept=".zip"
                hidden
                onChange={(e) =>
                  pickFiles(e.target.files, e.target.files?.[0]?.name ?? "")
                }
              />
            </div>
            {fileLabel && (
              <p className="mt-3 flex items-center gap-2 text-sm font-medium text-emerald-700">
                <CheckCircle2 className="h-4 w-4" />
                {fileLabel}
              </p>
            )}
          </section>

          <Separator />

          {/* === STEP 2: Patient === */}
          <section>
            <div className="mb-4 flex items-center gap-3">
              <span className="grid h-6 w-6 place-items-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                2
              </span>
              <h3 className="text-base font-semibold">
                Bệnh nhân
                <span className="ml-2 text-xs font-normal text-muted-foreground">
                  — bỏ trống nếu chưa rõ
                </span>
              </h3>
            </div>
            <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-4">
              <Field label="Tuổi">
                <Input
                  type="number"
                  min={18}
                  max={100}
                  value={patient.age}
                  onChange={(e) => setField("age", Number(e.target.value))}
                />
              </Field>
              <Field label="Giới">
                <Select
                  value={patient.sex}
                  onValueChange={(v) => setField("sex", v as "male" | "female")}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="male">Nam</SelectItem>
                    <SelectItem value="female">Nữ</SelectItem>
                  </SelectContent>
                </Select>
              </Field>
              <Field label="Số điếu/ngày">
                <Input
                  type="number" min={0} max={80} step={1}
                  value={cigsPerDay}
                  onChange={(e) => setCigsPerDay(Number(e.target.value))}
                />
              </Field>
              <Field label="Số năm hút">
                <Input
                  type="number" min={0} max={80} step={1}
                  value={yearsSmoked}
                  onChange={(e) => setYearsSmoked(Number(e.target.value))}
                />
              </Field>
            </div>

            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <Field label="Số năm đã bỏ (nếu đã bỏ)">
                <Input
                  type="number" min={0} max={80}
                  value={patient.years_since_quit}
                  onChange={(e) => setField("years_since_quit", Number(e.target.value))}
                />
              </Field>
              <div className="flex items-end">
                <div className="w-full rounded-md border border-dashed bg-muted/40 px-3 py-2 text-sm">
                  <span className="text-muted-foreground">Tổng phơi nhiễm: </span>
                  <span className="font-mono font-bold tabular-nums">{packYears.toFixed(1)}</span>
                  <span className="text-muted-foreground"> gói-năm</span>
                  {packYears >= 20 && (
                    <span className="ml-2 inline-flex rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-amber-700">
                      ≥ 20 (đủ tầm soát)
                    </span>
                  )}
                </div>
              </div>
            </div>
            <div className="mt-4 grid gap-2 sm:grid-cols-3">
              <ChipCheckbox
                checked={patient.currently_smoking}
                onChange={(v) => setField("currently_smoking", v)}
                label="Đang hút thuốc"
              />
              <ChipCheckbox
                checked={patient.family_hx}
                onChange={(v) => setField("family_hx", v)}
                label="Tiền sử gia đình ung thư phổi"
              />
              <ChipCheckbox
                checked={patient.emphysema}
                onChange={(v) => setField("emphysema", v)}
                label="COPD / khí phế thũng"
              />
            </div>
          </section>

          <Separator />

          {/* === STEP 3: Symptoms === */}
          <section>
            <div className="mb-4 flex items-center gap-3">
              <span className="grid h-6 w-6 place-items-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                3
              </span>
              <h3 className="text-base font-semibold">
                Triệu chứng
                <span className="ml-2 text-xs font-normal text-muted-foreground">
                  — tick nếu có
                </span>
              </h3>
            </div>
            <div className="grid gap-2 sm:grid-cols-2 md:grid-cols-2">
              {SYMPTOMS.map((s) => (
                <SymCard
                  key={s.key}
                  weight={s.weight}
                  label={s.label}
                  tag={s.tag}
                  checked={patient[s.key as keyof PatientForm] as boolean}
                  onChange={(v) => setField(s.key as keyof PatientForm, v as never)}
                />
              ))}
            </div>
          </section>

          <Button
            type="submit"
            size="lg"
            disabled={submitting}
            className="h-12 w-full text-base"
          >
            <Zap className="mr-2 h-4 w-4" />
            {submitting ? "Đang phân tích…" : "Phân tích bằng AI"}
          </Button>

          {submitting && (
            <div className="rounded-lg border bg-muted/30 p-4">
              <Progress value={progress} className="h-2" />
              <div className="mt-3 flex items-baseline justify-between">
                <span className="font-mono text-2xl font-bold tabular-nums">{progress}%</span>
                <span className="text-sm text-muted-foreground">{progressText}</span>
              </div>
              <ul className="mt-3 max-h-48 overflow-y-auto rounded border bg-background p-2 font-mono text-[11px]">
                {log.map((l, i) => (
                  <li
                    key={i}
                    className={cn(
                      "border-b py-0.5 last:border-b-0",
                      l.ok && "text-emerald-600",
                      l.err && "text-red-600",
                      !l.ok && !l.err && "text-muted-foreground"
                    )}
                  >
                    {l.msg}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </CardContent>
      </Card>
    </form>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <Label className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </Label>
      {children}
    </div>
  );
}

function DropZone({
  icon,
  title,
  sub,
  onClick,
}: {
  icon: React.ReactNode;
  title: string;
  sub: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex min-h-[140px] flex-col items-center justify-center gap-2 rounded-lg border-[1.5px] border-dashed border-border bg-muted/30 p-6 text-center transition-colors hover:border-primary hover:bg-primary/5"
    >
      <div className="grid h-10 w-10 place-items-center rounded-lg bg-primary/10">
        {icon}
      </div>
      <div className="text-sm font-semibold">{title}</div>
      <div className="text-xs text-muted-foreground">{sub}</div>
    </button>
  );
}

function ChipCheckbox({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <label
      className={cn(
        "flex cursor-pointer items-center gap-3 rounded-lg border-[1.5px] px-4 py-2.5 text-sm font-medium transition-colors",
        checked ? "border-primary bg-primary/5 text-foreground" : "border-border hover:border-border/80 hover:bg-muted/40"
      )}
    >
      <Checkbox checked={checked} onCheckedChange={(v) => onChange(v === true)} />
      <span>{label}</span>
    </label>
  );
}

function SymCard({
  weight,
  label,
  tag,
  checked,
  onChange,
}: {
  weight: "high" | "medium" | "low";
  label: string;
  tag: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label
      className={cn(
        "flex cursor-pointer items-start gap-3 rounded-lg border-[1.5px] p-3 transition-colors",
        checked ? SYM_STYLES[weight] : "border-border bg-card hover:border-border/80 hover:bg-muted/40"
      )}
    >
      <Checkbox
        checked={checked}
        onCheckedChange={(v) => onChange(v === true)}
        className="mt-0.5"
      />
      <div className="flex-1">
        <div className="text-sm font-medium leading-tight text-foreground">{label}</div>
        <span
          className={cn(
            "mt-1.5 inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider",
            TAG_STYLES[weight]
          )}
        >
          {tag}
        </span>
      </div>
    </label>
  );
}

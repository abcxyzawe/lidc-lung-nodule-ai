"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { toast } from "sonner";
import { Folder, FileArchive, Zap, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { streamProgress, uploadAndAnalyze } from "@/lib/api";

export function UploadForm() {
  const router = useRouter();
  const dirRef = useRef<HTMLInputElement>(null);
  const zipRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<FileList | null>(null);
  const [fileLabel, setFileLabel] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState("");
  const [log, setLog] = useState<{ msg: string; ok?: boolean; err?: boolean }[]>([]);

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

    try {
      const r = await uploadAndAnalyze(files, (loaded, total) => {
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
        <CardContent className="space-y-5 pt-6">
          <div>
            <h3 className="mb-1 text-base font-semibold">Upload ảnh CT (DICOM)</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Chọn folder chứa file .dcm hoặc 1 file .zip nén folder DICOM của 1 bệnh nhân.
            </p>
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
                // @ts-expect-error directory upload non-standard but supported
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
          </div>

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

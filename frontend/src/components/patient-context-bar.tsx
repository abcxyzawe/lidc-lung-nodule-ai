import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { ClinicalAssessment } from "@/lib/types";

export function PatientContextBar({ cli }: { cli: ClinicalAssessment }) {
  const p = cli.patient;
  const items: { k: string; v: React.ReactNode }[] = [
    { k: "Tuổi · Giới", v: `${p.age} · ${p.sex === "male" ? "Nam" : "Nữ"}` },
    {
      k: "Hút thuốc",
      v: `${p.pack_years} gói-năm${
        p.currently_smoking
          ? " (đang hút)"
          : p.years_since_quit
          ? ` (bỏ ${p.years_since_quit} năm)`
          : ""
      }`,
    },
    { k: "Tiền sử gia đình", v: p.family_hx ? "Có" : "Không" },
    { k: "COPD", v: p.emphysema ? "Có" : "Không" },
  ];

  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-lg border bg-card px-5 py-3 shadow-sm">
      {items.map((it, i) => (
        <div key={it.k} className="flex items-center gap-x-6">
          <div className="flex flex-col">
            <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
              {it.k}
            </span>
            <span className="text-sm font-medium">{it.v}</span>
          </div>
          {i < items.length - 1 && <div className="hidden h-7 w-px bg-border md:block" />}
        </div>
      ))}
      <div className="hidden h-7 w-px bg-border md:block" />
      <div className="flex flex-col">
        <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
          USPSTF
        </span>
        <Badge
          className={cn(
            "mt-0.5 font-bold uppercase tracking-wide",
            cli.uspstf.eligible
              ? "bg-red-100 text-red-700 hover:bg-red-100"
              : "bg-muted text-muted-foreground hover:bg-muted"
          )}
        >
          {cli.uspstf.eligible ? "Đủ điều kiện" : "Không tầm soát"}
        </Badge>
      </div>
    </div>
  );
}

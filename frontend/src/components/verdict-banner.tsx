import { AlertTriangle, BellRing, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

type Verdict = "low" | "medium" | "high";

const STYLES: Record<
  Verdict,
  { wrap: string; icon: string; iconColor: string; title: string }
> = {
  high: {
    wrap: "border-red-500 bg-red-50",
    icon: "bg-red-500",
    iconColor: "text-white",
    title: "text-red-700",
  },
  medium: {
    wrap: "border-amber-500 bg-amber-50",
    icon: "bg-amber-500",
    iconColor: "text-white",
    title: "text-amber-700",
  },
  low: {
    wrap: "border-emerald-500 bg-emerald-50",
    icon: "bg-emerald-500",
    iconColor: "text-white",
    title: "text-emerald-700",
  },
};

export function VerdictBanner({
  verdict,
  title,
  text,
}: {
  verdict: Verdict;
  title: string;
  text: string;
}) {
  const s = STYLES[verdict];
  const Icon = verdict === "high" ? BellRing : verdict === "medium" ? AlertTriangle : CheckCircle2;
  return (
    <div
      className={cn(
        "flex items-center gap-4 rounded-xl border-l-4 p-5",
        s.wrap
      )}
    >
      <div className={cn("grid h-12 w-12 flex-shrink-0 place-items-center rounded-lg", s.icon)}>
        <Icon className={cn("h-6 w-6", s.iconColor)} />
      </div>
      <div className="flex-1">
        <div className={cn("text-lg font-bold leading-tight", s.title)}>{title}</div>
        <div className="mt-1 text-sm text-foreground/70">{text}</div>
      </div>
    </div>
  );
}

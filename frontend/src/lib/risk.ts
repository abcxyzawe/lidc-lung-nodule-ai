import type { Risk, BrockBand } from "./types";

export function riskBadgeClass(risk: Risk | BrockBand): string {
  switch (risk) {
    case "high":
      return "bg-red-100 text-red-700 hover:bg-red-100";
    case "medium":
      return "bg-amber-100 text-amber-700 hover:bg-amber-100";
    case "low":
      return "bg-emerald-100 text-emerald-700 hover:bg-emerald-100";
    case "review":
      return "bg-sky-100 text-sky-700 hover:bg-sky-100";
    default:
      return "";
  }
}

export function rowTintClass(risk: Risk): string {
  switch (risk) {
    case "high":
      return "bg-red-50/40";
    case "medium":
      return "bg-amber-50/30";
    case "review":
      return "bg-sky-50/30";
    default:
      return "";
  }
}

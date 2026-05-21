import type { ModelKey } from "./types";

export const MODEL_META: Record<
  ModelKey,
  {
    label: string;
    sub: string;
    badgeClass: string; // tailwind classes for inline badge (bg + text + optional border)
  }
> = {
  mine: {
    label: "Mô hình của mình",
    sub: "Stage2+FPR_v3, F1=0.618",
    badgeClass: "bg-blue-100 text-blue-700 border-blue-300",
  },
  monai: {
    label: "MONAI RetinaNet 3D",
    sub: "Pretrained trên LUNA16",
    badgeClass: "bg-purple-100 text-purple-700 border-purple-300",
  },
  gt: {
    label: "Ground truth (LIDC)",
    sub: "Đáp án LIDC (XML)",
    badgeClass: "bg-emerald-100 text-emerald-700 border-emerald-300",
  },
};

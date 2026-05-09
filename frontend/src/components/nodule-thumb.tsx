"use client";

import { useState } from "react";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { API_URL } from "@/lib/api";
import type { Nodule } from "@/lib/types";

export function NoduleThumb({ n }: { n: Nodule }) {
  const [open, setOpen] = useState(false);
  if (!n.thumb_url) return <span className="text-muted-foreground">—</span>;
  const url = `${API_URL}${n.thumb_url}`;
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="block h-20 w-20 overflow-hidden rounded border transition hover:ring-2 hover:ring-primary"
        title="Click để phóng to"
      >
        <img src={url} alt={`Nodule ${n.id}`} className="h-full w-full object-cover" />
      </button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-3xl">
          <DialogTitle>
            Nodule #{n.id} — {n.diameter_mm.toFixed(1)} mm ({n.nodule_type ?? "solid"})
          </DialogTitle>
          <img
            src={url}
            alt={`Nodule ${n.id}`}
            className="mx-auto max-h-[70vh] w-auto rounded border"
          />
          <p className="text-center text-xs text-muted-foreground">
            Slice axial qua centroid · Hộp đỏ = vùng AI khoanh · vị trí (z, y, x) ={" "}
            {n.centroid_zyx_voxel.map((v) => Math.round(v)).join(", ")}
          </p>
        </DialogContent>
      </Dialog>
    </>
  );
}

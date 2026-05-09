import type { AnalyzeResponse, CaseDetail, CaseListItem, PatientForm } from "./types";

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8081";

export async function listCases(): Promise<CaseListItem[]> {
  const r = await fetch(`${API_URL}/api/cases`, { cache: "no-store" });
  if (!r.ok) throw new Error("Cannot fetch cases");
  return r.json();
}

export async function getCase(id: string): Promise<CaseDetail> {
  const r = await fetch(`${API_URL}/api/case/${id}`, { cache: "no-store" });
  if (!r.ok) throw new Error("Case not found");
  return r.json();
}

export async function deleteCase(id: string): Promise<void> {
  const r = await fetch(`${API_URL}/api/case/${id}`, { method: "DELETE" });
  if (!r.ok) throw new Error("Delete failed");
}

export function uploadAndAnalyze(
  files: FileList,
  patient: PatientForm,
  onUploadProgress: (loaded: number, total: number) => void
): Promise<AnalyzeResponse> {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    for (const f of Array.from(files)) {
      // preserve folder structure
      // @ts-expect-error webkitRelativePath exists on directory uploads
      fd.append("files", f, f.webkitRelativePath || f.name);
    }
    for (const [k, v] of Object.entries(patient)) {
      if (typeof v === "boolean") {
        if (v) fd.append(k, "true");
      } else {
        fd.append(k, String(v));
      }
    }

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/api/analyze`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onUploadProgress(e.loaded, e.total);
    };
    xhr.onload = () => {
      if (xhr.status === 200) resolve(JSON.parse(xhr.responseText));
      else reject(new Error(xhr.responseText || `HTTP ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error("Network error"));
    xhr.send(fd);
  });
}

export function streamProgress(
  caseId: string,
  onEvent: (pct: number, msg: string) => void,
  onDone: (pct: number) => void
): EventSource {
  const es = new EventSource(`${API_URL}/api/events/${caseId}`);
  es.onmessage = (e) => {
    const data = JSON.parse(e.data);
    onEvent(data.pct, data.msg);
    if (data.pct === 100 || data.pct === -1) {
      es.close();
      onDone(data.pct);
    }
  };
  es.onerror = () => {
    /* server may close after DONE */
  };
  return es;
}

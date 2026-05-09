// API response types — match backend in src/webapp/

export type Risk = "low" | "medium" | "high";

export interface LungRads {
  category: string;
  label: string;
  size_risk: Risk;
}

export interface Nodule {
  id: number;
  voxels: number;
  core_voxels?: number;
  volume_mm3: number;
  diameter_mm: number;
  diameter_full_mm?: number;
  centroid_zyx_voxel: [number, number, number];
  bbox_zyx_voxel: [number, number, number, number, number, number];
  nodule_type?: "solid" | "part-solid" | "non-solid";
  upper_lobe?: boolean;
  lung_rads?: LungRads;
  confidence?: number;     // 0..1, mean prob inside blob
  thumb_url?: string;
}

export interface CaseMeta {
  id: string;
  name: string;
  timestamp: string;
  n_slices: number;
  n_nodules: number;
  voxel_spacing_zyx_mm: [number, number, number];
  lung_voxels: number;
  pred_voxels: number;
  nodules: Nodule[];
  timing_sec: Record<string, number>;
}

export interface CaseListItem {
  id: string;
  name: string;
  n_nodules: number;
  n_slices: number;
  ts: string;
}

export interface CaseDetail {
  meta: CaseMeta;
  plot_html: string;
}

export interface AnalyzeResponse {
  case_id: string;
  name: string;
  n_dicoms: number;
}

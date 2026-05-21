// API response types — match backend in src/webapp/

export type Risk = "low" | "medium" | "high" | "review";

export type BrockBand = "low" | "medium" | "high" | "review";

export interface Diagnosis {
  action_band: "urgent" | "soon" | "routine" | "none";
  action_title: string;
  summary: string;
  action: string;
  nodule_buckets: {
    actionable: number;
    monitor: number;
    incidental: number;
  };
  max_diameter_mm: number;
  max_brock_pct: number;
  index_nodule_id?: number | null;
  index_reason?: string;
  risk_factors: Array<{ factor: string; weight: "low" | "medium" | "high" }>;
}

export interface ClinicalAssessment {
  patient: {
    age: number;
    sex: "male" | "female";
    pack_years: number;
    currently_smoking: boolean;
    years_since_quit?: number;
    family_hx: boolean;
    emphysema: boolean;
  };
  uspstf: { eligible: boolean };
}

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
  // AI scoring fields (mine/monai modes)
  risk_combined?: Risk;
  ai_class?: number;
  ai_susp_prob?: number;
  brock_prob?: number;
  // GT consensus tier fields (gt mode, F-3)
  confidence_tier?: "high" | "low";
  n_readers?: number;
  malignancy_score?: number | null;
  malignant?: boolean | null;
}

export interface CaseMeta {
  id: string;
  name: string;
  model?: "mine" | "monai" | "gt";
  timestamp: string;
  n_slices: number;
  n_nodules: number;
  voxel_spacing_zyx_mm: [number, number, number];
  lung_voxels: number;
  pred_voxels: number;
  nodules: Nodule[];
  timing_sec: Record<string, number>;
  model_confidence_warning?: boolean;
  low_pred_reason?: "no_candidate" | "low_pred_voxels" | "no_candidate_and_low_pred_voxels" | null;
}

export type ModelKey = "mine" | "monai" | "gt";

export interface CaseListItem {
  id: string;
  name: string;
  model?: ModelKey;
  n_nodules: number;
  n_slices: number;
  ts: string;
}

export interface CaseDetail {
  meta: CaseMeta;
  plot_html: string;
  verdicts?: Record<string, "accept" | "reject" | "review">;
}

export interface AnalyzeResponse {
  case_id: string;
  name: string;
  n_dicoms: number;
}

export interface TrainingMetrics {
  stage2: { epoch: number; loss: number; val_dice: number; is_best: boolean }[];
  fpr: {
    epoch: number;
    loss: number;
    val_acc: number;
    val_bal_acc: number;
    val_susp_f1: number;
    is_best: boolean;
  }[];
  summary: {
    stage2_best_epoch: number;
    stage2_best_val_dice: number;
    stage2_total_epochs: number;
    fpr_best_epoch: number;
    fpr_auc: number;
    fpr_best_thr: number;
    fpr_total_epochs: number;
    test_acc: number | null;
    test_bal_acc: number | null;
    test_susp_f1: number | null;
    mine_f1_test_panel: number;
  };
}

// API response types — match backend in src/webapp/

export type Risk = "low" | "medium" | "high" | "review";
export type BrockBand = "low" | "medium" | "high";
export type SymptomLevel = "low" | "medium" | "high";

export interface LungRads {
  category: string;
  label: string;
  size_risk: Risk;
}

export interface Nodule {
  id: number;
  voxels: number;
  volume_mm3: number;
  diameter_mm: number;
  centroid_zyx_voxel: [number, number, number];
  bbox_zyx_voxel: [number, number, number, number, number, number];
  ai_class?: number; // 1..5
  ai_expected?: number;
  ai_susp_prob?: number;
  ai_risk_label?: Risk;
  lung_rads?: LungRads;
  risk_combined?: Risk;
  nodule_type?: "solid" | "part-solid" | "non-solid";
  upper_lobe?: boolean;
  brock_prob?: number;
  brock_band?: BrockBand;
  thumb_url?: string;
}

export interface Patient {
  age: number;
  sex: "male" | "female";
  pack_years: number;
  currently_smoking: boolean;
  years_since_quit: number;
  family_hx: boolean;
  emphysema: boolean;
  symptoms: Record<string, boolean>;
}

export interface ActiveSymptom {
  key: string;
  label: string;
  weight: SymptomLevel;
}

export type ActionBand = "urgent" | "soon" | "routine" | "none";

export interface RiskFactor {
  factor: string;
  weight: "low" | "medium" | "high";
}

export interface Diagnosis {
  summary: string;
  action: string;
  action_band: ActionBand;
  action_title: string;
  nodule_buckets: { actionable: number; monitor: number; incidental: number };
  risk_factors: RiskFactor[];
  max_brock_pct: number;
  max_diameter_mm: number;
  index_nodule_id: number | null;
  index_reason: string;
}

export interface ClinicalAssessment {
  patient: Patient;
  uspstf: { eligible: boolean; message: string };
  symptoms: { level: SymptomLevel; message: string; active: ActiveSymptom[]; count: number };
  diagnosis?: Diagnosis;
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
  clinical?: ClinicalAssessment;
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

export interface ProgressEvent {
  pct: number;
  msg: string;
}

export interface PatientForm {
  age: number;
  sex: "male" | "female";
  pack_years: number;
  currently_smoking: boolean;
  years_since_quit: number;
  family_hx: boolean;
  emphysema: boolean;
  sym_hemoptysis: boolean;
  sym_weight_loss: boolean;
  sym_clubbing: boolean;
  sym_hoarseness: boolean;
  sym_cough: boolean;
  sym_chest_pain: boolean;
  sym_dyspnea: boolean;
  sym_recurrent_infection: boolean;
}

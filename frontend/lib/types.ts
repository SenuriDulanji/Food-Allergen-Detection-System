// ─── Scan API Types ───────────────────────────────────────────────────────────

export interface AllergenMatch {
  allergen_category: string;
  triggered_by: string[];
  source: string;
}

export interface ClinicalAlert {
  ingredient: string;
  model_key: string;
  risk_level: number;
  confidence: string;
  msg: string;
}

export interface SafetyNetWarning {
  trigger: string;
  linked_risks: string[];
  msg: string;
}

export interface MLRiskReport {
  clinical_alerts: ClinicalAlert[];
  safety_net_warnings: SafetyNetWarning[];
  models_evaluated: number;
  ingredients_checked: number;
}

export interface ScanResponse {
  identified_dish: string;
  confidence: "high" | "medium" | "low";
  ingredients: string[];
  rag_context_used: boolean;
  detected_allergens: AllergenMatch[];
  is_safe: boolean;
  safety_message: string;
  llm_explanation: string;
  ml_risk_report: MLRiskReport;
  llm_raw_dish_guess: string | null;
}

// ─── User API Types ───────────────────────────────────────────────────────────

export interface UserResponse {
  id: number;
  email: string;
  full_name: string | null;
  age: number | null;
  gender: string | null;
  province: string | null;
  blood_type: string | null;
  dietary_pattern: string | null;
  medical_conditions: string[] | null;
  lactose_intolerance: boolean | null;
  outside_food_frequency: number | null;
  personal_allergy_history: boolean | null;
  work_env: string | null;
  family_has_history: boolean | null;
  family_asthma: boolean | null;
  family_eczema: boolean | null;
  family_allergies: string[] | null;
  is_active: boolean;
  allergens: string[];
  created_at: string;
  updated_at: string;
}

export interface UserCreate {
  email: string;
  password: string;
  full_name?: string;
  age?: number;
  gender?: string;
  province?: string;
  blood_type?: string;
  dietary_pattern?: string;
  medical_conditions?: string[];
  lactose_intolerance?: boolean;
  outside_food_frequency?: number;
  personal_allergy_history?: boolean;
  work_env?: string;
  family_has_history?: boolean;
  family_asthma?: boolean;
  family_eczema?: boolean;
  family_allergies?: string[];
  allergens?: string[];
}

export interface UserUpdate {
  email?: string;
  password?: string;
  full_name?: string;
  age?: number;
  gender?: string;
  province?: string;
  blood_type?: string;
  dietary_pattern?: string;
  medical_conditions?: string[];
  lactose_intolerance?: boolean;
  outside_food_frequency?: number;
  personal_allergy_history?: boolean;
  work_env?: string;
  family_has_history?: boolean;
  family_asthma?: boolean;
  family_eczema?: boolean;
  family_allergies?: string[];
  allergens?: string[];
}

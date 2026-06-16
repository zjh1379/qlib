export interface RiskFlag {
  type: string;
  severity: 'high' | 'medium' | 'low';
  reason: string;
  source: string;
  source_date: string;
  verified: boolean; // guardrail: source matched a provided news/notice + valid date
}

export interface AiAnalysis {
  interpretation: string;
  risk_flags?: RiskFlag[];
  stance: 'favorable' | 'neutral' | 'caution';
  model: string;
  as_of_date: string;
  status: string; // ok | partial | failed
  adjustments?: string[]; // guardrail interventions (audit) — optional: backend defaults to []
  news_count: number; // provenance: news items fed the model
  notice_count: number; // provenance: announcements fed the model
}

export interface AnalysisJob {
  job_id: string;
  status: 'running' | 'done' | 'failed';
  started_at: string;
  finished_at: string | null;
  analyzed: number | null;
  as_of_date: string | null;
  error: string | null;
  reason: string | null;
}

export interface AnalysisStatus {
  last_run_at: string | null;
  last_success_at: string | null;
  last_error: string | null;
  is_running: boolean;
}

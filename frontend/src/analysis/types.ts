export interface RiskFlag {
  type: string;
  severity: 'high' | 'medium' | 'low';
  reason: string;
  source: string;
  source_date: string;
}

export interface AiAnalysis {
  interpretation: string;
  risk_flags?: RiskFlag[];
  stance: 'favorable' | 'neutral' | 'caution';
  model: string;
  as_of_date: string;
  status: string; // ok | partial | failed
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

export type EvidenceSupportStatus = "supported" | "unsupported" | "unknown";

export interface ClaimEvidenceAssessment {
  status: EvidenceSupportStatus;
  support_density: number;
  unsupported_fraction: number;
  contradiction_fraction: number;
  evidence_commitment: string;
}

export type FileCriticality = "normal" | "important" | "critical" | "safety";

export class TrustEngine {
  constructor(profile?: "rag" | "qa" | "summarization" | "dialogue" | "fact_check" | "default" | string);
  assessClaim(evidence: string, claim: string): ClaimEvidenceAssessment;
  fileCriticality(path: string): FileCriticality;
  hasSafetySignal(content: string): boolean;
}

export type Mode = 'update-render' | 'style-transfer' | 'new-angle';

export interface JobResult {
  jobId: string;
  status: 'pending' | 'processing' | 'done' | 'error';
  outputUrl?: string;
  error?: string;
}

export interface InpaintRequest {
  baseImageUrl: string;
  maskDataUrl: string;
  prompt: string;
}

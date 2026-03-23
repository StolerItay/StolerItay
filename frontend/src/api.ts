import axios from 'axios';

const BASE = '/api';

export async function submitUpdateRender(
  renderFile: File,
  massFile: File,
  prompt: string,
  model: string,
  apiKey?: string
): Promise<{ jobId: string }> {
  const form = new FormData();
  form.append('render', renderFile);
  form.append('mass', massFile);
  form.append('prompt', prompt);
  form.append('model', model);
  if (apiKey) form.append('replicate_api_token', apiKey);
  const { data } = await axios.post(`${BASE}/update-render`, form);
  return data;
}

export async function submitStyleTransfer(
  referenceFile: File,
  massFile: File,
  prompt: string,
  model: string,
  apiKey?: string,
  originalMassFile?: File | null,
): Promise<{ jobId: string }> {
  const form = new FormData();
  form.append('reference', referenceFile);
  form.append('mass', massFile);
  form.append('prompt', prompt);
  form.append('model', model);
  if (apiKey) form.append('replicate_api_token', apiKey);
  if (originalMassFile) form.append('original_mass', originalMassFile);
  const { data } = await axios.post(`${BASE}/style-transfer`, form);
  return data;
}

export async function submitNewAngle(
  renderFile: File,
  referenceFile: File | null,
  anglePrompt: string,
  stylePrompt: string,
  model: string,
  apiKey?: string
): Promise<{ jobId: string }> {
  const form = new FormData();
  form.append('render', renderFile);
  if (referenceFile) form.append('reference', referenceFile);
  form.append('angle_prompt', anglePrompt);
  form.append('style_prompt', stylePrompt);
  form.append('model', model);
  if (apiKey) form.append('replicate_api_token', apiKey);
  const { data } = await axios.post(`${BASE}/new-angle`, form);
  return data;
}

export async function submitInpaint(
  baseImage: File | string,
  maskDataUrl: string,
  prompt: string,
  model: string,
  apiKey?: string
): Promise<{ jobId: string }> {
  const form = new FormData();
  if (typeof baseImage === 'string') {
    form.append('base_image_url', baseImage);
  } else {
    form.append('base_image', baseImage);
  }
  form.append('mask_data_url', maskDataUrl);
  form.append('prompt', prompt);
  form.append('model', model);
  if (apiKey) form.append('replicate_api_token', apiKey);
  const { data } = await axios.post(`${BASE}/inpaint`, form);
  return data;
}

export async function pollJob(jobId: string): Promise<{
  status: string;
  output_url?: string;
  error?: string;
}> {
  const { data } = await axios.get(`${BASE}/job/${jobId}`);
  return data;
}

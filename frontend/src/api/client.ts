import type { FileUploadResponse, TaskStatusResponse } from './types';

const API_TIMEOUT = 90_000;
const API_TIMEOUT_DOWNLOAD = 180_000;

const isGHPagesBuild = import.meta.env.VITE_GHPAGES_BUILD === 'true';
const API_BASE_URL = isGHPagesBuild ? 'http://185.135.83.219/' : '/';
console.log({ isGHPagesBuild, API_BASE_URL });

interface ApiError extends Error {
  status?: number;
  message: string;
}

const handleResponse = async <T>(response: Response): Promise<T> => {
  if (!response.ok) {
    const errorText = await response.text();
    const error: ApiError = new Error(`HTTP error! status: ${response.status}`);
    error.status = response.status;
    if (errorText) {
      error.message += ` - ${errorText}`;
    }
    throw error;
  }
  const data = await response.json();
  return data as T;
};

export async function uploadFile(file: File): Promise<FileUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}api/upload/`, {
    method: 'POST',
    body: formData,
    headers: {},
  });

  return handleResponse<FileUploadResponse>(response);
}

export async function getTaskStatus(
  taskId: string
): Promise<TaskStatusResponse> {
  const response = await fetch(`${API_BASE_URL}api/status/${taskId}`, {
    signal: AbortSignal.timeout(API_TIMEOUT),
  });

  return handleResponse<TaskStatusResponse>(response);
}

export async function downloadResult(taskId: string): Promise<Blob> {
  const response = await fetch(`${API_BASE_URL}api/download/${taskId}`, {
    signal: AbortSignal.timeout(API_TIMEOUT_DOWNLOAD),
  });

  if (!response.ok) {
    const errorText = await response.text();
    const error: ApiError = new Error(`HTTP error! status: ${response.status}`);
    error.status = response.status;
    if (response.status === 500 && errorText) {
      error.message += ` - ${errorText}`;
    }
    throw error;
  }

  return response.blob();
}

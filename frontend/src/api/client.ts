import type { FileUploadResponse, TaskStatusResponse } from './types';

const API_TIMEOUT = 30000;
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

  const response = await fetch(`/api/upload/`, {
    method: 'POST',
    body: formData,
    headers: {},
  });

  return handleResponse<FileUploadResponse>(response);
}

export async function getTaskStatus(
  taskId: string
): Promise<TaskStatusResponse> {
  const response = await fetch(`/api/status/${taskId}`, {
    signal: AbortSignal.timeout(API_TIMEOUT),
  });

  return handleResponse<TaskStatusResponse>(response);
}

export async function downloadResult(taskId: string): Promise<Blob> {
  const response = await fetch(`/api/download/${taskId}`, {
    signal: AbortSignal.timeout(API_TIMEOUT),
  });

  if (!response.ok) {
    const errorText = await response.text();
    const error: ApiError = new Error(`HTTP error! status: ${response.status}`);
    error.status = response.status;
    if (errorText) {
      error.message += ` - ${errorText}`;
    }
    throw error;
  }

  return response.blob();
}

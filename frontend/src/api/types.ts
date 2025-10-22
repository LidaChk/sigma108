export interface UploadResponse {
  task_id: string;
  status: string;
}

export interface TaskStatusResponse {
  task_id: string;
  status: 'created' | 'processing' | 'completed' | 'failed';
}

export type TaskStatus = 'created' | 'processing' | 'completed' | 'failed';

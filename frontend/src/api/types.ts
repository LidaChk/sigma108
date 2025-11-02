export interface FileUploadResponse {
  task_id: string;
  status: TaskStatus;
  error?: string;
}

export interface TaskStatusResponse {
  task_id: string;
  status: TaskStatus;
  progress?: number;
  error?: string;
}

export type TaskStatus =
  | 'created'
  | 'processing'
  | 'completed'
  | 'failed'
  | 'error';

import { useCallback } from 'react';

const TASK_ID_STORAGE_KEY = 'sigma108_exam_grader_task_id';

export function useTaskIdPersistence() {
  const saveTaskId = useCallback((taskId: string) => {
    try {
      localStorage.setItem(TASK_ID_STORAGE_KEY, taskId);
    } catch (error) {
      console.warn('Не удалось сохранить taskId в localStorage:', error);
    }
  }, []);

  const getTaskId = useCallback((): string | null => {
    try {
      return localStorage.getItem(TASK_ID_STORAGE_KEY);
    } catch (error) {
      console.warn('Не удалось получить taskId из localStorage:', error);
      return null;
    }
  }, []);

  const clearTaskId = useCallback(() => {
    try {
      localStorage.removeItem(TASK_ID_STORAGE_KEY);
    } catch (error) {
      console.warn('Не удалось удалить taskId из localStorage:', error);
    }
  }, []);

  const hasStoredTaskId = useCallback((): boolean => {
    return getTaskId() !== null;
  }, [getTaskId]);

  return {
    saveTaskId,
    getTaskId,
    clearTaskId,
    hasStoredTaskId,
  };
}

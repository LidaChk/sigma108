
import { Alert, Progress, Text } from '@mantine/core';
import { IconInfoCircle } from '@tabler/icons-react';
import { useEffect, useState } from 'react';
import { getTaskStatus } from '../../api/client';
import type { TaskStatus as TaskStatusType } from '../../api/types';

interface TaskStatusProps {
  taskId: string;
  onTaskComplete: () => void;
  onTaskError: (error: string) => void;
}

export function TaskStatus({
  taskId,
  onTaskComplete,
  onTaskError,
}: TaskStatusProps) {
  const [status, setStatus] = useState<TaskStatusType>('created');
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const response = await getTaskStatus(taskId);
        setStatus(response.status);

        // Обновление прогресса в зависимости от статуса
        if (response.status === 'created') setProgress(10);
        if (response.status === 'processing') setProgress(50);
        if (response.status === 'completed') {
          setProgress(100);
          clearInterval(interval);
          onTaskComplete();
        }
        if (response.status === 'failed') {
          clearInterval(interval);
          onTaskError('Ошибка обработки файла');
        }
      } catch (error: unknown) {
        clearInterval(interval);
        onTaskError(`Ошибка получения статуса: ${(error as Error).message}`);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [taskId, onTaskComplete, onTaskError]);

  const statusText = {
    created: 'Файл загружен, ожидание начала обработки...',
    processing: 'Идет обработка файла...',
    completed: 'Обработка завершена!',
    failed: 'Ошибка обработки файла',
  };

  return (
    <div>
      <Text size="lg" mb="sm">
        {statusText[status]}
      </Text>
      <Progress value={progress} size="lg" radius="xl" mb="sm" />
      <Alert
        icon={<IconInfoCircle size="1rem" />}
        title="Информация"
        color="blue"
        className="neo-alert"
      >
        Идентификатор задачи: {taskId}
      </Alert>
    </div>
  );
}

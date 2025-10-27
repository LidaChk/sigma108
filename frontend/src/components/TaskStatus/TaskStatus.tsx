import { Alert, Progress, Text } from '@mantine/core';
import { IconInfoCircle } from '@tabler/icons-react';
import { useEffect, useState } from 'react';
import { getTaskStatus } from '../../api/client';
import type {
  TaskStatusResponse,
  TaskStatus as TaskStatusType,
} from '../../api/types';

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
        const response: TaskStatusResponse = await getTaskStatus(taskId);
        if (response.error) {
          clearInterval(interval);
          onTaskError(response.error);
          return;
        }
        setStatus(response.status);

        // Используем progress из ответа, если доступен, иначе fallback на статус
        const newProgress =
          response.progress ??
          (response.status === 'created'
            ? 10
            : response.status === 'processing'
            ? 100
            : response.status === 'completed'
            ? 100
            : 0);
        setProgress(newProgress);

        if (response.status === 'completed') {
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
      <Progress
        value={progress}
        size="xl"
        radius="md"
        mb="sm"
        animated
        className="neo-progress"
      />
      <Alert
        icon={<IconInfoCircle size="1rem" />}
        title="Информация"
        color="blue"
        className="neo-alert"
      >
        <Text size="sm" c="dark">
          Идентификатор задачи: {taskId}
        </Text>
        <Text size="sm" c="dark">
          Статус: {statusText[status]}
        </Text>
        <Text size="xs" c="dimmed" mt="xs">
          Обновление статуса каждые 2 секунды.
        </Text>
      </Alert>
    </div>
  );
}

import {
  Alert,
  Button,
  Card,
  Container,
  Stack,
  Text,
  Title,
} from '@mantine/core';
import { Notifications } from '@mantine/notifications';
import { useState } from 'react';
import { uploadFile } from './api/client';
import { FileUpload } from './components/FileUpload/FileUpload';
import { Results } from './components/Results/Results';
import { TaskStatus } from './components/TaskStatus/TaskStatus';

type AppStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'error';

export default function App() {
  const [status, setStatus] = useState<AppStatus>('idle');
  const [taskId, setTaskId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload = async (file: File) => {
    setStatus('uploading');
    setError(null);

    try {
      const response = await uploadFile(file);
      setTaskId(response.task_id);
      setStatus('processing');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Неизвестная ошибка');
      setStatus('error');
    }
  };

  const handleTaskComplete = () => {
    setStatus('completed');
  };

  const handleTaskError = (errorMessage: string) => {
    setError(errorMessage);
    setStatus('error');
  };

  const handleNewUpload = () => {
    setStatus('idle');
    setTaskId(null);
    setError(null);
  };

  return (
    <Container size="md" mt="0" className="app-content" h="100%">
      <Stack align="center" justify="center" h="100%" gap="xl">
        <Notifications />

        <Card shadow="md" p="xl" radius="md" withBorder className="neo-card">
          <Stack>
            <Title
              order={1}
              mb="md"
              ta="center"
              c="myColor.7"
              className="neo-text-balanced"
            >
              Автоматическая оценка ответов на экзамене
            </Title>
            <Text mb="xl" ta="center" c="dimmed" className="neo-text-balanced">
              Загрузите CSV файл с ответами для автоматической оценки
            </Text>
          </Stack>
          {status === 'idle' && (
            <FileUpload onFileUpload={handleFileUpload} isUploading={false} />
          )}

          {status === 'uploading' && (
            <FileUpload onFileUpload={handleFileUpload} isUploading={true} />
          )}

          {status === 'processing' && taskId && (
            <TaskStatus
              taskId={taskId}
              onTaskComplete={handleTaskComplete}
              onTaskError={handleTaskError}
            />
          )}

          {status === 'completed' && taskId && (
            <Results taskId={taskId} onNewUpload={handleNewUpload} />
          )}

          {status === 'error' && error && (
            <Alert
              title="Ошибка"
              color="red"
              variant="filled"
              radius="md"
              className="neo-alert"
            >
              {error}
              <Button
                onClick={handleNewUpload}
                mt="md"
                fullWidth
                variant="white"
                className="neo-button-filled"
              >
                Попробовать снова
              </Button>
            </Alert>
          )}
        </Card>
      </Stack>
    </Container>
  );
}

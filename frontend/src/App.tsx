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
import { IconRefresh } from '@tabler/icons-react';
import { useReducer } from 'react';
import { uploadFile } from './api/client';
import type { FileUploadResponse } from './api/types';
import { FileUpload } from './components/FileUpload/FileUpload';
import { Results } from './components/Results/Results';
import { TaskStatus } from './components/TaskStatus/TaskStatus';

type AppStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'error';

type AppState = {
  status: AppStatus;
  taskId: string | null;
  error: string | null;
};

type AppAction =
  | { type: 'SET_STATUS'; payload: AppStatus }
  | { type: 'SET_TASK_ID'; payload: string }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'RESET' };

const initialState: AppState = {
  status: 'idle',
  taskId: null,
  error: null,
};

const reducer = (state: AppState, action: AppAction): AppState => {
  switch (action.type) {
    case 'SET_STATUS':
      return { ...state, status: action.payload };
    case 'SET_TASK_ID':
      return { ...state, taskId: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    case 'RESET':
      return initialState;
    default:
      return state;
  }
};

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);

  const handleFileUpload = async (file: File) => {
    dispatch({ type: 'SET_STATUS', payload: 'uploading' });
    dispatch({ type: 'SET_ERROR', payload: null });

    try {
      const response: FileUploadResponse = await uploadFile(file);
      if (response.error) {
        throw new Error(response.error);
      }
      dispatch({ type: 'SET_TASK_ID', payload: response.task_id });
      dispatch({ type: 'SET_STATUS', payload: 'processing' });
    } catch (err: unknown) {
      const errorMessage =
        err instanceof Error ? err.message : 'Неизвестная ошибка';
      dispatch({ type: 'SET_ERROR', payload: errorMessage });
      dispatch({ type: 'SET_STATUS', payload: 'error' });
    }
  };

  const handleTaskComplete = () => {
    dispatch({ type: 'SET_STATUS', payload: 'completed' });
  };

  const handleTaskError = (errorMessage: string) => {
    dispatch({ type: 'SET_ERROR', payload: errorMessage });
    dispatch({ type: 'SET_STATUS', payload: 'error' });
  };

  const handleNewUpload = () => {
    dispatch({ type: 'RESET' });
  };

  return (
    <Container size="md" mt="0" className="app-content" h="100%">
      <Stack align="center" justify="center" h="100%" gap="xl">
        <Notifications />

        <Card shadow="md" p="xl" radius="md" withBorder className="neo-card">
          <Card.Section>
            <Title
              order={1}
              p="md"
              ta="center"
              c="myColor.7"
              className="neo-text-balanced"
            >
              Автоматическая оценка ответов на экзамене
            </Title>
            {state.status === 'idle' && (
              <Text
                mb="md"
                ta="center"
                c="dark"
                size="md"
                className="neo-text-balanced"
              >
                Загрузите CSV файл с ответами для автоматической оценки
              </Text>
            )}
          </Card.Section>
          <Card.Section>
            {state.status === 'idle' && (
              <FileUpload onFileUpload={handleFileUpload} isUploading={false} />
            )}

            {state.status === 'uploading' && (
              <FileUpload onFileUpload={handleFileUpload} isUploading={true} />
            )}

            {state.status === 'processing' && state.taskId && (
              <TaskStatus
                taskId={state.taskId}
                onTaskComplete={handleTaskComplete}
                onTaskError={handleTaskError}
              />
            )}

            {state.status === 'completed' && state.taskId && (
              <Results taskId={state.taskId} onNewUpload={handleNewUpload} />
            )}

            {state.status === 'error' && state.error && (
              <Stack mt="md" gap="md" align="center">
                <Alert
                  title="Ошибка"
                  color="red"
                  variant="filled"
                  radius="md"
                  className="neo-alert"
                  w="100%"
                >
                  <Text ta="start">{state.error}</Text>
                </Alert>
                <Button
                  onClick={handleNewUpload}
                  className="neo-button"
                  leftSection={<IconRefresh size={14} />}
                >
                  Попробовать снова
                </Button>
              </Stack>
            )}
          </Card.Section>
        </Card>
      </Stack>
    </Container>
  );
}

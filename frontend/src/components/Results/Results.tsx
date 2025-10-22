import { Alert, Button, Group } from '@mantine/core';
import { IconDownload, IconRefresh } from '@tabler/icons-react';
import { useState } from 'react';
import { downloadResult } from '../../api/client';

interface ResultsProps {
  taskId: string;
  onNewUpload: () => void;
}

export function Results({ taskId, onNewUpload }: ResultsProps) {
  const [isDownloading, setIsDownloading] = useState(false);

  const handleDownload = async () => {
    setIsDownloading(true);
    try {
      const blob = await downloadResult(taskId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `result_${taskId}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Ошибка скачивания:', error);
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div>
      <Alert title="Обработка завершена!" color="green">
        Файл успешно обработан. Вы можете скачать результаты.
      </Alert>

      <Group mt="md">
        <Button
          leftSection={<IconDownload size={14} />}
          onClick={handleDownload}
          loading={isDownloading}
        >
          Скачать результаты
        </Button>
        <Button
          variant="outline"
          leftSection={<IconRefresh size={14} />}
          onClick={onNewUpload}
        >
          Обработать новый файл
        </Button>
      </Group>
    </div>
  );
}

// components/FileUpload/FileUpload.tsx
import { Button, Group, rem, Text } from '@mantine/core';
import { Dropzone, type FileWithPath } from '@mantine/dropzone';
import { IconCheck, IconUpload, IconX } from '@tabler/icons-react';
import { useState } from 'react';
import './FileUpload.scss';

interface FileUploadProps {
  onFileUpload: (file: File) => void;
  isUploading: boolean;
}

export function FileUpload({ onFileUpload, isUploading }: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null);

  const handleDrop = (files: FileWithPath[]) => {
    if (files.length > 0) {
      setFile(files[0]);
    }
  };

  const handleUpload = () => {
    if (file) {
      onFileUpload(file);
    }
  };

  const getFileSize = (file: File) => {
    if (file.size < 1024) {
      return `${file.size} байт`;
    } else if (file.size < 1024 * 1024) {
      return `${(file.size / 1024).toFixed(1)} КБ`;
    } else {
      return `${(file.size / (1024 * 1024)).toFixed(1)} МБ`;
    }
  };

  return (
    <div className="file-upload-container">
      <Dropzone
        onDrop={handleDrop}
        onReject={(files) => console.log('rejected files', files)}
        maxSize={30 * 1024 * 1024}
        accept={['text/csv']}
        disabled={isUploading}
        className="file-upload-dropzone"
      >
        <Group
          justify="center"
          gap="xl"
          mih={220}
          style={{ pointerEvents: 'none' }}
        >
          <Dropzone.Accept>
            <IconCheck
              className="file-upload-icon"
              style={{
                width: rem(52),
                height: rem(52),
                color: 'var(--mantine-color-blue-6)',
              }}
              stroke={1.5}
            />
          </Dropzone.Accept>
          <Dropzone.Reject>
            <IconX
              className="file-upload-icon"
              style={{
                width: rem(52),
                height: rem(52),
                color: 'var(--mantine-color-red-6)',
              }}
              stroke={1.5}
            />
          </Dropzone.Reject>
          <Dropzone.Idle>
            <IconUpload
              className="file-upload-icon"
              style={{
                width: rem(52),
                height: rem(52),
                color: 'var(--mantine-color-dimmed)',
              }}
              stroke={1.5}
            />
          </Dropzone.Idle>

          <div className="file-upload-text">
            <Text size="xl" inline>
              Перетащите CSV файл сюда или нажмите для выбора
            </Text>
            <Text size="sm" c="dimmed" inline mt={7}>
              Файл должен быть в формате CSV
            </Text>
          </div>
        </Group>
      </Dropzone>

      {file && (
        <Group mt="md">
          <div className="file-upload-file-info">
            <div>
              <Text size="sm" className="file-upload-file-name">
                {file.name}
              </Text>
              <Text size="xs" className="file-upload-file-size">
                {getFileSize(file)}
              </Text>
            </div>
            <Button
              onClick={handleUpload}
              loading={isUploading}
              className="file-upload-button"
            >
              {isUploading ? 'Загрузка...' : 'Загрузить файл'}
            </Button>
          </div>
        </Group>
      )}
    </div>
  );
}

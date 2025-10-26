import { Button, Group, rem, Stack, Text } from '@mantine/core';
import { Dropzone, type FileWithPath } from '@mantine/dropzone';
import { IconCheck, IconCloudUpload, IconX } from '@tabler/icons-react';
import { useState } from 'react';

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

  const handleFileInput = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
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
    <div>
      <Dropzone
        variant="filled"
        onDrop={handleDrop}
        onReject={(files) => console.log('rejected files', files)}
        accept={['text/csv']}
        disabled={isUploading}
        className="neobrutal-dropzone"
      >
        <Stack
          justify="center"
          align="center"
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
            <IconCloudUpload
              style={{
                width: rem(52),
                height: rem(52),
                color: 'var(--mantine-color-myColor-5)',
              }}
              stroke={1.5}
            />
          </Dropzone.Idle>

          <div className="file-upload-text">
            <Text size="xl" ta="center" inline className="neo-text-balanced">
              Перетащите CSV файл сюда или нажмите для выбора
            </Text>
            <Text size="sm" ta="center" c="dark" inline mt={7}>
              Файл должен быть в формате CSV
            </Text>
          </div>
        </Stack>
      </Dropzone>

      <input
        type="file"
        id="fileInput"
        onChange={handleFileInput}
        accept=".csv"
        style={{ display: 'none' }}
        className="neo-input"
      />

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
              className="file-upload-button neo-button-filled"
            >
              {isUploading ? 'Загрузка...' : 'Загрузить файл'}
            </Button>
          </div>
        </Group>
      )}
    </div>
  );
}

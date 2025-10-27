import {
  Box,
  Button,
  Group,
  Loader,
  LoadingOverlay,
  rem,
  Stack,
  Text,
} from '@mantine/core';
import { Dropzone, type FileWithPath } from '@mantine/dropzone';
import { IconCheck, IconCloudUpload, IconX } from '@tabler/icons-react';
import { useState } from 'react';

interface FileUploadProps {
  readonly onFileUpload: (file: File) => void;
  readonly isUploading: boolean;
}

const useFileUpload = (onFileUpload: (file: File) => void) => {
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

  return { file, setFile, handleDrop, handleFileInput, handleUpload };
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

export function FileUpload({ onFileUpload, isUploading }: FileUploadProps) {
  const { file, handleDrop, handleFileInput, handleUpload } =
    useFileUpload(onFileUpload);

  return (
    <Box pos="relative">
      <LoadingOverlay
        visible={isUploading}
        loaderProps={{
          children: <Loader size={30} />,
        }}
      />
      <Dropzone
        variant="filled"
        onDrop={handleDrop}
        onReject={(files) => console.log('rejected files', files)}
        accept={['text/csv']}
        disabled={isUploading}
        className="neo-dropzone"
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
        <Stack mt="md" gap="md" align="center">
          <Group justify="apart" w="100%" gap="0">
            <Text size="sm" className="file-name">
              {file.name}
            </Text>
            <div className="dots-spacer"></div>
            <Text size="xs" className="file-size">
              {getFileSize(file)}
            </Text>
          </Group>
          <Button
            onClick={handleUpload}
            loading={isUploading}
            className="neo-button"
          >
            Загрузить файл
          </Button>
        </Stack>
      )}
    </Box>
  );
}

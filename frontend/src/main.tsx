import { MantineProvider, createTheme } from '@mantine/core';
import '@mantine/core/styles.css';
import '@mantine/dropzone/styles.css';
import '@mantine/notifications/styles.css';
import './index.css';
import { createRoot } from 'react-dom/client';
import App from './App';

const theme = createTheme({
  primaryColor: 'blue',
  fontFamily: 'Arial, sans-serif',
});

const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(
    <MantineProvider theme={theme}>
      <App />
    </MantineProvider>
  );
} else {
  console.error('Не удалось найти элемент с id="root"');
}

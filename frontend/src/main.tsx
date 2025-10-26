import '@fontsource-variable/roboto-mono/wght.css';
import {
  MantineProvider,
  createTheme,
  type MantineColorsTuple,
} from '@mantine/core';
import '@mantine/core/styles.css';
import '@mantine/dropzone/styles.css';
import '@mantine/notifications/styles.css';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

const myColor: MantineColorsTuple = [
  '#e5f3ff',
  '#cde2ff',
  '#9ac2ff',
  '#5294ff',
  '#3882fe',
  '#1d70fe',
  '#0967ff',
  '#0057e4',
  '#004dcd',
  '#0042b5',
];

const successColor: MantineColorsTuple = [
  '#f0fdf4',
  '#dcfce7',
  '#bbf7d0',
  '#86efac',
  '#4ade80',
  '#22c55e',
  '#16a34a',
  '#15803d',
  '#166534',
  '#14532d',
];

const lightBlue: MantineColorsTuple = [
  '#e0f2fe',
  '#bae6fd',
  '#7dd3fc',
  '#38bdf8',
  '#0ea5e9',
  '#0284c7',
  '#0369a1',
  '#075985',
  '#0c4a6e',
  '#0e4262',
];

const theme = createTheme({
  colors: {
    myColor,
    success: successColor,
    lightBlue,
  },
  primaryColor: 'myColor',
  primaryShade: { light: 6, dark: 8 },
  fontFamily: 'Roboto Mono Variable, Roboto Mono, monospace, sans-serif',
  headings: {
    fontFamily:
      'Montserrat, sans-serifRoboto Mono Variable, Roboto Mono, monospace, sans-serif',
    fontWeight: '700',
  },
  autoContrast: true,
  luminanceThreshold: 0.3,
  shadows: {
    md: '4px 4px 0 #000',
  },
  radius: {
    md: '8px',
  },
  components: {
    Button: {
      defaultProps: {
        radius: 'md',
      },
    },
    Card: {
      defaultProps: {
        shadow: 'md',
        radius: 'md',
      },
    },
  },
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

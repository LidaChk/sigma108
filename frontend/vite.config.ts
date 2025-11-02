import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

function getMoscowTime() {
  const now = new Date();
  const moscowTime = new Date(
    now.toLocaleString('en-US', {
      timeZone: 'Europe/Moscow',
    })
  );

  return moscowTime.toLocaleString('ru-RU', {
    timeZone: 'Europe/Moscow',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

export default defineConfig({
  plugins: [react()],
  base: '/sigma108/',
  build: {
    outDir: 'dist',
  },
  define: {
    'import.meta.env.VITE_BUILD_TIME': JSON.stringify(getMoscowTime()),
  },
});

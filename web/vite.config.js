import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Served at ideitas.online/logprob-gui (static; the browser calls OpenRouter with the visitor's key).
export default defineConfig({
  base: '/logprob-gui/',
  plugins: [react()],
  build: { outDir: 'dist', assetsDir: 'assets', emptyOutDir: true, chunkSizeWarningLimit: 900 },
});

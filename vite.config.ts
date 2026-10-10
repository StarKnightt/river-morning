import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  // Sandbox copies share node_modules through a junction and pre-bundle dependencies for their own
  // root, so the cache must live outside node_modules or they overwrite each other's.
  cacheDir: '.vite-cache',
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 2000,
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
  },
});

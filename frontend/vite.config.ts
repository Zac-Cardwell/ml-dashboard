import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev: `npm run dev` proxies the API to the backend on :8000.
// Prod: `npm run build` -> dist/, served by FastAPI itself.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', ws: true },
      '/docs': 'http://127.0.0.1:8000',
      '/openapi.json': 'http://127.0.0.1:8000',
    },
  },
  build: { chunkSizeWarningLimit: 6000 },
})

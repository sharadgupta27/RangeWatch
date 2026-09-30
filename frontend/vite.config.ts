/// <reference types="vitest/config" />
import { tanstackRouter } from '@tanstack/router-plugin/vite'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'
import { defineConfig, type Plugin } from 'vite'

/** Unique id per production build, exposed to the app and published as /version.json so
 *  long-open tabs can detect that a newer dashboard has been deployed. */
const BUILD_ID = new Date().toISOString()

function versionFile(): Plugin {
  return {
    name: 'sdm-version-file',
    apply: 'build',
    generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: 'version.json',
        source: JSON.stringify({ build: BUILD_ID }),
      })
    },
  }
}

export default defineConfig({
  plugins: [
    // Must come before the React plugin: generates src/routeTree.gen.ts from src/routes/.
    tanstackRouter({ target: 'react', autoCodeSplitting: true }),
    react(),
    tailwindcss(),
    versionFile(),
  ],
  define: { __BUILD_ID__: JSON.stringify(BUILD_ID) },
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    port: 5173,
    // Mirror the production gateway so the app always uses same-origin relative URLs.
    proxy: {
      '/api': { target: 'http://localhost:8000', rewrite: (p) => p.replace(/^\/api/, '') },
      '/tiles/raster': {
        target: 'http://localhost:8001',
        rewrite: (p) => p.replace(/^\/tiles\/raster/, ''),
      },
      '/tiles/vector': {
        target: 'http://localhost:7800',
        rewrite: (p) => p.replace(/^\/tiles\/vector/, ''),
      },
    },
  },
  worker: { format: 'es' },
  build: {
    chunkSizeWarningLimit: 3000,
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (/maplibre|deck.gl|luma.gl|loaders.gl|editable-layers/.test(id)) return 'maps'
          if (/recharts|@visx|d3-/.test(id)) return 'charts'
          return undefined
        },
      },
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
})

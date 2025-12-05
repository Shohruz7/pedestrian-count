import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
  },
  build: {
    // Optimize bundle size
    rollupOptions: {
      output: {
        manualChunks(id) {
          // Manual chunking for better code splitting
          if (id.includes('node_modules')) {
            if (id.includes('react') || id.includes('react-dom')) {
              return 'react-vendor'
            }
            if (id.includes('@mui') || id.includes('@emotion')) {
              return 'mui-vendor'
            }
            if (id.includes('leaflet')) {
              return 'map-vendor'
            }
            if (id.includes('recharts')) {
              return 'chart-vendor'
            }
            return 'vendor'
          }
        }
      }
    },
    // Enable minification
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true, // Remove console.log in production
        drop_debugger: true,
      }
    },
    // Chunk size warnings
    chunkSizeWarningLimit: 1000,
  },
  // Optimize dependencies
  optimizeDeps: {
    include: ['react', 'react-dom', '@mui/material', 'leaflet', 'react-leaflet'],
    exclude: ['react-router-dom']
  }
})


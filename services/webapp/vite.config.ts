import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // Expose on all network interfaces
    port: 5173,
    strictPort: true,
    // Allow ngrok and any LAN hosts
    allowedHosts: ['all', 'hooded-gastronomical-merissa.ngrok-free.dev'],
    cors: true,
    // HMR config for ngrok tunneling
    hmr: {
      // For ngrok: use websocket through the tunnel
      clientPort: 443, // ngrok uses HTTPS (443)
      protocol: 'wss', // Secure websocket for ngrok
    },
  },
  // Allow loading from any origin (needed for ngrok)
  preview: {
    host: '0.0.0.0',
    port: 5173,
  },
})

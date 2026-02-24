import { defineConfig } from '@playwright/test';

export default defineConfig({
    testDir: './e2e',
    webServer: {
        command: 'npm run dev',
        port: 5173,
        timeout: 120 * 1000,
        reuseExistingServer: true,
    },
    use: {
        browserName: 'chromium',
    },
});

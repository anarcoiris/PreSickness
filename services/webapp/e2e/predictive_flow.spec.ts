import { test, expect } from '@playwright/test';

test.describe('Predictive Flow', () => {
    test('Visualizar Generación de Predicciones', async ({ page }) => {
        // 1. Login
        await page.goto('http://localhost:5173/login');
        await page.fill('input[type="email"]', 'patient1@em.com');
        await page.fill('input[type="password"]', 'test');
        await page.click('button[type="submit"]');

        // 2. Verificar Dashboard inicial
        await expect(page).toHaveURL(/.*dashboard/);
        await expect(page.locator('h1')).toContainText('EM-Predictor');

        // Captura visual del Dashboard con predicción inicial
        await page.screenshot({ path: 'e2e-screenshots/1-dashboard-prediction.png' });

        // 3. Navegar a Análisis Profundo (Modelo ML)
        await page.click('text=Modelo ML'); // NavLink en sibebar
        await expect(page).toHaveURL(/.*analysis/);

        // 4. Verificar carga de gráficas
        await expect(page.locator('.recharts-wrapper')).toBeVisible({ timeout: 10000 });
        await page.screenshot({ path: 'e2e-screenshots/2-model-analysis.png' });

        // 5. Validar detalles específicos
        await expect(page.locator('text=Mejor Modelo')).toBeVisible();
        const metric = await page.textContent('.stat-value'); // Example selector
        console.log('Métrica AUROC:', metric);
    });
});

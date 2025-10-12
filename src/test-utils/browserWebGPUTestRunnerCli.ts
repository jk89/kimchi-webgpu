import {
    bundleTests,
    findBrave,
    startServer,
    ROOT_DIR,
} from './browserTestRunnerUtils.js';
import path from 'path';
import puppeteer from 'puppeteer';

async function main() {
    const entryFile = path.resolve(ROOT_DIR, 'src/test-utils/index.ts');
    await bundleTests(entryFile);

    const { url } = await startServer();

    const brave = findBrave();
    console.log('Launching headless Brave for tests');

    const browser = await puppeteer.launch({
        headless: true,
        executablePath: brave,
        args: [
            '--enable-unsafe-webgpu',
            '--ignore-gpu-blocklist',
            '--enable-features=DefaultANGLEVulkan,Vulkan,VulkanFromANGLE',
            '--disable-features=PdfUseSkiaRenderer',
        ],
    });

    const page = await browser.newPage();

    page.on('console', async (msg) => {
        const args = await Promise.all(msg.args().map((a) => a.jsonValue()));
        console.log('[browser]', ...args);
    });

    await page.goto(url, { waitUntil: 'domcontentloaded' });

    await page.waitForFunction(
        () => (window as any).testsFinished === true,
        { polling: 100, timeout: 0 }
    );

    const failuresCount = await page.evaluate(
        () => (window as any).testsFailures || 0
    );

    console.log(`Tests finished. Failures: ${failuresCount}`);
    process.exit(failuresCount ? 1 : 0);
}

main().catch(console.error);

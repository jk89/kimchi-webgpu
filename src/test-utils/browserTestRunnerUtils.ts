import express from 'express';
import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import * as esbuild from 'esbuild';
import { fileURLToPath } from 'url';

export const __filename = fileURLToPath(import.meta.url);
export const __dirname = path.dirname(__filename);

// Root and public folder
export const ROOT_DIR = path.resolve(__dirname, '..', '..', '..');
export const PUBLIC_DIR = path.resolve(ROOT_DIR, 'public');

/** Find Brave Nightly executable */
export function findBrave(): string {
    try {
        return execSync(
            'which brave-browser-nightly || which brave-browser || which brave',
            { encoding: 'utf8' }
        ).trim();
    } catch {
        throw new Error('Brave not found! Install Brave Nightly.');
    }
}

/** Start Express server serving public folder */
export async function startServer(port = 3001) {
    const app = express();
    app.use(express.static(PUBLIC_DIR));

    return new Promise<{ server: any; url: string }>((resolve) => {
        const server = app.listen(port, () => {
            const url = `http://localhost:${port}/index.html`;
            console.log(`Server running at: ${url}. Please open this in brave with the nessesary flags.`);
            resolve({ server, url });
        });
    });
}

/** Bundle all tests into ESM for browser */
export async function bundleTests(
    entryFile: string,
    outFile = 'bundle.tests.js'
) {
    if (!fs.existsSync(PUBLIC_DIR))
        fs.mkdirSync(PUBLIC_DIR, { recursive: true });

    const outfile = path.resolve(PUBLIC_DIR, outFile);

    await esbuild.build({
        entryPoints: [entryFile],
        bundle: true,
        format: 'esm',
        outfile,
        platform: 'browser',
        define: { 'process.env.NODE_ENV': '"test"' },
    });

    // Create a minimal index.html if missing
    const htmlPath = path.resolve(PUBLIC_DIR, 'index.html');
    if (!fs.existsSync(htmlPath)) {
        const html = `<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>WebGPU Tests</title></head>
<body>
<h1>WebGPU Tests</h1>
<div id="test-results"></div>
<script type="module">
  import './${outFile}';
  window.addEventListener('DOMContentLoaded', () => {
    if (!window.runTests) {
      console.error('runTests not found on globalThis!');
      return;
    }
    window.runTests();
  });
</script>
</body>
</html>`;
        fs.writeFileSync(htmlPath, html, 'utf-8');
    }

    return outfile;
}

import { bundleTests, findBrave, startServer, ROOT_DIR } from './browserTestRunnerUtils.js';
import path from 'path';
import { spawn } from 'child_process';

async function main() {
  const entryFile = path.resolve(ROOT_DIR, 'src/tests/index.ts');
  await bundleTests(entryFile);

  const { url } = await startServer();
  const brave = findBrave();

  console.log('Opening Brave at', url);

  spawn(
    brave,
    [
      '--enable-unsafe-webgpu',
      '--ignore-gpu-blocklist',
      '--enable-features=DefaultANGLEVulkan,Vulkan,VulkanFromANGLE',
      '--disable-features=PdfUseSkiaRenderer',
      url,
    ],
    { stdio: 'inherit', detached: true }
  ).unref(); // unref allows Node to exit independently
}

main().catch(console.error);

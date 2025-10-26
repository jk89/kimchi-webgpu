import { pippengerMSMPallas } from './pippenger_msm.js';

describe('GPU pippengerMSMPallas 1K', function () {
    it('Test 1 thousand points/scalars', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();

        // Generate a large number of scalars and points
        const N = 1000;
        const scalars: bigint[] = [];
        const points: { x: bigint; y: bigint }[] = [];
        for (let i = 0; i < N; i++) {
            scalars.push(BigInt(i + 1)); // 1n, 2n, 3n, ...
            points.push({ x: BigInt(5 + i * 6), y: BigInt(7 + i * 6) }); // deterministic increasing sequence
        }

        // GPU timing
        const gpuStart = performance.now();
        const gpuResults = await pippengerMSMPallas(device, scalars, points, {bucketWidthBits: 4});
        const gpuEnd = performance.now();
        console.log(`GPU MSM took ${gpuEnd - gpuStart} ms`);
    });
});

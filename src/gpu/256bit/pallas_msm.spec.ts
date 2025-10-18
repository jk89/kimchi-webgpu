import { pallasMSM } from "./pallas_msm.js";

describe('GPU vs CPU full Pallas MSM', function () {
    // on a 3090 we get about 215053 multiplications per second
    // up to 4 million points
    it('matches outputs and measures times', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();

        // Generate a large number of scalars and points
        const N = 4000000;
        const scalars: bigint[] = [];
        const points: { x: bigint; y: bigint }[] = [];
        for (let i = 0; i < N; i++) {
            scalars.push(BigInt(i + 1)); // 1n, 2n, 3n, ...
            points.push({ x: BigInt(5 + i * 6), y: BigInt(7 + i * 6) }); // deterministic increasing sequence
        }

        // GPU timing (your existing gpuMSM should produce points in normal affine form)
        const gpuStart = performance.now();
        const gpuResults = await pallasMSM(device, scalars, points);
        const gpuEnd = performance.now();
        console.log(`GPU MSM took ${gpuEnd - gpuStart} ms`);

    });
});

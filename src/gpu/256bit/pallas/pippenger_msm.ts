import {
    pippengerShaderPassAProjectiveConversion as ShaderPassA,
    pippengerShaderPassBi1BucketScalarWeightedPointContribution as ShaderPassBi1,
    pippengerShaderPassBi2TreeReduceBucket as ShaderPassBi2,
    pippengerShaderPassCBucketAggregation as ShaderPassC,
} from './pippenger_msm.wgslc.js';
import {
    bigint256ToLimbs,
    BYTES_PER_ELEMENT_256,
    LIMBS_PER_ELEMENT_256,
} from '../helpers.js';
import { Point } from '../../../types/point.js';

// For now we will FIX the number of buckets at 128
// We need to account for batching of the N points
const buckets = 128;

export function pippengerMSMPallas(
    device: GPUDevice,
    scalars: bigint[],
    P: Point[]
): Point {
    const n = scalars.length;

    if (n === 0) {
        throw new Error('scalars and points arrays cannot be empty');
    }

    if (P.length !== n) {
        throw new Error(
            `scalars and points must have the same length: got ${n} scalars and ${P.length} points`
        );
    }

    // Create shaders

    const shaderPassAModule = device.createShaderModule({
        code: ShaderPassA,
    });
    const shaderPassBi1Module = device.createShaderModule({
        code: ShaderPassBi1,
    });
    const shaderPassBi2Module = device.createShaderModule({
        code: ShaderPassBi2,
    });
    const shaderPassCModule = device.createShaderModule({
        code: ShaderPassC,
    });

    // Create pipelines

    const shaderPassAPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderPassAModule,
            entryPoint: 'main',
        },
    });

    const shaderPassBi1Pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderPassBi1Module,
            entryPoint: 'main',
        },
    });

    const shaderPassBi2Pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderPassBi2Module,
            entryPoint: 'main',
        },
    });

    const shaderPassCPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderPassCModule,
            entryPoint: 'main',
        },
    });

    type ComputePasses = {
        pipeline: GPUComputePipeline;
        passBindGroups: { idx: number; bindGroup: GPUBindGroup }[][];
    };

    const computePassA: ComputePasses = {
        pipeline: shaderPassAPipeline,
        passBindGroups: [],
    };
    const computePassBi1: ComputePasses = {
        pipeline: shaderPassBi1Pipeline,
        passBindGroups: [],
    };
    const computePassBi2: ComputePasses = {
        pipeline: shaderPassBi2Pipeline,
        passBindGroups: [],
    };
    const computePassC: ComputePasses = {
        pipeline: shaderPassCPipeline,
        passBindGroups: [],
    };

    // Whats the max sized buffer supported.
    const maxBufferSize = device.limits.maxBufferSize;
    // This should be the number of elements we can fit in one buffer.
    const maxChunkN = maxBufferSize / BYTES_PER_ELEMENT_256;
    // This is the number of chunks needed to cover the whole set of input.
    const chunks = Math.ceil(n / maxChunkN);

    let inputsRemaining = n;

    // Prepare input data
    for (let chunkIdx = 0; chunkIdx < chunks; chunkIdx++) {
        const chunkOffset = chunkIdx * maxChunkN;
        const currentChunkN = Math.min(inputsRemaining, maxChunkN);
        const size = currentChunkN * BYTES_PER_ELEMENT_256;
        const k = new Uint32Array(currentChunkN * LIMBS_PER_ELEMENT_256);
        const Px = new Uint32Array(currentChunkN * LIMBS_PER_ELEMENT_256);
        const Py = new Uint32Array(currentChunkN * LIMBS_PER_ELEMENT_256);

        for (let i = 0; i < currentChunkN; i++) {
            const sourceIdx = chunkOffset + i;
            k.set(
                bigint256ToLimbs(scalars[sourceIdx]),
                i * LIMBS_PER_ELEMENT_256
            );
            Px.set(bigint256ToLimbs(P[sourceIdx].x), i * LIMBS_PER_ELEMENT_256);
            Py.set(bigint256ToLimbs(P[sourceIdx].y), i * LIMBS_PER_ELEMENT_256);
        }

        const kBuffer = device.createBuffer({
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(kBuffer.getMappedRange()).set(k);
        kBuffer.unmap();

        const PxBuffer = device.createBuffer({
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(PxBuffer.getMappedRange()).set(Px);
        PxBuffer.unmap();

        const PyBuffer = device.createBuffer({
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(PyBuffer.getMappedRange()).set(Py);
        PyBuffer.unmap();

        const PPxBuffer = device.createBuffer({
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const PPyBuffer = device.createBuffer({
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const PPzBuffer = device.createBuffer({
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Create bind group for ShaderA inputs
        const shaderAInputChunkBindGroup = device.createBindGroup({
            layout: device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'read-only-storage' },
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'read-only-storage' },
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'storage' },
                    },
                    {
                        binding: 3,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'storage' },
                    },
                    {
                        binding: 4,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'storage' },
                    },
                ],
            }),
            entries: [
                { binding: 0, resource: { buffer: PxBuffer } },
                { binding: 1, resource: { buffer: PyBuffer } },
                { binding: 2, resource: { buffer: PPyBuffer } },
                { binding: 3, resource: { buffer: PPxBuffer } },
                { binding: 4, resource: { buffer: PPzBuffer } },
            ],
        });

        computePassA.passBindGroups.push([
            { bindGroup: shaderAInputChunkBindGroup, idx: 0 },
        ]);

        // Create bind layout for PassBi1
        const shaderBi1InputChunkBindGroup = device.createBindGroup({
            layout: device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'read-only-storage' },
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'read-only-storage' },
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'read-only-storage' },
                    },
                    {
                        binding: 3,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: 'read-only-storage' },
                    },
                ],
            }),
            entries: [
                { binding: 0, resource: { buffer: kBuffer } },
                { binding: 1, resource: { buffer: PPxBuffer } },
                { binding: 2, resource: { buffer: PPyBuffer } },
                { binding: 3, resource: { buffer: PPzBuffer } },
            ],
        });

        computePassBi1.passBindGroups.push([{bindGroup: shaderBi1InputChunkBindGroup, idx: 2}])

        inputsRemaining -= currentChunkN;
    }

    return { x: 0n, y: 0n };
}

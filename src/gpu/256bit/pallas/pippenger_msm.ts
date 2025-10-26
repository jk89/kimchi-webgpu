import {
    pippengerShaderPassAProjectiveConversion as ShaderPassA,
    pippengerShaderPassBi1BucketScalarWeightedPointContribution as ShaderPassBi1,
    pippengerShaderPassBi2TreeReduceBucket as ShaderPassBi2,
    pippengerShaderPassCBucketAggregation as ShaderPassC,
    pippengerShaderPassDTreeReduceFinalPoint as ShaderPassD,
    pippengerShaderPassEFinalAccumulation as ShaderPassE,
} from './pippenger_msm.wgslc.js';
import {
    bigint256ToLimbs,
    limbs256ToBigint,
    BYTES_PER_ELEMENT_256,
    LIMBS_PER_ELEMENT_256,
} from '../helpers.js';
import { Point } from '../../../types/point.js';

const WORKGROUP_SIZE_A = 64;
const WORKGROUP_SIZE_Bi1 = 64;
const WORKGROUP_SIZE_Bi2 = 64;
const WORKGROUP_SIZE_C = 64;
const WORKGROUP_SIZE_D = 64;
const WORKGROUP_SIZE_E = 64;

export interface PippengerMSMConfig {
    bucketWidthBits?: number;
    verbose?: boolean;
}

type BindGroupEntry = { idx: number; bindGroup: GPUBindGroup };

class ComputePass {
    pipeline: GPUComputePipeline;
    passBindGroups: BindGroupEntry[][];

    constructor(pipeline: GPUComputePipeline) {
        this.pipeline = pipeline;
        this.passBindGroups = [];
    }

    addBindGroupSet(bindGroups: BindGroupEntry[]) {
        this.passBindGroups.push(bindGroups);
        return this;
    }

    dispatch(commandEncoder: GPUCommandEncoder, workgroups: number) {
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        this.passBindGroups.forEach((bgSet) => {
            bgSet.forEach(({ idx, bindGroup }) =>
                pass.setBindGroup(idx, bindGroup)
            );
        });
        pass.dispatchWorkgroups(workgroups);
        pass.end();
    }
}

export async function pippengerMSMPallas(
    device: GPUDevice,
    scalars: bigint[],
    P: Point[],
    config?: PippengerMSMConfig
): Promise<Point> {
    const n = scalars.length;
    if (n === 0) throw new Error('scalars and points arrays cannot be empty');
    if (P.length !== n)
        throw new Error('scalars and points must have same length');

    const BUCKET_WIDTH_BITS = config?.bucketWidthBits ?? 8;
    if (BUCKET_WIDTH_BITS < 1 || BUCKET_WIDTH_BITS > 22)
        throw new Error('bucketWidthBits must be 1–22');

    const NUMBER_OF_BUCKETS = 1 << BUCKET_WIDTH_BITS;
    const verbose = config?.verbose ?? true;

    const maxBufferSize = device.limits.maxStorageBufferBindingSize;
    const maxChunkN = Math.floor(maxBufferSize / BYTES_PER_ELEMENT_256);
    const numBatches = Math.ceil(n / maxChunkN);

    if (verbose) {
        console.log('=== Pippenger MSM Configuration ===');
        console.log(`Total points: ${n}`);
        console.log(`Bucket width (bits): ${BUCKET_WIDTH_BITS}`);
        console.log(`Number of buckets: ${NUMBER_OF_BUCKETS}`);
        console.log(`Max points per batch: ${maxChunkN}`);
        console.log(`Number of batches: ${numBatches}`);
        console.log('===================================');
    }

    let passCountA = 0;
    let passCountBi1 = 0;
    let passCountBi2 = 0;
    let passCountC = 0;
    let passCountD = 0;
    let passCountE = 0;

    // ---- Layouts ----
    const layoutPassA = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' },
            }, // x
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' },
            }, // y
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            }, // Px
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            }, // Py
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            }, // Pz
        ],
    });

    const layoutBucketWidthBits = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'uniform' },
            },
        ],
    });

    const layoutBucketIdx = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'uniform' },
            },
        ],
    });

    const layoutUniformN = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'uniform' },
            },
        ],
    });

    const layoutPassBi1_Input = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' },
            }, // k
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' },
            }, // PPx
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' },
            }, // PPy
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' },
            }, // PPz
        ],
    });

    const layoutWGG = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            },
        ],
    });

    const layoutBucketsStorage = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            },
        ],
    });

    const layoutUniformN_BatchIdx = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'uniform' },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'uniform' },
            },
        ],
    });

    const layoutFStorage = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            },
        ],
    });

    const layoutBatchFinalPoints = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            },
        ],
    });

    const layoutFinalPoint = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            },
        ],
    });

    // ---- Shader modules and pipelines ----
    const shaderModules = {
        A: device.createShaderModule({ code: ShaderPassA }),
        Bi1: device.createShaderModule({ code: ShaderPassBi1 }),
        Bi2: device.createShaderModule({ code: ShaderPassBi2 }),
        C: device.createShaderModule({ code: ShaderPassC }),
        D: device.createShaderModule({ code: ShaderPassD }),
        E: device.createShaderModule({ code: ShaderPassE }),
    };

    const pipelineA = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [layoutPassA],
        }),
        compute: { module: shaderModules.A, entryPoint: 'main' },
    });

    const pipelineBi1 = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [
                layoutBucketWidthBits,
                layoutBucketIdx,
                layoutPassBi1_Input,
                layoutWGG,
            ],
        }),
        compute: { module: shaderModules.Bi1, entryPoint: 'main' },
    });

    const pipelineBi2 = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [
                layoutUniformN,
                layoutBucketIdx,
                layoutWGG,
                layoutBucketsStorage,
            ],
        }),
        compute: { module: shaderModules.Bi2, entryPoint: 'main' },
    });

    const pipelineC = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [layoutBucketsStorage, layoutFStorage],
        }),
        compute: { module: shaderModules.C, entryPoint: 'main' },
    });

    const pipelineD = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [
                layoutUniformN_BatchIdx,
                layoutFStorage,
                layoutBatchFinalPoints,
            ],
        }),
        compute: { module: shaderModules.D, entryPoint: 'main' },
    });

    const pipelineE = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [
                layoutUniformN,
                layoutBatchFinalPoints,
                layoutFinalPoint,
            ],
        }),
        compute: { module: shaderModules.E, entryPoint: 'main' },
    });

    // ---- Persistent buffers (buckets, final points, final point) ----
    const BxBuffer = device.createBuffer({
        size: NUMBER_OF_BUCKETS * BYTES_PER_ELEMENT_256,
        usage: GPUBufferUsage.STORAGE,
    });
    const ByBuffer = device.createBuffer({
        size: NUMBER_OF_BUCKETS * BYTES_PER_ELEMENT_256,
        usage: GPUBufferUsage.STORAGE,
    });
    const BzBuffer = device.createBuffer({
        size: NUMBER_OF_BUCKETS * BYTES_PER_ELEMENT_256,
        usage: GPUBufferUsage.STORAGE,
    });

    const batchFinalPointsSize = Math.max(
        numBatches * BYTES_PER_ELEMENT_256,
        BYTES_PER_ELEMENT_256
    );
    const batchFinalPointsXBuffer = device.createBuffer({
        size: batchFinalPointsSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const batchFinalPointsYBuffer = device.createBuffer({
        size: batchFinalPointsSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const batchFinalPointsZBuffer = device.createBuffer({
        size: batchFinalPointsSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const finalPointXBuffer = device.createBuffer({
        size: BYTES_PER_ELEMENT_256,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const finalPointYBuffer = device.createBuffer({
        size: BYTES_PER_ELEMENT_256,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const finalPointXStagingBuffer = device.createBuffer({
        size: BYTES_PER_ELEMENT_256,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const finalPointYStagingBuffer = device.createBuffer({
        size: BYTES_PER_ELEMENT_256,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // ---- Persistent bind groups ----
    const bucketWidthBitsBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM,
        mappedAtCreation: true,
    });
    new Uint32Array(bucketWidthBitsBuffer.getMappedRange()).set([
        BUCKET_WIDTH_BITS,
    ]);
    bucketWidthBitsBuffer.unmap();
    const bindGroupBucketWidthBits = device.createBindGroup({
        layout: layoutBucketWidthBits,
        entries: [{ binding: 0, resource: { buffer: bucketWidthBitsBuffer } }],
    });

    const bindGroupBucketsStorage = device.createBindGroup({
        layout: layoutBucketsStorage,
        entries: [
            { binding: 0, resource: { buffer: BxBuffer } },
            { binding: 1, resource: { buffer: ByBuffer } },
            { binding: 2, resource: { buffer: BzBuffer } },
        ],
    });

    const bindGroupBatchFinalPoints = device.createBindGroup({
        layout: layoutBatchFinalPoints,
        entries: [
            { binding: 0, resource: { buffer: batchFinalPointsXBuffer } },
            { binding: 1, resource: { buffer: batchFinalPointsYBuffer } },
            { binding: 2, resource: { buffer: batchFinalPointsZBuffer } },
        ],
    });

    const bindGroupFinalPoint = device.createBindGroup({
        layout: layoutFinalPoint,
        entries: [
            { binding: 0, resource: { buffer: finalPointXBuffer } },
            { binding: 1, resource: { buffer: finalPointYBuffer } },
        ],
    });

    // ---- Reusable small uniform buffers & bind groups ----
    const u32Size = 4;

    // Bi2 'n' uniform (currentN_Bi2)
    const bi2NUniform = device.createBuffer({
        size: u32Size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroupBi2N = device.createBindGroup({
        layout: layoutUniformN,
        entries: [{ binding: 0, resource: { buffer: bi2NUniform } }],
    });

    // Pass D uniforms (n, batchIdx)
    const passD_N_Uniform = device.createBuffer({
        size: u32Size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const passD_batchIdxUniform = device.createBuffer({
        size: u32Size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroupPassD_Uniforms = device.createBindGroup({
        layout: layoutUniformN_BatchIdx,
        entries: [
            { binding: 0, resource: { buffer: passD_N_Uniform } },
            { binding: 1, resource: { buffer: passD_batchIdxUniform } },
        ],
    });

    // Pass E: n uniform
    const passE_N_Uniform = device.createBuffer({
        size: u32Size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroupPassE_N = device.createBindGroup({
        layout: layoutUniformN,
        entries: [{ binding: 0, resource: { buffer: passE_N_Uniform } }],
    });

    // Single bucket index uniform and bind group (updated per-bucket)
    const bucketIdxUniform = device.createBuffer({
        size: u32Size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroupBucketIdxShared = device.createBindGroup({
        layout: layoutBucketIdx,
        entries: [{ binding: 0, resource: { buffer: bucketIdxUniform } }],
    });

    // ---- Pre-allocate all per-batch buffers once and reuse ----
    const maxNumWorkgroupsBi1 = Math.ceil(maxChunkN / WORKGROUP_SIZE_Bi1);
    const wggSizeMax = maxNumWorkgroupsBi1 * BYTES_PER_ELEMENT_256;

    const WGGxBuffer = device.createBuffer({
        size: wggSizeMax,
        usage: GPUBufferUsage.STORAGE,
    });
    const WGGyBuffer = device.createBuffer({
        size: wggSizeMax,
        usage: GPUBufferUsage.STORAGE,
    });
    const WGGzBuffer = device.createBuffer({
        size: wggSizeMax,
        usage: GPUBufferUsage.STORAGE,
    });

    // Preallocated input & PP buffers (maxChunkN)
    const perBatchBufferSize = maxChunkN * BYTES_PER_ELEMENT_256;
    // Use COPY_DST so we can write with queue.writeBuffer
    const kBuffer = device.createBuffer({
        size: perBatchBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const PxBuffer = device.createBuffer({
        size: perBatchBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const PyBuffer = device.createBuffer({
        size: perBatchBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Projective outputs preallocated
    const PPxBuffer = device.createBuffer({
        size: perBatchBufferSize,
        usage: GPUBufferUsage.STORAGE,
    });
    const PPyBuffer = device.createBuffer({
        size: perBatchBufferSize,
        usage: GPUBufferUsage.STORAGE,
    });
    const PPzBuffer = device.createBuffer({
        size: perBatchBufferSize,
        usage: GPUBufferUsage.STORAGE,
    });

    // Preallocated WGG bindgroups (reuse)
    const bindGroupPassBi1_Output = device.createBindGroup({
        layout: pipelineBi1.getBindGroupLayout(3),
        entries: [
            { binding: 0, resource: { buffer: WGGxBuffer } },
            { binding: 1, resource: { buffer: WGGyBuffer } },
            { binding: 2, resource: { buffer: WGGzBuffer } },
        ],
    });

    const bindGroupPassBi2_WGG = device.createBindGroup({
        layout: pipelineBi2.getBindGroupLayout(2),
        entries: [
            { binding: 0, resource: { buffer: WGGxBuffer } },
            { binding: 1, resource: { buffer: WGGyBuffer } },
            { binding: 2, resource: { buffer: WGGzBuffer } },
        ],
    });

    // Pre-allocate F buffers to max needed (maxNumWorkgroupsC)
    const maxNumWorkgroupsC = Math.ceil(NUMBER_OF_BUCKETS / WORKGROUP_SIZE_C);
    const fBufferSizeMax = maxNumWorkgroupsC * BYTES_PER_ELEMENT_256;
    const FxBuffer = device.createBuffer({
        size: fBufferSizeMax,
        usage: GPUBufferUsage.STORAGE,
    });
    const FyBuffer = device.createBuffer({
        size: fBufferSizeMax,
        usage: GPUBufferUsage.STORAGE,
    });
    const FzBuffer = device.createBuffer({
        size: fBufferSizeMax,
        usage: GPUBufferUsage.STORAGE,
    });

    // Bind groups which can be reused (point their buffers to preallocated ones)
    const bindGroupPassA = device.createBindGroup({
        layout: layoutPassA,
        entries: [
            { binding: 0, resource: { buffer: PxBuffer } },
            { binding: 1, resource: { buffer: PyBuffer } },
            { binding: 2, resource: { buffer: PPxBuffer } },
            { binding: 3, resource: { buffer: PPyBuffer } },
            { binding: 4, resource: { buffer: PPzBuffer } },
        ],
    });

    const bindGroupPassBi1_Input = device.createBindGroup({
        layout: pipelineBi1.getBindGroupLayout(2),
        entries: [
            { binding: 0, resource: { buffer: kBuffer } },
            { binding: 1, resource: { buffer: PPxBuffer } },
            { binding: 2, resource: { buffer: PPyBuffer } },
            { binding: 3, resource: { buffer: PPzBuffer } },
        ],
    });

    const bindGroupPassC_FOutput = device.createBindGroup({
        layout: layoutFStorage,
        entries: [
            { binding: 0, resource: { buffer: FxBuffer } },
            { binding: 1, resource: { buffer: FyBuffer } },
            { binding: 2, resource: { buffer: FzBuffer } },
        ],
    });

    const bindGroupPassD_Input = device.createBindGroup({
        layout: layoutFStorage,
        entries: [
            { binding: 0, resource: { buffer: FxBuffer } },
            { binding: 1, resource: { buffer: FyBuffer } },
            { binding: 2, resource: { buffer: FzBuffer } },
        ],
    });

    // Command encoder
    const commandEncoder = device.createCommandEncoder();

    // ---- Main per-batch loop: writeBuffer into preallocated buffers and reuse bind groups ----
    for (let batchIdx = 0; batchIdx < numBatches; batchIdx++) {
        const batchOffset = batchIdx * maxChunkN;
        const currentBatchN = Math.min(n - batchOffset, maxChunkN);

        if (verbose)
            console.log(
                `\nProcessing batch ${
                    batchIdx + 1
                }/${numBatches} (${currentBatchN} points)`
            );

        // Fill typed arrays locally (CPU-side)
        const kArr = new Uint32Array(currentBatchN * LIMBS_PER_ELEMENT_256);
        const PxArr = new Uint32Array(currentBatchN * LIMBS_PER_ELEMENT_256);
        const PyArr = new Uint32Array(currentBatchN * LIMBS_PER_ELEMENT_256);

        for (let i = 0; i < currentBatchN; i++) {
            const idx = batchOffset + i;
            kArr.set(bigint256ToLimbs(scalars[idx]), i * LIMBS_PER_ELEMENT_256);
            PxArr.set(bigint256ToLimbs(P[idx].x), i * LIMBS_PER_ELEMENT_256);
            PyArr.set(bigint256ToLimbs(P[idx].y), i * LIMBS_PER_ELEMENT_256);
        }

        // Upload to GPU via writeBuffer (no new allocations)
        device.queue.writeBuffer(kBuffer, 0, kArr);
        device.queue.writeBuffer(PxBuffer, 0, PxArr);
        device.queue.writeBuffer(PyBuffer, 0, PyArr);

        // ---- Pass A: affine -> projective (reused bindGroupPassA) ----
        const numWorkgroupsA = Math.ceil(currentBatchN / WORKGROUP_SIZE_A);
        new ComputePass(pipelineA)
            .addBindGroupSet([{ idx: 0, bindGroup: bindGroupPassA }])
            .dispatch(commandEncoder, numWorkgroupsA);
        passCountA++;

        // ---- Bi1 / Bi2 (per-bucket passes preserved) ----
        const numWorkgroupsBi1 = Math.ceil(currentBatchN / WORKGROUP_SIZE_Bi1);

        for (let bucketIdx = 0; bucketIdx < NUMBER_OF_BUCKETS; bucketIdx++) {
            // update bucket index uniform (single precreated uniform)
            device.queue.writeBuffer(
                bucketIdxUniform,
                0,
                new Uint32Array([bucketIdx])
            );

            // Bi1 dispatch
            const passBi1 = commandEncoder.beginComputePass();
            passBi1.setPipeline(pipelineBi1);
            passBi1.setBindGroup(0, bindGroupBucketWidthBits);
            passBi1.setBindGroup(1, bindGroupBucketIdxShared);
            passBi1.setBindGroup(2, bindGroupPassBi1_Input);
            passBi1.setBindGroup(3, bindGroupPassBi1_Output);
            passBi1.dispatchWorkgroups(numWorkgroupsBi1);
            passBi1.end();
            passCountBi1++;

            // Bi2 reduce (reuse bi2NUniform)
            let currentN_Bi2 = numWorkgroupsBi1;
            while (currentN_Bi2 > 1) {
                device.queue.writeBuffer(
                    bi2NUniform,
                    0,
                    new Uint32Array([currentN_Bi2])
                );

                const passBi2 = commandEncoder.beginComputePass();
                passBi2.setPipeline(pipelineBi2);
                passBi2.setBindGroup(0, bindGroupBi2N);
                passBi2.setBindGroup(1, bindGroupBucketIdxShared);
                passBi2.setBindGroup(2, bindGroupPassBi2_WGG);
                passBi2.setBindGroup(3, bindGroupBucketsStorage);
                passBi2.dispatchWorkgroups(
                    Math.ceil(currentN_Bi2 / WORKGROUP_SIZE_Bi2)
                );
                passBi2.end();
                passCountBi2++;

                currentN_Bi2 = Math.ceil(currentN_Bi2 / WORKGROUP_SIZE_Bi2);
            }
        }

        // ---- Pass C: use preallocated Fx/Fy/Fz and reused bindGroupPassC_FOutput ----
        const numWorkgroupsC = Math.ceil(NUMBER_OF_BUCKETS / WORKGROUP_SIZE_C);
        const passC = commandEncoder.beginComputePass();
        passC.setPipeline(pipelineC);
        passC.setBindGroup(0, bindGroupBucketsStorage);
        passC.setBindGroup(1, bindGroupPassC_FOutput);
        passC.dispatchWorkgroups(numWorkgroupsC);
        passC.end();
        passCountC++;

        // ---- Pass D: reduce F -> batchFinalPoints (reused bindGroupPassD_Input & bindGroupPassD_Uniforms) ----
        let currentN_D = numWorkgroupsC;
        while (currentN_D > 1) {
            device.queue.writeBuffer(
                passD_N_Uniform,
                0,
                new Uint32Array([currentN_D])
            );
            device.queue.writeBuffer(
                passD_batchIdxUniform,
                0,
                new Uint32Array([batchIdx])
            );

            const passD = commandEncoder.beginComputePass();
            passD.setPipeline(pipelineD);
            passD.setBindGroup(0, bindGroupPassD_Uniforms);
            passD.setBindGroup(1, bindGroupPassD_Input);
            passD.setBindGroup(2, bindGroupBatchFinalPoints);
            passD.dispatchWorkgroups(Math.ceil(currentN_D / WORKGROUP_SIZE_D));
            passD.end();
            passCountD++;

            currentN_D = Math.ceil(currentN_D / WORKGROUP_SIZE_D);
        }
    } // end batches

    // ---- Pass E: final accumulation (reused passE_N_Uniform & bind group) ----
    let currentN_E = numBatches;
    while (currentN_E >= 1) {
        device.queue.writeBuffer(
            passE_N_Uniform,
            0,
            new Uint32Array([currentN_E])
        );

        const passE = commandEncoder.beginComputePass();
        passE.setPipeline(pipelineE);
        passE.setBindGroup(0, bindGroupPassE_N);
        passE.setBindGroup(1, bindGroupBatchFinalPoints);
        passE.setBindGroup(2, bindGroupFinalPoint);
        passE.dispatchWorkgroups(Math.ceil(currentN_E / WORKGROUP_SIZE_E));
        passE.end();
        passCountE++;

        currentN_E = Math.ceil(currentN_E / WORKGROUP_SIZE_E);

        // Break after final pass (when we've reduced to 1 and dispatched it)
        if (currentN_E === 1 && numBatches === 1) {
            break;
        }
        if (currentN_E <= 1) {
            break;
        }
    }

    commandEncoder.copyBufferToBuffer(
        finalPointXBuffer,
        0,
        finalPointXStagingBuffer,
        0,
        BYTES_PER_ELEMENT_256
    );
    commandEncoder.copyBufferToBuffer(
        finalPointYBuffer,
        0,
        finalPointYStagingBuffer,
        0,
        BYTES_PER_ELEMENT_256
    );

    // Print pass counts before submission
    if (verbose) {
        console.log(`\n--- Actual Dispatches per Stage ---`);
        console.log(`Pass A:   ${passCountA} dispatches`);
        console.log(`Pass Bi1: ${passCountBi1} dispatches`);
        console.log(`Pass Bi2: ${passCountBi2} dispatches`);
        console.log(`Pass C:   ${passCountC} dispatches`);
        console.log(`Pass D:   ${passCountD} dispatches`);
        console.log(`Pass E:   ${passCountE} dispatches`);
        console.log(
            `TOTAL:    ${
                passCountA +
                passCountBi1 +
                passCountBi2 +
                passCountC +
                passCountD +
                passCountE
            } dispatches`
        );
        console.log('===================================\n');
    }

    // Submit and read back
    device.queue.submit([commandEncoder.finish()]);

    await finalPointXStagingBuffer.mapAsync(GPUMapMode.READ);
    await finalPointYStagingBuffer.mapAsync(GPUMapMode.READ);

    const finalX = limbs256ToBigint(
        new Uint32Array(finalPointXStagingBuffer.getMappedRange())
    );
    const finalY = limbs256ToBigint(
        new Uint32Array(finalPointYStagingBuffer.getMappedRange())
    );

    finalPointXStagingBuffer.unmap();
    finalPointYStagingBuffer.unmap();

    return { x: finalX, y: finalY };
}

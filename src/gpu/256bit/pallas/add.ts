import {
    bigint256ToLimbs,
    limbs256ToBigint,
    BYTES_PER_ELEMENT_256,
    LIMBS_PER_ELEMENT_256,
} from '../helpers.js';
import addShader from './add.wgslc.js';

export async function addPallas(
    device: GPUDevice,
    a: bigint[],
    b: bigint[]
): Promise<bigint[]> {
    const n = a.length;

    if (n === 0 || b.length !== n) {
        throw new Error('Input arrays must be the same non-zero length');
    }

    const bufferSize = n * BYTES_PER_ELEMENT_256;

    // Prepare input data
    const aData = new Uint32Array(n * LIMBS_PER_ELEMENT_256);
    const bData = new Uint32Array(n * LIMBS_PER_ELEMENT_256);

    for (let i = 0; i < n; i++) {
        aData.set(bigint256ToLimbs(a[i]), i * LIMBS_PER_ELEMENT_256);
        bData.set(bigint256ToLimbs(b[i]), i * LIMBS_PER_ELEMENT_256);
    }

    // Create input buffers
    const aBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(aBuffer.getMappedRange()).set(aData);
    aBuffer.unmap();

    const bBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(bBuffer.getMappedRange()).set(bData);
    bBuffer.unmap();

    // Create output buffer
    const outBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const readBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Create shader module and pipeline
    const shaderModule = device.createShaderModule({ code: addShader });

    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: 'main',
        },
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: aBuffer } },
            { binding: 1, resource: { buffer: bBuffer } },
            { binding: 2, resource: { buffer: outBuffer } },
        ],
    });

    // Execute shader
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 64));
    passEncoder.end();

    commandEncoder.copyBufferToBuffer(outBuffer, 0, readBuffer, 0, bufferSize);
    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Map and read results
    await readBuffer.mapAsync(GPUMapMode.READ);
    const outData = new Uint32Array(readBuffer.getMappedRange()).slice();
    readBuffer.unmap();

    const result: bigint[] = [];
    for (let i = 0; i < n; i++) {
        result.push(
            limbs256ToBigint(
                outData.slice(
                    i * LIMBS_PER_ELEMENT_256,
                    (i + 1) * LIMBS_PER_ELEMENT_256
                )
            )
        );
    }

    return result;
}

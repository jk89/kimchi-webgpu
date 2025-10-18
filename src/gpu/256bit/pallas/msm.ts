// pallas_msm.ts
import { Point } from '../../../types/point.js';
import {
    bigint256ToLimbs,
    limbs256ToBigint,
    BYTES_PER_ELEMENT_256,
    LIMBS_PER_ELEMENT_256,
} from '../helpers.js';
import pallasMsmShader from './msm.wgslc.js';

export async function msmPallas(
    device: GPUDevice,
    scalars: bigint[],
    P: Point[]
): Promise<Point[]> {
    const n = scalars.length;

    if (n === 0) {
        throw new Error('scalars and points arrays cannot be empty');
    }

    if (P.length !== n) {
        throw new Error(
            `scalars and points must have the same length: got ${n} scalars and ${P.length} points`
        );
    }

    const bufferSize = n * BYTES_PER_ELEMENT_256;

    // Prepare input data
    const scalarsData = new Uint32Array(n * LIMBS_PER_ELEMENT_256);
    const Px_Data = new Uint32Array(n * LIMBS_PER_ELEMENT_256);
    const Py_Data = new Uint32Array(n * LIMBS_PER_ELEMENT_256);

    for (let i = 0; i < n; i++) {
        scalarsData.set(
            bigint256ToLimbs(scalars[i]),
            i * LIMBS_PER_ELEMENT_256
        );
        Px_Data.set(bigint256ToLimbs(P[i].x), i * LIMBS_PER_ELEMENT_256);
        Py_Data.set(bigint256ToLimbs(P[i].y), i * LIMBS_PER_ELEMENT_256);
    }

    // Create input buffers with mappedAtCreation
    const scalarsBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(scalarsBuffer.getMappedRange()).set(scalarsData);
    scalarsBuffer.unmap();

    const Px_Buffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(Px_Buffer.getMappedRange()).set(Px_Data);
    Px_Buffer.unmap();

    const Py_Buffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(Py_Buffer.getMappedRange()).set(Py_Data);
    Py_Buffer.unmap();

    // Create output buffers
    const Qx_Buffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const Qy_Buffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Create read buffers
    const Qx_ReadBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const Qy_ReadBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Create shader module and pipeline
    const shaderModule = device.createShaderModule({
        code: pallasMsmShader,
    });

    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: 'main',
        },
    });

    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: scalarsBuffer } },
            { binding: 1, resource: { buffer: Px_Buffer } },
            { binding: 2, resource: { buffer: Py_Buffer } },
            { binding: 3, resource: { buffer: Qx_Buffer } },
            { binding: 4, resource: { buffer: Qy_Buffer } },
        ],
    });

    // Execute compute shader
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 64));
    passEncoder.end();

    commandEncoder.copyBufferToBuffer(
        Qx_Buffer,
        0,
        Qx_ReadBuffer,
        0,
        bufferSize
    );
    commandEncoder.copyBufferToBuffer(
        Qy_Buffer,
        0,
        Qy_ReadBuffer,
        0,
        bufferSize
    );

    device.queue.submit([commandEncoder.finish()]);

    // Wait for GPU work to complete
    await device.queue.onSubmittedWorkDone();

    // Map and read results
    await Qx_ReadBuffer.mapAsync(GPUMapMode.READ);
    const Q_x_Result = new Uint32Array(Qx_ReadBuffer.getMappedRange()).slice();
    Qx_ReadBuffer.unmap();

    await Qy_ReadBuffer.mapAsync(GPUMapMode.READ);
    const Q_y_Result = new Uint32Array(Qy_ReadBuffer.getMappedRange()).slice();
    Qy_ReadBuffer.unmap();

    // Convert results back to bigints
    const Q: Point[] = [];
    for (let i = 0; i < n; i++) {
        Q.push({
            x: limbs256ToBigint(
                Q_x_Result.slice(
                    i * LIMBS_PER_ELEMENT_256,
                    (i + 1) * LIMBS_PER_ELEMENT_256
                )
            ),
            y: limbs256ToBigint(
                Q_y_Result.slice(
                    i * LIMBS_PER_ELEMENT_256,
                    (i + 1) * LIMBS_PER_ELEMENT_256
                )
            ),
        });
    }

    return Q;
}

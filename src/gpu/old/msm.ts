import msmShader from './msm.wgsl';

export interface Point {
    x: bigint;
    y: bigint;
}

function toLimbs(x: bigint): Uint32Array {
    const limbs = new Uint32Array(8);
    let tmp = x;
    for (let i = 0; i < 8; i++) {
        limbs[i] = Number(tmp & 0xffffffffn);
        tmp >>= 32n;
    }
    return limbs;
}

function limbsToBigInt(limbs: Uint32Array): bigint {
    let res = 0n;
    for (let i = 7; i >= 0; i--) res = (res << 32n) + BigInt(limbs[i]);
    return res;
}

export async function gpuMSM(
    device: GPUDevice,
    scalars: bigint[],
    points: Point[]
): Promise<Point[]> {
    // Need to validate points and scalars have the same length
    if (points.length !== scalars.length)
        throw new Error('Points and Scalars must have the same length.');

    const bufferSize = points.length * 8;

    // Flatten scalars and points into limbs
    const scalarLimbs = new Uint32Array(bufferSize);
    const pointXLimbs = new Uint32Array(bufferSize);
    const pointYLimbs = new Uint32Array(bufferSize);

    scalars.forEach((s, i) => scalarLimbs.set(toLimbs(s), i * 8));
    points.forEach((p, i) => {
        pointXLimbs.set(toLimbs(p.x), i * 8);
        pointYLimbs.set(toLimbs(p.y), i * 8);
    });

    // Input storage buffers
    const scalarBuffer = device.createBuffer({
        size: scalarLimbs.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(scalarBuffer.getMappedRange()).set(scalarLimbs);
    scalarBuffer.unmap();

    const pointXBuffer = device.createBuffer({
        size: pointXLimbs.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(pointXBuffer.getMappedRange()).set(pointXLimbs);
    pointXBuffer.unmap();

    const pointYBuffer = device.createBuffer({
        size: pointYLimbs.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(pointYBuffer.getMappedRange()).set(pointYLimbs);
    pointYBuffer.unmap();

    // Output storage buffers
    const outXBuffer = device.createBuffer({
        size: pointXLimbs.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const outYBuffer = device.createBuffer({
        size: pointYLimbs.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Read buffer

    const readXBuffer = device.createBuffer({
        size: pointXLimbs.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const readYBuffer = device.createBuffer({
        size: pointYLimbs.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const module = device.createShaderModule({ code: msmShader });
    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: scalarBuffer } },
            { binding: 1, resource: { buffer: pointXBuffer } },
            { binding: 2, resource: { buffer: pointYBuffer } },
            { binding: 3, resource: { buffer: outXBuffer } },
            { binding: 4, resource: { buffer: outYBuffer } },
        ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const workgroups = Math.ceil(scalars.length / 64);
    pass.dispatchWorkgroups(workgroups);
    pass.end();

    encoder.copyBufferToBuffer(
        outXBuffer,
        0,
        readXBuffer,
        0,
        pointXLimbs.byteLength
    );
    encoder.copyBufferToBuffer(
        outYBuffer,
        0,
        readYBuffer,
        0,
        pointYLimbs.byteLength
    );
    device.queue.submit([encoder.finish()]);

    await device.queue.onSubmittedWorkDone();

    // Map readback buffers
    await readXBuffer.mapAsync(GPUMapMode.READ);
    const resultXArray = new Uint32Array(readXBuffer.getMappedRange()).slice();
    readXBuffer.unmap();

    await readYBuffer.mapAsync(GPUMapMode.READ);
    const resultYArray = new Uint32Array(readYBuffer.getMappedRange()).slice();
    readYBuffer.unmap();

    const results: Point[] = [];
    for (let i = 0; i < scalars.length; i++) {
        const x = resultXArray.slice(i * 8, i * 8 + 8);
        const y = resultYArray.slice(i * 8, i * 8 + 8);
        results.push({ x: limbsToBigInt(x), y: limbsToBigInt(y) });
    }

    return results;
}

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
    console.log(
        'Scalars:',
        scalars.map((s) => s.toString())
    );
    console.log(
        'Points:',
        points.map((p) => ({ x: p.x.toString(), y: p.y.toString() }))
    );

    // Flatten scalars and points into limbs
    const scalarLimbs = new Uint32Array(scalars.length * 8);
    scalars.forEach((s, i) => scalarLimbs.set(toLimbs(s), i * 8));
    console.log('Scalar limbs:', scalarLimbs);

    const pointLimbs = new Uint32Array(points.length * 16);
    points.forEach((p, i) => {
        pointLimbs.set(toLimbs(p.x), i * 16);
        pointLimbs.set(toLimbs(p.y), i * 16 + 8);
    });
    console.log('Point limbs:', pointLimbs);

    // Storage buffers
    const scalarBuffer = device.createBuffer({
        size: scalarLimbs.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(scalarBuffer.getMappedRange()).set(scalarLimbs);
    scalarBuffer.unmap();

    const pointBuffer = device.createBuffer({
        size: pointLimbs.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(pointBuffer.getMappedRange()).set(pointLimbs);
    pointBuffer.unmap();

    const outBuffer = device.createBuffer({
        size: pointLimbs.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const readBuffer = device.createBuffer({
        size: pointLimbs.byteLength,
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
            { binding: 1, resource: { buffer: pointBuffer } },
            { binding: 2, resource: { buffer: outBuffer } },
        ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const workgroups = Math.ceil(scalars.length / 64);
    console.log('Dispatching workgroups:', workgroups);
    pass.dispatchWorkgroups(workgroups);
    pass.end();

    encoder.copyBufferToBuffer(
        outBuffer,
        0,
        readBuffer,
        0,
        pointLimbs.byteLength
    );
    device.queue.submit([encoder.finish()]);

    await device.queue.onSubmittedWorkDone();

    // Map readback buffer
    await readBuffer.mapAsync(GPUMapMode.READ);
    const resultArray = new Uint32Array(readBuffer.getMappedRange()).slice();
    readBuffer.unmap();

    console.log('Result buffer limbs:', resultArray);

    const results: Point[] = [];
    for (let i = 0; i < scalars.length; i++) {
        const x = resultArray.slice(i * 16, i * 16 + 8);
        const y = resultArray.slice(i * 16 + 8, i * 16 + 16);
        console.log(`Result ${i}: x limbs=${x}, y limbs=${y}`);
        results.push({ x: limbsToBigInt(x), y: limbsToBigInt(y) });
    }

    console.log(
        'Final results:',
        results.map((p) => ({ x: p.x.toString(), y: p.y.toString() }))
    );
    return results;
}

import mulAddCarryShader from './mul_add_carry.wgslc.js';

export async function mulAddCarryGPU(
    device: GPUDevice,
    a: number[],
    b: number[],
    acc: number[] = [],
    carry: number[] = []
): Promise<{ res: number; carry: number }[]> {
    const n = a.length;
    if (b.length !== n) throw new Error('a and b must have same length');

    const aBufferData = new Uint32Array(a);
    const bBufferData = new Uint32Array(b);
    const accCarryData = new Uint32Array(n * 2);

    for (let i = 0; i < n; i++) {
        accCarryData[i * 2] = acc[i] ?? 0;
        accCarryData[i * 2 + 1] = carry[i] ?? 0;
    }

    const aBuffer = device.createBuffer({
        size: aBufferData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(aBuffer.getMappedRange()).set(aBufferData);
    aBuffer.unmap();

    const bBuffer = device.createBuffer({
        size: bBufferData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(bBuffer.getMappedRange()).set(bBufferData);
    bBuffer.unmap();

    const accCarryBuffer = device.createBuffer({
        size: accCarryData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint32Array(accCarryBuffer.getMappedRange()).set(accCarryData);
    accCarryBuffer.unmap();

    const readBuffer = device.createBuffer({
        size: accCarryData.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const shaderModule = device.createShaderModule({ code: mulAddCarryShader });
    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: 'main' },
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: aBuffer } },
            { binding: 1, resource: { buffer: bBuffer } },
            { binding: 2, resource: { buffer: accCarryBuffer } },
        ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.end();

    encoder.copyBufferToBuffer(accCarryBuffer, 0, readBuffer, 0, accCarryData.byteLength);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await readBuffer.mapAsync(GPUMapMode.READ);
    const resultData = new Uint32Array(readBuffer.getMappedRange()).slice();
    readBuffer.unmap();

    const results: { res: number; carry: number }[] = [];
    for (let i = 0; i < n; i++) {
        results.push({
            res: resultData[i * 2],
            carry: resultData[i * 2 + 1],
        });
    }
    return results;
}

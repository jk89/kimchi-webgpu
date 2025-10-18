import {
  bigint256ToLimbs,
  limbs256ToBigint,
  BYTES_PER_ELEMENT_256,
  LIMBS_PER_ELEMENT_256
} from '../helpers.js';
import shader from './toMontgomery.wgslc.js';

export async function toMontgomeryGPU(
  device: GPUDevice,
  a: bigint[]
): Promise<bigint[]> {
  const n = a.length;
  const bufferSize = n * BYTES_PER_ELEMENT_256;

  const aData = new Uint32Array(n * LIMBS_PER_ELEMENT_256);
  for (let i = 0; i < n; i++) {
    aData.set(bigint256ToLimbs(a[i]), i * LIMBS_PER_ELEMENT_256);
  }

  const aBuffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(aBuffer.getMappedRange()).set(aData);
  aBuffer.unmap();

  const outBuffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const outReadBuffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const shaderModule = device.createShaderModule({ code: shader });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: 'main' },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aBuffer } },
      { binding: 1, resource: { buffer: outBuffer } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const pass = commandEncoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(n / 64));
  pass.end();

  commandEncoder.copyBufferToBuffer(outBuffer, 0, outReadBuffer, 0, bufferSize);
  device.queue.submit([commandEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  await outReadBuffer.mapAsync(GPUMapMode.READ);
  const outData = new Uint32Array(outReadBuffer.getMappedRange()).slice();
  outReadBuffer.unmap();

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

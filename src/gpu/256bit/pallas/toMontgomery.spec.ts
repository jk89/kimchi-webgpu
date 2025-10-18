import { expect } from 'chai';
import { toMontgomeryGPU } from './toMontgomery.js';
import { Field } from 'o1js';

describe('GPU toMontgomery debug', () => {
  it('computes Montgomery form of 1', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter!.requestDevice();

    const a = [1n]; // input 1
    const gpuResult = await toMontgomeryGPU(device, a);

    // Compute R mod P dynamically
    const R = 1n << 256n;
    const expectedR = R % Field.ORDER;

    console.log('GPU returned:', gpuResult[0].toString());
    console.log('Expected R mod p:', expectedR.toString());

    expect(gpuResult[0].toString()).to.equal(expectedR.toString());
  });
});

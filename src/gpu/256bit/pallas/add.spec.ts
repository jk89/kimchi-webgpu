import { pallasAdd } from './add.js';
import { Field } from 'o1js';
import { expect } from 'chai';

describe('GPU Pallas Add', () => {
    it('computes addition correctly', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();

        const a = [BigInt(1), BigInt(2), BigInt(123456789)];
        const b = [BigInt(3), BigInt(4), BigInt(987654321)];

        const gpuResults = await pallasAdd(device, a, b);

        for (let i = 0; i < a.length; i++) {
            const expected = Field(a[i]).add(Field(b[i])).toBigInt();
            expect(gpuResults[i].toString()).to.equal(expected.toString());
        }
    });
});

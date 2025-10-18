import { mulPallas } from './mul.js';
import { Field } from 'o1js';
import { expect } from 'chai';

describe('GPU mulPallas', () => {
    it('multiplies 1 and 1 correctly', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();

        const a = [1n];
        const b = [1n];

        const gpuResults = await mulPallas(device, a, b);
        const expected = Field(1).mul(1).toBigInt();

        expect(gpuResults[0].toString()).to.equal(expected.toString()); // Should be 1
    });

    it('computes simple multiplications correctly', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();

        const a = [1n, 2n, 123456789n];
        const b = [3n, 4n, 987654321n];

        const gpuResults = await mulPallas(device, a, b);

        for (let i = 0; i < a.length; i++) {
            const expected = Field(a[i]).mul(Field(b[i])).toBigInt();
            expect(gpuResults[i].toString()).to.equal(expected.toString());
        }
    });

    it('wraps correctly under Field.ORDER', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();

        const ORDER = Field.ORDER;

        const a = [ORDER - 1n, ORDER - 5n];
        const b = [2n, 10n];

        const gpuResults = await mulPallas(device, a, b);

        for (let i = 0; i < a.length; i++) {
            const expected = Field(a[i]).mul(Field(b[i])).toBigInt();
            expect(gpuResults[i].toString()).to.equal(expected.toString());
        }
    });
});

import { pallasSub } from './sub.js';
import { Field } from 'o1js';
import { expect } from 'chai';

describe('GPU Pallas Sub', () => {
    it('computes simple subtractions correctly', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();

        const a = [3n, 10n, 987654321n];
        const b = [1n, 4n, 123456789n];

        const gpuResults = await pallasSub(device, a, b);

        for (let i = 0; i < a.length; i++) {
            const expected = Field(a[i]).sub(Field(b[i])).toBigInt();
            expect(gpuResults[i].toString()).to.equal(expected.toString());
        }
    });

    it('wraps correctly under Field.ORDER (negative results)', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();

        const ORDER = Field.ORDER;

        // Cases that would go negative before mod reduction
        const a = [1n, 5n, 10n];
        const b = [2n, 6n, ORDER - 1n];

        const gpuResults = await pallasSub(device, a, b);

        for (let i = 0; i < a.length; i++) {
            const expected = Field(a[i]).sub(Field(b[i])).toBigInt();
            expect(gpuResults[i].toString()).to.equal(expected.toString());
        }
    });

    it('handles zero, identity, and full-field edge cases', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();

        const ORDER = Field.ORDER;

        const a = [0n, ORDER - 1n, 0n];
        const b = [0n, 1n, ORDER - 1n];

        const gpuResults = await pallasSub(device, a, b);

        for (let i = 0; i < a.length; i++) {
            const expected = Field(a[i]).sub(Field(b[i])).toBigInt();
            expect(gpuResults[i].toString()).to.equal(expected.toString());
        }
    });
});

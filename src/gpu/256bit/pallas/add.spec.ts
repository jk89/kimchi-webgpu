import { addPallas } from './add.js';
import { Field } from 'o1js';
import { expect } from 'chai';

describe('GPU addPallas', () => {
    it('computes simple additions correctly', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();

        const a = [1n, 2n, 123456789n];
        const b = [3n, 4n, 987654321n];

        const gpuResults = await addPallas(device, a, b);

        for (let i = 0; i < a.length; i++) {
            const expected = Field(a[i]).add(Field(b[i])).toBigInt();
            expect(gpuResults[i].toString()).to.equal(expected.toString());
        }
    });

    it('wraps around Field.ORDER correctly', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();

        const ORDER = Field.ORDER;

        // cases that exceed the modulus
        const a = [ORDER - 1n, ORDER - 5n, ORDER - 10n];
        const b = [2n, 6n, ORDER - 1n];

        const gpuResults = await addPallas(device, a, b);

        for (let i = 0; i < a.length; i++) {
            // compute expected mod result using o1js
            const expected = Field(a[i]).add(Field(b[i])).toBigInt();
            expect(gpuResults[i].toString()).to.equal(expected.toString());
        }
    });

    it('handles zero, identity, and full-field edge cases', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();

        const ORDER = Field.ORDER;

        const a = [0n, 0n, ORDER - 1n];
        const b = [0n, 5n, 1n];

        const gpuResults = await addPallas(device, a, b);

        for (let i = 0; i < a.length; i++) {
            const expected = Field(a[i]).add(Field(b[i])).toBigInt();
            expect(gpuResults[i].toString()).to.equal(expected.toString());
        }
    });
});

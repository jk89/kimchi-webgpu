import { expect } from 'chai';
import { mulAddCarryGPU } from './mul_add_carry.js';

describe('GPU mul_add_carry', () => {
    it('multiplies simple numbers correctly', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();

        const a = [1, 0xFFFF, 123456];
        const b = [2, 2, 654321];

        const results = await mulAddCarryGPU(device, a, b);

        for (let i = 0; i < a.length; i++) {
            const expected = BigInt(a[i]) * BigInt(b[i]);
            const actual = BigInt(results[i].res) + (BigInt(results[i].carry) << 32n);
            expect(actual).to.equal(expected);
        }
    });

    it('handles zero and one', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();

        const a = [0, 1];
        const b = [0, 1];

        const results = await mulAddCarryGPU(device, a, b);

        expect(BigInt(results[0].res) + (BigInt(results[0].carry) << 32n)).to.equal(0n);
        expect(BigInt(results[1].res) + (BigInt(results[1].carry) << 32n)).to.equal(1n);
    });

    it('handles max 32-bit values', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();

        const a = [0xFFFF_FFFF];
        const b = [0xFFFF_FFFF];

        const results = await mulAddCarryGPU(device, a, b);

        const expected = BigInt(a[0]) * BigInt(b[0]);
        const actual = BigInt(results[0].res) + (BigInt(results[0].carry) << 32n);
        expect(actual).to.equal(expected);
    });
});

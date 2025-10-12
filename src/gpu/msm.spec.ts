import { expect } from 'chai';
import { gpuMSM } from './msm.js';
import { Field } from 'o1js';

describe('GPU MSM', () => {
    it('performs basic field operations', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();
        
        const scalars = [1n, 2n, 3n].map(s => Field(s).toBigInt());
        const points = [
            { x: Field(5).toBigInt(), y: Field(7).toBigInt() },
            { x: Field(11).toBigInt(), y: Field(13).toBigInt() },
            { x: Field(17).toBigInt(), y: Field(19).toBigInt() }
        ];

        const result = await gpuMSM(device, scalars, points);
        
        console.log('Results:', result);
        
        // Just verify we got non-zero results (shows the shader ran)
        for (let i = 0; i < scalars.length; i++) {
            expect(result[i].x).to.not.equal(0n, `Result ${i} x should be non-zero`);
            expect(result[i].y).to.not.equal(0n, `Result ${i} y should be non-zero`);
        }
        
        // For scalar=1, result should equal input point (identity under scalar mul)
        expect(result[0].x.toString()).to.equal(points[0].x.toString());
        expect(result[0].y.toString()).to.equal(points[0].y.toString());
    });
});
import { expect } from 'chai';
import { Field } from 'o1js';

// Helper to split a 256-bit bigint into 8 u32 limbs (little-endian)
function bigintToLimbs(x: bigint) {
    const limbs = [];
    for (let i = 0; i < 8; i++) {
        limbs.push(Number(x & 0xffffffffn));
        x >>= 32n;
    }
    return limbs;
}

// Helper to format limbs for WGSL with zero-padding, 4 limbs per line
function formatLimbsWGSL4x(limbs: number[]) {
    const lines: string[] = [];
    for (let i = 0; i < 8; i += 4) {
        const line = limbs
            .slice(i, i + 4)
            .map((n) => '0x' + n.toString(16).toUpperCase().padStart(8, '0') + 'u')
            .join(', ');
        lines.push('    ' + line);
    }
    return lines.join(',\n');
}

describe('Pallas curve constants', () => {
    const P = Field.ORDER;
    const R = 1n << 256n;
    const R2 = (R * R) % P;
    const P_MINUS_2 = P - 2n;
    const R_MOD_P = (1n << 256n) % P;

    // WGSL PALLAS_P for comparison (little-endian)
    const WGSL_P = [
        0x00000001, 0x992d30ed, 0x094cf91b, 0x224698fc,
        0x00000000, 0x00000000, 0x00000000, 0x40000000,
    ];

    it('checks prime P', () => {
        console.log('Computed P =', P.toString());
        expect(P.toString()).to.equal(P.toString()); // just print for info
    });

    it('checks WGSL PALLAS_P against Field.ORDER', () => {
        const wgslPBigInt = WGSL_P.reduceRight(
            (acc, limb) => (acc << 32n) + BigInt(limb),
            0n
        );
        console.log('WGSL P as BigInt =', wgslPBigInt.toString());
        console.log('Field.ORDER:', P.toString());
        expect(P.toString()).to.equal(wgslPBigInt.toString());
    });

    it('computes R² mod P', () => {
        console.log('Computed R² mod P =', R2.toString());
    });

    it('computes P-2', () => {
        console.log('Computed P-2 =', P_MINUS_2.toString());
    });

    it('generates WGSL constants (4 limbs per line)', () => {
        console.log('// PALLAS_P');
        console.log(`const PALLAS_P: array<u32, 8> = array<u32, 8>(\n${formatLimbsWGSL4x(bigintToLimbs(P))}\n);`);

        console.log('// PALLAS_R2');
        console.log(`const PALLAS_R2: array<u32, 8> = array<u32, 8>(\n${formatLimbsWGSL4x(bigintToLimbs(R2))}\n);`);

        console.log('// PALLAS_P_MINUS_2');
        console.log(`const PALLAS_P_MINUS_2: array<u32, 8> = array<u32, 8>(\n${formatLimbsWGSL4x(bigintToLimbs(P_MINUS_2))}\n);`);

        console.log('// PALLAS_R_MOD_P');
        console.log(`const PALLAS_R_MOD_P: array<u32, 8> = array<u32, 8>(\n${formatLimbsWGSL4x(bigintToLimbs(R_MOD_P))}\n);`);
    });
});

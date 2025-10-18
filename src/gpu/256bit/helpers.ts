// Helper to convert bigint to 256-bit u32 array (little-endian, 8 limbs)
export function bigint256ToLimbs(value: bigint): Uint32Array {
    const limbs = new Uint32Array(8);
    let v = value;
    for (let i = 0; i < 8; i++) {
        limbs[i] = Number(v & 0xFFFFFFFFn);
        v >>= 32n;
    }
    return limbs;
}

// Helper to convert 256-bit u32 array back to bigint
export function limbs256ToBigint(limbs: Uint32Array): bigint {
    let result = 0n;
    for (let i = 7; i >= 0; i--) {
        result = (result << 32n) | BigInt(limbs[i]);
    }
    return result;
}

export const LIMBS_PER_ELEMENT_256 = 8;
export const BYTES_PER_LIMB = 4;
export const BYTES_PER_ELEMENT_256 = LIMBS_PER_ELEMENT_256 * BYTES_PER_LIMB;

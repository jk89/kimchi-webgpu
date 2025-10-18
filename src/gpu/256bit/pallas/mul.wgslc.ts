import importTypes from '../types.wgsl';
import importArithmetic256 from '../arithmetic.wgsl';
import importPallas from './curve.wgsl';

export default `
${importTypes}
${importArithmetic256}
${importPallas}

@group(0) @binding(0) var<storage, read> a: array<Limbs256>;
@group(0) @binding(1) var<storage, read> b: array<Limbs256>;
@group(0) @binding(2) var<storage, read_write> out: array<Limbs256>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&a)) { return; }

    // Convert inputs to Montgomery form
    let a_mont = to_montgomery_256(
        a[idx].limbs,
        PALLAS_CURVE.r2,
        PALLAS_CURVE.mont_inv32,
        PALLAS_CURVE.p
    );

    let b_mont = to_montgomery_256(
        b[idx].limbs,
        PALLAS_CURVE.r2,
        PALLAS_CURVE.mont_inv32,
        PALLAS_CURVE.p
    );

    // Montgomery multiplication
    let prod_mont = mont_mul_256(
        a_mont,
        b_mont,
        PALLAS_CURVE.mont_inv32,
        PALLAS_CURVE.p
    );

    // Convert back from Montgomery form
    out[idx].limbs = from_montgomery(
        prod_mont,
        PALLAS_CURVE.mont_inv32,
        PALLAS_CURVE.p
    );
}
`;

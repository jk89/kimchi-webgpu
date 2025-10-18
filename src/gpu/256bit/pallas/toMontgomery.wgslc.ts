import importTypes from '../types.wgsl';
import importArithmetic256 from '../arithmetic.wgsl';
import importCurve from './curve.wgsl';

export default `
${importTypes}
${importArithmetic256}
${importCurve}

@group(0) @binding(0) var<storage, read> a: array<Limbs256>;
@group(0) @binding(1) var<storage, read_write> out: array<Limbs256>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&a)) { return; }

    out[idx].limbs = to_montgomery_256(
        a[idx].limbs,
        PALLAS_CURVE.r2,
        PALLAS_CURVE.mont_inv32,
        PALLAS_CURVE.p
    );
}
`;

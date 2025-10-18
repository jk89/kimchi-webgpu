import importTypes from '../types.wgsl';
import importArithmetic256 from '../arithmetic.wgsl';
import importPallas from './pallas.wgsl';

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

    out[idx].limbs = sub_mod_256(a[idx].limbs, b[idx].limbs, PALLAS_CURVE.p);
}
`;

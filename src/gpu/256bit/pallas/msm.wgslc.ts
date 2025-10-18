import importTypes from '../types.wgsl';
import importArithmetic256 from '../arithmetic.wgsl';
import importPallas from './curve.wgsl';

export default `
${importTypes}
${importArithmetic256}
${importPallas}

@group(0) @binding(0) var<storage, read> scalars: array<Limbs256>;
@group(0) @binding(1) var<storage, read> P_x: array<Limbs256>;
@group(0) @binding(2) var<storage, read> P_y: array<Limbs256>;
@group(0) @binding(3) var<storage, read_write> Q_x: array<Limbs256>;
@group(0) @binding(4) var<storage, read_write> Q_y: array<Limbs256>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    
    if (idx >= arrayLength(&scalars)) {
        return;
    }
    
    let Q = curve_scalar_mul(scalars[idx], P_x[idx], P_y[idx], PALLAS_CURVE);
    Q_x[idx] = Q.x;
    Q_y[idx] = Q.y;
}
`;

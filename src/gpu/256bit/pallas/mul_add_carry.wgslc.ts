import importTypes from '../types.wgsl';
import importArithmetic256 from '../arithmetic.wgsl';
import importPallas from './curve.wgsl';

export default `
${importTypes}
${importArithmetic256}
${importPallas}

// Inputs: a, b, acc, carry in separate buffers
@group(0) @binding(0) var<storage, read> aBuffer: array<u32>;
@group(0) @binding(1) var<storage, read> bBuffer: array<u32>;
@group(0) @binding(2) var<storage, read_write> accCarryBuffer: array<vec2<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&aBuffer)) { return; }

    let a = aBuffer[idx];
    let b = bBuffer[idx];
    let acc = accCarryBuffer[idx].x;
    var carry = accCarryBuffer[idx].y;

    let res = mul_add_carry(a, b, acc, &carry);

    accCarryBuffer[idx] = vec2<u32>(res, carry);
}
`;

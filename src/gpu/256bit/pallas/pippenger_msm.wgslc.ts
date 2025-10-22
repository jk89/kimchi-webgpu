/*
Shader based reduction of multi-scalar multiplication (MSM) into double and add EC operations.
https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm

=====================================================================================================

Overview:
This implementation processes multi-scalar multiplication using Pippenger's bucket method.
The algorithm is structured in passes (A, Bi_1, Bi_2, C) that may execute multiple times
depending on memory constraints.

Batching Strategy:
If the total number of scalar/point pairs (N_TOTAL) exceeds available GPU memory, 
the computation is split into batches of size BATCH_SIZE:

For each bucket_idx (1 to NUMBER_OF_BUCKETS):
  - Initialize B[bucket_idx] = infinity (z.limbs[i] = 0u for all i)
  - For each batch (batch_offset = 0, BATCH_SIZE, 2*BATCH_SIZE, ..., N_TOTAL):
    - Run Pass A: Convert this batch's affine points to projective
    - Run Pass Bi_1: Accumulate batch contributions into workgroup buffers
    - Run Pass Bi_2: Reduce and accumulate into B[bucket_idx]

Once all batches for all buckets are complete, run Pass C for final aggregation.

Example: N_TOTAL=1M, BATCH_SIZE=100K, NUMBER_OF_BUCKETS=255
  → 255 buckets × 10 batches = 2,550 (A + Bi_1 + Bi_2) cycles, then 1 Pass C

=====================================================================================================

Pass A — Affine → Projective Conversion

Purpose: Convert affine inputs to projective (Montgomery) form once per batch.

Method: 
    For each thread t with global index idx:
        temp = to_projective_256(x[idx], y[idx], PALLAS_CURVE.r2, PALLAS_CURVE.mont_inv32, PALLAS_CURVE.p)
        Px[idx] = temp.x
        Py[idx] = temp.y
        Pz[idx] = temp.z

Input Buffers (global):

    - x — length N, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total N × 32 B

    - y — length N, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total N × 32 B

Output Buffers (global):

    - Px — length N, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total N × 32 B

    - Py — length N, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total N × 32 B

    - Pz — length N, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total N × 32 B
    (P_x,y,z[idx] initialized with Pz[idx] = PALLAS_CURVE.r_mod_p, the Montgomery form of 1)

-----------------------------------------------------------------------------------------------------

Pass Bi_1:

Purpose: Decompose scalar contribution over all points/scalar pairs (N) for a particular bucket.

Method: 

    1. Zero out local WGL_x,y,z buffers:
       For each thread t:
         WGLx[t].limbs[i] = 0u for all i
         WGLy[t].limbs[i] = 0u for all i
         WGLz[t].limbs[i] = 0u for all i (projective infinity)
       workgroupBarrier()
    
    2. Each thread t with global index idx extracts k_ij from k[idx]:
       - bit_offset = bucket_idx * BUCKET_WIDTH_BITS
       - limb_index = bit_offset / 32u
       - bit_in_limb = bit_offset % 32u
       - mask = (1u << BUCKET_WIDTH_BITS) - 1u
       
       - If bit_in_limb + BUCKET_WIDTH_BITS <= 32u:
           k_ij = (k[idx].limbs[limb_index] >> bit_in_limb) & mask
       - Else:
           bits_in_first_limb = 32u - bit_in_limb
           low_bits = k[idx].limbs[limb_index] >> bit_in_limb
           high_bits = k[idx].limbs[limb_index + 1u] << bits_in_first_limb
           k_ij = (low_bits | high_bits) & mask
       
       - If k_ij == bucket_idx:
           WGLx[t] = Px[idx]
           WGLy[t] = Py[idx]
           WGLz[t] = Pz[idx]
       - Else:
           Leave WGL_x,y,z[t] as infinity (already zeroed in step 1)
       
       workgroupBarrier()
    
    3. Tree reduce WGL_x,y,z using binary halving:
       For stride = WORKGROUP_SIZE/2 down to 1:
         if t < stride:
           temp = point_add_proj_256(
             ProjectivePoint256(WGLx[t], WGLy[t], WGLz[t]), 
             ProjectivePoint256(WGLx[t + stride], WGLy[t + stride], WGLz[t + stride]),
             PALLAS_CURVE.r2, PALLAS_CURVE.mont_inv32, PALLAS_CURVE.p
           )
           WGLx[t] = temp.x
           WGLy[t] = temp.y
           WGLz[t] = temp.z
         workgroupBarrier()
       After loop: WGL_x,y,z[0] contains the reduced workgroup point
    
    4. Thread 0 writes WGL_x,y,z[0] to WGG_x,y,z[workgroup_id]:
       if t == 0:
         WGGx[workgroup_id] = WGLx[0]
         WGGy[workgroup_id] = WGLy[0]
         WGGz[workgroup_id] = WGLz[0]

Input Buffers (global):

    - bucket_idx — uniform u32, updated per shader dispatch (1 to NUMBER_OF_BUCKETS)
    
    - BUCKET_WIDTH_BITS — uniform u32, number of bits per bucket (typically 8-16)
    
    - k — length N, array of Limbs256 (scalars mod r), each element = 8 × u32 limbs = 32 bytes → total N × 32 B

    - Px — length N, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total N × 32 B

    - Py — length N, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total N × 32 B

    - Pz — length N, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total N × 32 B

Workgroup buffers (local):

    - WGLx — length WORKGROUP_SIZE, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total WORKGROUP_SIZE × 32 B

    - WGLy — length WORKGROUP_SIZE, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total WORKGROUP_SIZE × 32 B

    - WGLz — length WORKGROUP_SIZE, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total WORKGROUP_SIZE × 32 B

Output Buffers (global):

    - WGGx — length NUM_WORKGROUPS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUM_WORKGROUPS × 32 B

    - WGGy — length NUM_WORKGROUPS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUM_WORKGROUPS × 32 B

    - WGGz — length NUM_WORKGROUPS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUM_WORKGROUPS × 32 B

-----------------------------------------------------------------------------------------------------

Pass Bi_2:

Purpose: Tree reduce WGG_x,y,z global buffer to a single point and accumulate into B[bucket_idx].

Method:

    - Tree reduce global WGG_x,y,z buffers using binary halving:
      For stride = NUM_WORKGROUPS/2 down to 1:
        if workgroup_id < stride:
          temp = point_add_proj_256(
            ProjectivePoint256(WGGx[workgroup_id], WGGy[workgroup_id], WGGz[workgroup_id]),
            ProjectivePoint256(WGGx[workgroup_id + stride], WGGy[workgroup_id + stride], WGGz[workgroup_id + stride]),
            PALLAS_CURVE.r2, PALLAS_CURVE.mont_inv32, PALLAS_CURVE.p
          )
          WGGx[workgroup_id] = temp.x
          WGGy[workgroup_id] = temp.y
          WGGz[workgroup_id] = temp.z
        barrier()
      After loop: WGG_x,y,z[0] contains the reduced point for all workgroups
    
    - Workgroup 0, thread 0 performs:
      if workgroup_id == 0 && thread_id == 0:
        B[bucket_idx] = point_add_proj_256(B[bucket_idx], ProjectivePoint256(WGGx[0], WGGy[0], WGGz[0]), PALLAS_CURVE.r2, PALLAS_CURVE.mont_inv32, PALLAS_CURVE.p)
      
      This accumulates contributions across multiple batches when N is too large to process at once.

Input Buffers (global):

    - WGGx — length NUM_WORKGROUPS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUM_WORKGROUPS × 32 B

    - WGGy — length NUM_WORKGROUPS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUM_WORKGROUPS × 32 B

    - WGGz — length NUM_WORKGROUPS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUM_WORKGROUPS × 32 B

Output Buffers (global):

    - B — length NUMBER_OF_BUCKETS, array of ProjectivePoint256, each element = 3 × Limbs256 = 96 bytes → total NUMBER_OF_BUCKETS × 96 B

Pass Bi_1 and Bi_2 continue until all scalar/point pairs and all buckets have been processed.

-----------------------------------------------------------------------------------------------------

Pass C:

Purpose: Aggregate all buckets using running sum technique, then convert to affine.

Method:
    
    running_sum = infinity (ProjectivePoint256 with z.limbs[i] = 0u for all i)
    result = infinity (ProjectivePoint256 with z.limbs[i] = 0u for all i)
    
    for bucket_idx from NUMBER_OF_BUCKETS-1 down to 1:
        running_sum = point_add_proj_256(running_sum, B[bucket_idx], PALLAS_CURVE.r2, PALLAS_CURVE.mont_inv32, PALLAS_CURVE.p)
        result = point_add_proj_256(result, running_sum, PALLAS_CURVE.r2, PALLAS_CURVE.mont_inv32, PALLAS_CURVE.p)
    
    final_point = to_affine_256(result, PALLAS_CURVE.r2, PALLAS_CURVE.mont_inv32, PALLAS_CURVE.p, PALLAS_CURVE.p_minus_2)
    
    Output final_point.x and final_point.y

Input Buffers (global):
    - B — length NUMBER_OF_BUCKETS, array of ProjectivePoint256, each element = 3 × Limbs256 = 96 bytes → total NUMBER_OF_BUCKETS × 96 B

Output Buffers (global):
    - final_point — Point256 (affine coordinates)
      final_point.x — Limbs256, 8 × u32 limbs = 32 bytes
      final_point.y — Limbs256, 8 × u32 limbs = 32 bytes

*/

import importTypes from '../types.wgsl';
import importArithmetic256 from '../arithmetic.wgsl';
import importPallas from './curve.wgsl';

const pippengerShaderPassAProjectiveConversion = `
${importTypes}
${importArithmetic256}
${importPallas}

@group(0) @binding(0) var<storage, read> x: array<Limbs256>;
@group(0) @binding(1) var<storage, read> y: array<Limbs256>;
@group(0) @binding(2) var<storage, read_write> Px: array<Limbs256>;
@group(0) @binding(3) var<storage, read_write> Py: array<Limbs256>;
@group(0) @binding(4) var<storage, read_write> Pz: array<Limbs256>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    
    if (idx >= arrayLength(&x)) {
        return;
    }
    
    let P = to_projective_256(x[idx], y[idx], P_y[idx], PALLAS_CURVE.r2, PALLAS_CURVE.mont_inv32, PALLAS_CURVE.p);
    P_x[idx] = P.x;
    P_y[idx] = P.y;
    P_z[idx] = P.z;
}
`;

// Check local id use below
const pippengerShaderPassBi1BucketScalarWeightedPointContribution = `
${importTypes}
${importArithmetic256}
${importPallas}

@group(0) @binding(0) var<uniform, read> BUCKET_WIDTH_BITS: u32;

@group(1) @binding(0) var<uniform, read> bucket_idx: u32;

@group(2) @binding(0) var<storage, read> k: array<Limbs256>;
@group(2) @binding(1) var<storage, read> Px: array<Limbs256>;
@group(2) @binding(2) var<storage, read> Py: array<Limbs256>;
@group(2) @binding(3) var<storage, read> Pz: array<Limbs256>;

@group(3) @binding(1) var<storage, read_write> WGGx: array<Limbs256>;
@group(3) @binding(2) var<storage, read_write> WGGy: array<Limbs256>;
@group(3) @binding(3) var<storage, read_write> WGGz: array<Limbs256>;

var<workgroup> WGLx: array<Limbs256, 64>;
var<workgroup> WGLy: array<Limbs256, 64>;
var<workgroup> WGLz: array<Limbs256, 64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wgid: vec3<u32>) {
    let idx = gid.x;
    let workgroup_idx = wgid.x;

    if (idx >= arrayLength(&k)) {
        return;
    }

    // Compute k_ij by extracting the bits corresponding to this bucket from k[idx]

    let bit_offset = bucket_idx * BUCKET_WIDTH_BITS;
    let limb_index = bit_offset / 32u;
    let bit_in_limb = bit_offset % 32u;
    let mask = (1u << BUCKET_WIDTH_BITS) - 1u;

    let k_ij = 0u;
    if (bit_in_limb + BUCKET_WIDTH_BITS) <= 32u {
        k_ij = (k[idx].limbs[limb_index] >> bit_in_limb) & mask;
    }
    else {
        bits_in_first_limb = 32u - bit_in_limb;
        low_bits = k[idx].limbs[limb_index] >> bit_in_limb;
        high_bits = k[idx].limbs[limb_index + 1u] << bits_in_first_limb;
        k_ij = (low_bits | high_bits) & mask;
    }

    if (k_ij == bucket_idx) {
        WGLx[idx] = Px[idx];
        WGLy[idx] = Py[idx];
        WGLz[idx] = Pz[idx];
    }
    else {
        WGLx[idx] = 0u;
        WGLy[idx] = 0u;
        WGLz[idx] = 0u;
    }
    workgroupBarrier();

    // Tree reduce the workgroup memory bucket values by binary halving.

    let WORKGROUP_SIZE = 64;
    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (idx < stride) {
           let temp = point_add_proj_256(
              ProjectivePoint256(WGLx[idx], WGLy[idx], WGLz[idx]),
              ProjectivePoint256(WGLx[idx + stride], WGLy[idx + stride], WGLz[idx + stride]),
           );
           WGLx[idx] = temp.x;
           WGLy[idx] = temp.y;
           WGLz[idx] = temp.z;
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Set the global buffer based on the value of WGL_x,y,z[0] which contains the reduction.

    if (idx == 0) {
        WGGx[workgroup_idx] = WGLx[0];
        WGGy[workgroup_idx] = WGLy[0];
        WGGz[workgroup_idx] = WGLz[0];
    }
}
`;

const pippengerShaderPassBi2TreeReduceBucket = `
${importTypes}
${importArithmetic256}
${importPallas}

@group(0) @binding(0) var<uniform, read> stride: u32;

@group(1) @binding(0) var<uniform, read> bucket_idx: u32;
@group(1) @binding(1) var<storage, read_write> WGGx: array<Limbs256>;
@group(1) @binding(2) var<storage, read_write> WGGy: array<Limbs256>;
@group(1) @binding(3) var<storage, read_write> WGGz: array<Limbs256>;
@group(1) @binding(4) var<storage, read_write> B: array<ProjectivePoint256>;

var<workgroup> WGLx: array<Limbs256, 64>;
var<workgroup> WGLy: array<Limbs256, 64>;
var<workgroup> WGLz: array<Limbs256, 64>;

const WORKGROUP_SIZE = 128u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec<u32>) {
    let idx = gid.x;

    if (idx >= stride) { return; }

    var temp: ProjectivePoint256;

    //--------------------------
    // Workgroup-local reduction for small strides
    //--------------------------
    if (stride <= workgroup_size) {
        WGLx[idx] = WGGx[idx];
        WGLy[idx] = WGGy[idx];
        WGLz[idx] = WGGz[idx];
        workgroupBarrier();

        var wg_stride = stride;
        while (wg_stride > 1u) {
            let half = wg_stride / 2u;
            if (idx < half) {
                temp = point_add_proj_256(
                    ProjectivePoint256(WGLx[idx], WGLy[idx], WGLz[idx]),
                    ProjectivePoint256(WGLx[idx + half], WGLy[idx + half], WGLz[idx + half]),
                    PALLAS_CURVE.r2,
                    PALLAS_CURVE.mont_inv32,
                    PALLAS_CURVE.p
                );
                WGLx[idx] = temp.x;
                WGLy[idx] = temp.y;
                WGLz[idx] = temp.z;
            }
            workgroupBarrier();
            wg_stride = half;
        }

        // Thread 0 writes final result to B
        if (idx == 0u) {
            temp = B[bucket_idx];
            B[bucket_idx] = point_add_proj_256(
                temp,
                ProjectivePoint256(WGLx[0], WGLy[0], WGLz[0]),
                PALLAS_CURVE.r2,
                PALLAS_CURVE.mont_inv32,
                PALLAS_CURVE.p
            );
        }
        return;
    }

    //--------------------------
    // Global memory reduction for large strides
    //--------------------------
    temp = point_add_proj_256(
        ProjectivePoint256(WGGx[idx], WGGy[idx], WGGz[idx]),
        ProjectivePoint256(WGGx[idx + stride], WGGy[idx + stride], WGGz[idx + stride]),
        PALLAS_CURVE.r2,
        PALLAS_CURVE.mont_inv32,
        PALLAS_CURVE.p
    );
    WGGx[idx] = temp.x;
    WGGy[idx] = temp.y;
    WGGz[idx] = temp.z;
}
`;
// Pass C: Parallel Bucket Aggregation with Point Doubling
// Single shader, single workgroup, 128 threads = 128 buckets
// Each thread scales its bucket, then parallel tree reduction combines them all

const pippengerShaderPassCBucketAggregation = `
${importTypes}
${importArithmetic256}
${importPallas}

@group(0) @binding(0) var<storage, read> B: array<ProjectivePoint256>;
@group(0) @binding(1) var<storage, read_write> final_point_x: Limbs256;
@group(0) @binding(2) var<storage, read_write> final_point_y: Limbs256;

const WORKGROUP_SIZE: u32 = 128u;
var<workgroup> scaled: array<ProjectivePoint256, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let NUM_BUCKETS = arrayLength(&B);
    
    // Step 1: Each thread scales its bucket by its weight
    // B[tid] gets weight (NUM_BUCKETS - tid) to match Pippenger's algorithm
    // This ensures: B[1] has highest weight, B[NUM_BUCKETS-1] has weight 1
    
    if (tid < NUM_BUCKETS && tid > 0u) {
        var weight = NUM_BUCKETS - tid;
        var accumulator = ProjectivePoint256(
            Limbs256(array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)),
            Limbs256(array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)),
            Limbs256(array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u))
        );
        var temp = B[tid];
        
        // Binary scalar multiplication: compute weight * B[tid]
        // Process each bit of weight from LSB to MSB
        while (weight > 0u) {
            if ((weight & 1u) != 0u) {
                accumulator = point_add_proj_256(
                    accumulator,
                    temp,
                    PALLAS_CURVE.r2,
                    PALLAS_CURVE.mont_inv32,
                    PALLAS_CURVE.p
                );
            }
            weight = weight >> 1u;
            if (weight > 0u) {
                temp = point_double_proj_256(
                    temp,
                    PALLAS_CURVE.r2,
                    PALLAS_CURVE.mont_inv32,
                    PALLAS_CURVE.p
                );
            }
        }
        
        scaled[tid] = accumulator;
    } else {
        // Thread 0 or out of bounds set a projective infinity.
        scaled[tid] = ProjectivePoint256(
            Limbs256(array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)),
            Limbs256(array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)),
            Limbs256(array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u))
        );
    }
    
    workgroupBarrier();
    
    // Step 2: Tree reduction to sum all scaled buckets
    // Binary tree: stride goes 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            scaled[tid] = point_add_proj_256(
                scaled[tid],
                scaled[tid + stride],
                PALLAS_CURVE.r2,
                PALLAS_CURVE.mont_inv32,
                PALLAS_CURVE.p
            );
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Step 3: Thread 0 converts result to affine and writes output
    if (tid == 0u) {
        let final_affine = to_affine_256(
            scaled[0],
            PALLAS_CURVE.r2,
            PALLAS_CURVE.mont_inv32,
            PALLAS_CURVE.p,
            PALLAS_CURVE.p_minus_2
        );
        
        final_point_x = final_affine.x;
        final_point_y = final_affine.y;
    }
}
`;

export {
    pippengerShaderPassAProjectiveConversion,
    pippengerShaderPassBi1BucketScalarWeightedPointContribution,
    pippengerShaderPassBi2TreeReduceBucket,
    pippengerShaderPassCBucketAggregation,
};

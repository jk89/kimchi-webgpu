/*
Shader based reduction of multi-scalar multiplication (MSM) into double and add EC operations.
https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm

=====================================================================================================

Overview:
This implementation processes multi-scalar multiplication using Pippenger's bucket method.
The algorithm is structured in passes (A, Bi_1, Bi_2, C, D, E) that execute in sequence,
with passes potentially repeating based on memory constraints and batch sizing.

Algorithm Structure:

The MSM computation Σ(k_i · P_i) for i ∈ [0, N_TOTAL) is performed as follows:

1. **Bucket Decomposition**: Each scalar k_i is decomposed into buckets based on BUCKET_WIDTH_BITS.
   For example, with 8-bit buckets, a 256-bit scalar yields 32 buckets (256/8 = 32).
   Each bucket j contains the sum of all points P_i where bits [j*8, (j+1)*8) of k_i equal j.

2. **Weighted Accumulation**: Buckets are weighted and accumulated using Pippenger's optimization,
   where bucket weights follow the pattern: weight(bucket_j) = NUMBER_OF_BUCKETS - j.

3. **Batching**: If N_TOTAL exceeds GPU memory capacity, computation splits into batches of size N.
   Each batch produces a batch_final_point, which are later accumulated to produce the final result.

Pass Flow:

For a single batch of N points:
  → Pass A: Convert affine input points to projective (Montgomery) form
  → For each bucket_idx (1 to NUMBER_OF_BUCKETS):
      → Pass Bi_1: Extract scalar bits, accumulate matching points within workgroups
      → Pass Bi_2: Tree-reduce workgroup results into bucket B[bucket_idx]
  → Pass C: Weight and aggregate all buckets into intermediate results F
  → Pass D: Tree-reduce F into a single batch_final_point (projective)

For multiple batches (N_TOTAL > N):
  → Repeat (A, Bi_1, Bi_2, C, D) for each batch → produces batch_final_point[0..NUM_BATCHES-1]
  → Pass E: Accumulate all batch_final_points and convert final result to affine

Example: N_TOTAL=1M, N=100K, NUMBER_OF_BUCKETS=255, BUCKET_WIDTH_BITS=8
  → NUM_BATCHES = ceil(1M / 100K) = 10 batches
  → Per batch: 1 Pass A + (255 × Bi_1 + 255 × Bi_2) + 1 Pass C + Pass D reductions
  → Total: 10 batches × (A + 255×Bi_1 + 255×Bi_2 + C + D) = 10 batch_final_points
  → Then: Pass E reduces 10 batch_final_points → 1 final affine point

Memory Management:

The host (CPU) manages:
- Batching N_TOTAL into chunks of size N that fit in GPU memory
- Uniform parameters: n (number of points in current reduction), bucket_idx, batch_idx
- Buffer allocations and dispatch sizes for each pass
- Progressive reduction passes when intermediate results exceed WORKGROUP_SIZE

Key Parameters:
- N: Number of points processed per batch (limited by GPU memory)
- N_TOTAL: Total number of scalar/point pairs in the MSM
- NUMBER_OF_BUCKETS: 2^BUCKET_WIDTH_BITS (typically 256 for 8-bit buckets)
- BUCKET_WIDTH_BITS: Bits per bucket (typically 8-16)
- WORKGROUP_SIZE: GPU workgroup size (64 for Bi_1, 64 for Bi_2/C/D/E)

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
    (Pz[idx] initialized with PALLAS_CURVE.r_mod_p, the Montgomery form of 1)

-----------------------------------------------------------------------------------------------------

Pass Bi_1:

Purpose: Decompose scalar contribution over all points/scalar pairs (N) for a particular bucket.

Method: 

    1. Zero out local WGLx,y,z buffers:
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
           WGLx[idx] = Px[idx]
           WGLy[idx] = Py[idx]
           WGLz[idx] = Pz[idx]
       - Else:
           WGLx[idx] = 0u
           WGLy[idx] = 0u
           WGLz[idx] = 0u
       
       workgroupBarrier()
    
    3. Tree reduce WGLx,y,z using binary halving:
       For stride = WORKGROUP_SIZE/2 down to 1:
         if t < stride:
           temp = point_add_proj_256(
             ProjectivePoint256(WGLx[t], WGLy[t], WGLz[t]), 
             ProjectivePoint256(WGLx[t + stride], WGLy[t + stride], WGLz[t + stride])
           )
           WGLx[t] = temp.x
           WGLy[t] = temp.y
           WGLz[t] = temp.z
         workgroupBarrier()
       After loop: WGLx,y,z[0] contains the reduced workgroup point
    
    4. Thread 0 writes WGLx,y,z[0] to WGGx,y,z[workgroup_id]:
       if t == 0:
         WGGx[workgroup_idx] = WGLx[0]
         WGGy[workgroup_idx] = WGLy[0]
         WGGz[workgroup_idx] = WGLz[0]

Input Buffers (global):

    - BUCKET_WIDTH_BITS — uniform u32, number of bits per bucket (typically 8-16)
    
    - bucket_idx — uniform u32, updated per shader dispatch (1 to NUMBER_OF_BUCKETS)
    
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

Purpose: Tree reduce WGGx,y,z global buffer to a single point and accumulate into B[bucket_idx].

Method:

    1. Load WGGx,y,z into workgroup-local memory WGLx,y,z:
       if idx >= n:
         WGLx[local_idx] = IDENTITY_LIMBS_256
         WGLy[local_idx] = IDENTITY_LIMBS_256
         WGLz[local_idx] = IDENTITY_LIMBS_256
       else:
         WGLx[local_idx] = WGGx[idx]
         WGLy[local_idx] = WGGy[idx]
         WGLz[local_idx] = WGGz[idx]
    
    2. Tree reduce WGLx,y,z using binary halving:
       For stride = WORKGROUP_SIZE/2 down to 1:
         let half = stride >> 1u
         if local_idx < half:
           temp = point_add_proj_256(
             ProjectivePoint256(WGLx[local_idx], WGLy[local_idx], WGLz[local_idx]),
             ProjectivePoint256(WGLx[local_idx + half], WGLy[local_idx + half], WGLz[local_idx + half]),
             PALLAS_CURVE.r2,
             PALLAS_CURVE.mont_inv32,
             PALLAS_CURVE.p
           )
           WGLx[local_idx] = temp.x
           WGLy[local_idx] = temp.y
           WGLz[local_idx] = temp.z
         workgroupBarrier()
         stride = half
       After loop: WGLx,y,z[0] contains the reduced point for this workgroup
    
    3. Thread 0 writes WGLx,y,z[0] back to WGGx,y,z[workgroup_idx]:
       if local_idx == 0:
         WGGx[workgroup_idx] = WGLx[0]
         WGGy[workgroup_idx] = WGLy[0]
         WGGz[workgroup_idx] = WGLz[0]
    
    4. For final pass (n <= WORKGROUP_SIZE), first global thread writes to Bx,y,z[bucket_idx]:
       if idx == 0 && n <= WORKGROUP_SIZE:
         Bx[bucket_idx] = WGGx[0]
         By[bucket_idx] = WGGy[0]
         Bz[bucket_idx] = WGGz[0]
       
       This accumulates contributions across multiple batches when N is too large to process at once.
       The host manages n (uniform parameter representing number of WGG points to reduce in this dispatch).
       Pass Bi_2 may be invoked multiple times with progressively smaller n until n <= WORKGROUP_SIZE.

Input Buffers (global):

    - n — uniform u32, number of WGG points to reduce in this dispatch (managed by host)
    
    - bucket_idx — uniform u32, current bucket index being processed
    
    - WGGx — length NUM_WORKGROUPS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUM_WORKGROUPS × 32 B

    - WGGy — length NUM_WORKGROUPS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUM_WORKGROUPS × 32 B

    - WGGz — length NUM_WORKGROUPS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUM_WORKGROUPS × 32 B

Workgroup buffers (local):

    - WGLx — length WORKGROUP_SIZE (64), array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total 64 × 32 B

    - WGLy — length WORKGROUP_SIZE (64), array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total 64 × 32 B

    - WGLz — length WORKGROUP_SIZE (64), array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total 64 × 32 B

Output Buffers (global):

    - Bx — length NUMBER_OF_BUCKETS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUMBER_OF_BUCKETS × 32 B

    - By — length NUMBER_OF_BUCKETS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUMBER_OF_BUCKETS × 32 B

    - Bz — length NUMBER_OF_BUCKETS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUMBER_OF_BUCKETS × 32 B

Pass Bi_1 and Bi_2 continue until all scalar/point pairs and all buckets have been processed.

-----------------------------------------------------------------------------------------------------

Pass C:

Purpose: Aggregate all buckets using weighted accumulation, then perform tree reduction of results.

Method:
    
    1. Load bucket points Bx,y,z into workgroup-local scaled array:
       if idx >= NUM_BUCKETS:
         scaled[local_idx] = ProjectivePoint256(IDENTITY_LIMBS_256, IDENTITY_LIMBS_256, IDENTITY_LIMBS_256)
       else:
         scaled[local_idx] = ProjectivePoint256(Bx[idx], By[idx], Bz[idx])
    
    2. Each thread scales its bucket by weight (NUM_BUCKETS - idx):
       if idx < NUM_BUCKETS:
         var weight = NUM_BUCKETS - idx
         var accumulator = ProjectivePoint256(IDENTITY_LIMBS_256, IDENTITY_LIMBS_256, IDENTITY_LIMBS_256)
         var temp = scaled[local_idx]
         
         while weight > 0u:
           if (weight & 1u) != 0u:
             accumulator = point_add_proj_256(
               accumulator,
               temp,
               PALLAS_CURVE.r2,
               PALLAS_CURVE.mont_inv32,
               PALLAS_CURVE.p
             )
           weight = weight >> 1u
           if weight > 0u:
             temp = point_double_proj_256(
               temp,
               PALLAS_CURVE.r2,
               PALLAS_CURVE.mont_inv32,
               PALLAS_CURVE.p
             )
         
         scaled[local_idx] = accumulator
       workgroupBarrier()
    
    3. Tree reduce scaled array using binary halving:
       var stride = WORKGROUP_SIZE / 2u
       while stride > 0u:
         if local_idx < stride:
           scaled[local_idx] = point_add_proj_256(
             scaled[local_idx],
             scaled[local_idx + stride],
             PALLAS_CURVE.r2,
             PALLAS_CURVE.mont_inv32,
             PALLAS_CURVE.p
           )
         workgroupBarrier()
         stride = stride / 2u
    
    4. Thread 0 writes scaled[0] to Fx,y,z[workgroup_idx]:
       if local_idx == 0u:
         Fx[workgroup_idx] = scaled[0].x
         Fy[workgroup_idx] = scaled[0].y
         Fz[workgroup_idx] = scaled[0].z

Input Buffers (global):

    - Bx — length NUMBER_OF_BUCKETS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUMBER_OF_BUCKETS × 32 B

    - By — length NUMBER_OF_BUCKETS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUMBER_OF_BUCKETS × 32 B

    - Bz — length NUMBER_OF_BUCKETS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUMBER_OF_BUCKETS × 32 B

Workgroup buffers (local):

    - scaled — length WORKGROUP_SIZE (64), array of ProjectivePoint256, each element = 3 × Limbs256 = 96 bytes → total 64 × 96 B

Output Buffers (global):

    - Fx — length NUM_WORKGROUPS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUM_WORKGROUPS × 32 B

    - Fy — length NUM_WORKGROUPS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUM_WORKGROUPS × 32 B

    - Fz — length NUM_WORKGROUPS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUM_WORKGROUPS × 32 B

-----------------------------------------------------------------------------------------------------

Pass D:

Purpose: Tree reduce Fx,y,z global buffer to final point for the batch.

Method:

    1. Load Fx,y,z into workgroup-local memory WGLx,y,z:
       if idx >= n:
         WGLx[local_idx] = IDENTITY_LIMBS_256
         WGLy[local_idx] = IDENTITY_LIMBS_256
         WGLz[local_idx] = IDENTITY_LIMBS_256
       else:
         WGLx[local_idx] = Fx[idx]
         WGLy[local_idx] = Fy[idx]
         WGLz[local_idx] = Fz[idx]
       workgroupBarrier()
    
    2. Tree reduce WGLx,y,z using binary halving:
       var stride = WORKGROUP_SIZE / 2u
       while stride > 0u:
         if local_idx < stride:
           temp = point_add_proj_256(
             ProjectivePoint256(WGLx[local_idx], WGLy[local_idx], WGLz[local_idx]),
             ProjectivePoint256(WGLx[local_idx + stride], WGLy[local_idx + stride], WGLz[local_idx + stride]),
             PALLAS_CURVE.r2,
             PALLAS_CURVE.mont_inv32,
             PALLAS_CURVE.p
           )
           WGLx[local_idx] = temp.x
           WGLy[local_idx] = temp.y
           WGLz[local_idx] = temp.z
         workgroupBarrier()
         stride = stride / 2u
    
    3. Thread 0 writes WGLx,y,z[0] back to Fx,y,z[workgroup_idx]:
       if local_idx == 0u:
         Fx[workgroup_idx] = WGLx[0]
         Fy[workgroup_idx] = WGLy[0]
         Fz[workgroup_idx] = WGLz[0]

    4. For final pass (n <= WORKGROUP_SIZE), first global thread writes projective batch final point:
       if idx == 0u && n <= WORKGROUP_SIZE:
         batch_final_points_x[batch_idx] = Fx[0]
         batch_final_points_y[batch_idx] = Fy[0]
         batch_final_points_z[batch_idx] = Fz[0]
       
       The host manages n (uniform parameter representing number of F points to reduce in this dispatch)
       and batch_idx (uniform parameter identifying which batch result to store).
       Pass D may be invoked multiple times with progressively smaller n until n <= WORKGROUP_SIZE.
       
       When processing N_TOTAL > N, each batch produces one entry in batch_final_points_x,y,z.
       After all batches complete, batch_final_points array contains all batch results in projective form.
       A final reduction pass (external or additional shader) accumulates all batch_final_points and 
       converts the final accumulated result to affine coordinates.

Input Buffers (global):

    - n — uniform u32, number of F points to reduce in this dispatch (managed by host)
    
    - Fx — length NUM_WORKGROUPS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUM_WORKGROUPS × 32 B

    - Fy — length NUM_WORKGROUPS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUM_WORKGROUPS × 32 B

    - Fz — length NUM_WORKGROUPS, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total NUM_WORKGROUPS × 32 B

Workgroup buffers (local):

    - WGLx — length WORKGROUP_SIZE (64), array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total 64 × 32 B

    - WGLy — length WORKGROUP_SIZE (64), array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total 64 × 32 B

    - WGLz — length WORKGROUP_SIZE (64), array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total 64 × 32 B

Output Buffers (global):

    - batch_final_points_x — array of Limbs256, length = number of batches, each element = 8 × u32 limbs = 32 bytes

    - batch_final_points_y — array of Limbs256, length = number of batches, each element = 8 × u32 limbs = 32 bytes

    - batch_final_points_z — array of Limbs256, length = number of batches, each element = 8 × u32 limbs = 32 bytes

-----------------------------------------------------------------------------------------------------

Pass E:

Purpose: Accumulate all batch final points and convert the final result to affine coordinates.

Method:

    1. Load batch_final_points_x,y,z into workgroup-local memory WGLx,y,z:
       if idx >= n:
         WGLx[local_idx] = IDENTITY_LIMBS_256
         WGLy[local_idx] = IDENTITY_LIMBS_256
         WGLz[local_idx] = IDENTITY_LIMBS_256
       else:
         WGLx[local_idx] = batch_final_points_x[idx]
         WGLy[local_idx] = batch_final_points_y[idx]
         WGLz[local_idx] = batch_final_points_z[idx]
       workgroupBarrier()
    
    2. Tree reduce WGLx,y,z using binary halving:
       var stride = WORKGROUP_SIZE / 2u
       while stride > 0u:
         if local_idx < stride:
           temp = point_add_proj_256(
             ProjectivePoint256(WGLx[local_idx], WGLy[local_idx], WGLz[local_idx]),
             ProjectivePoint256(WGLx[local_idx + stride], WGLy[local_idx + stride], WGLz[local_idx + stride]),
             PALLAS_CURVE.r2,
             PALLAS_CURVE.mont_inv32,
             PALLAS_CURVE.p
           )
           WGLx[local_idx] = temp.x
           WGLy[local_idx] = temp.y
           WGLz[local_idx] = temp.z
         workgroupBarrier()
         stride = stride / 2u
    
    3. Thread 0 writes WGLx,y,z[0] back to batch_final_points_x,y,z[workgroup_idx]:
       if local_idx == 0u:
         batch_final_points_x[workgroup_idx] = WGLx[0]
         batch_final_points_y[workgroup_idx] = WGLy[0]
         batch_final_points_z[workgroup_idx] = WGLz[0]
    
    4. For final pass (n <= WORKGROUP_SIZE), first global thread converts to affine and writes final point:
       if idx == 0u && n <= WORKGROUP_SIZE:
         finalProj = ProjectivePoint256(batch_final_points_x[0], batch_final_points_y[0], batch_final_points_z[0])
         finalAffine = point_to_affine_256(finalProj, PALLAS_CURVE.p)
         final_point_x = finalAffine.x
         final_point_y = finalAffine.y
       
       The host manages n (uniform parameter representing number of batch final points to reduce).
       Pass E may be invoked multiple times with progressively smaller n until n <= WORKGROUP_SIZE,
       at which point the final affine point is written.
       
       This pass only runs after all batches have completed and all batch_final_points have been computed.
       It accumulates all batch results and produces the final MSM result in affine coordinates.

Input Buffers (global):

    - n — uniform u32, number of batch final points to reduce in this dispatch (managed by host)
    
    - batch_final_points_x — array of Limbs256, length = number of batches, each element = 8 × u32 limbs = 32 bytes

    - batch_final_points_y — array of Limbs256, length = number of batches, each element = 8 × u32 limbs = 32 bytes

    - batch_final_points_z — array of Limbs256, length = number of batches, each element = 8 × u32 limbs = 32 bytes

Workgroup buffers (local):

    - WGLx — length WORKGROUP_SIZE (64), array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total 64 × 32 B

    - WGLy — length WORKGROUP_SIZE (64), array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total 64 × 32 B

    - WGLz — length WORKGROUP_SIZE (64), array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total 64 × 32 B

Output Buffers (global):

    - final_point_x — Limbs256, 8 × u32 limbs = 32 bytes

    - final_point_y — Limbs256, 8 × u32 limbs = 32 bytes

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

const WORKGROUP_SIZE: u32 = 64u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    
    if (idx >= arrayLength(&x)) {
        return;
    }
    
    let P = to_projective_256(x[idx], y[idx], PALLAS_CURVE.r2, PALLAS_CURVE.mont_inv32, PALLAS_CURVE.p);
    Px[idx] = P.x;
    Py[idx] = P.y;
    Pz[idx] = P.z;
}
`;

const pippengerShaderPassBi1BucketScalarWeightedPointContribution = `
${importTypes}
${importArithmetic256}
${importPallas}

@group(0) @binding(0) var<uniform> BUCKET_WIDTH_BITS: u32;

@group(1) @binding(0) var<uniform> bucket_idx: u32;

@group(2) @binding(0) var<storage, read> k: array<Limbs256>;
@group(2) @binding(1) var<storage, read> Px: array<Limbs256>;
@group(2) @binding(2) var<storage, read> Py: array<Limbs256>;
@group(2) @binding(3) var<storage, read> Pz: array<Limbs256>;

@group(3) @binding(1) var<storage, read_write> WGGx: array<Limbs256>;
@group(3) @binding(2) var<storage, read_write> WGGy: array<Limbs256>;
@group(3) @binding(3) var<storage, read_write> WGGz: array<Limbs256>;

const WORKGROUP_SIZE: u32 = 64u;

var<workgroup> WGLx: array<Limbs256, 64>;
var<workgroup> WGLy: array<Limbs256, 64>;
var<workgroup> WGLz: array<Limbs256, 64>;

@compute @workgroup_size(WORKGROUP_SIZE)
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

    var k_ij = 0u;
    if (bit_in_limb + BUCKET_WIDTH_BITS) <= 32u {
        k_ij = (k[idx].limbs[limb_index] >> bit_in_limb) & mask;
    }
    else {
        let bits_in_first_limb = 32u - bit_in_limb;
        let low_bits = k[idx].limbs[limb_index] >> bit_in_limb;
        let high_bits = k[idx].limbs[limb_index + 1u] << bits_in_first_limb;
        k_ij = (low_bits | high_bits) & mask;
    }

    if (k_ij == bucket_idx) {
        WGLx[idx] = Px[idx];
        WGLy[idx] = Py[idx];
        WGLz[idx] = Pz[idx];
    }
    else {
        WGLx[idx] = IDENTITY_LIMBS_256;
        WGLy[idx] = IDENTITY_LIMBS_256;
        WGLz[idx] = IDENTITY_LIMBS_256;
    }
    workgroupBarrier();

    // Tree reduce the workgroup memory bucket values by binary halving.

    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (idx < stride) {
           let temp = point_add_proj_256(
              ProjectivePoint256(WGLx[idx], WGLy[idx], WGLz[idx]),
              ProjectivePoint256(WGLx[idx + stride], WGLy[idx + stride], WGLz[idx + stride]),
              PALLAS_CURVE.r2,
              PALLAS_CURVE.mont_inv32,
              PALLAS_CURVE.p
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

@group(0) @binding(0) var<uniform> n: u32;

@group(1) @binding(0) var<uniform> bucket_idx: u32;

@group(2) @binding(0) var<storage, read_write> WGGx: array<Limbs256>;
@group(2) @binding(1) var<storage, read_write> WGGy: array<Limbs256>;
@group(2) @binding(2) var<storage, read_write> WGGz: array<Limbs256>;

@group(3) @binding(0) var<storage, read_write> Bx: array<Limbs256>;
@group(3) @binding(1) var<storage, read_write> By: array<Limbs256>;
@group(3) @binding(2) var<storage, read_write> Bz: array<Limbs256>;

// 128 is an upper limit if we consider how much local memory there is 16,384 (16 KB)
// We have 3, 8 limbs points so that 3 (points) * 8 (limbs) * 4 (bytes in a u32) * 128 (work group threads) = 12,288 (12.3KB)
// This must be a power of 2 reduce the amount of bound checking we need.
const WORKGROUP_SIZE: u32 = 64u;

var<workgroup> WGLx: array<Limbs256, WORKGROUP_SIZE>;
var<workgroup> WGLy: array<Limbs256, WORKGROUP_SIZE>;
var<workgroup> WGLz: array<Limbs256, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wgid: vec3<u32>) {
    let idx = gid.x;
    let local_idx = lid.x;
    let workgroup_idx = wgid.x;

    // Compute number of workgroups needed for current pass
    let workgroups_needed = (n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE; // Ceil(n/WORKGROUP_SIZE)
    // No threads needed beyond n
    if (workgroup_idx >= workgroups_needed) {
        return; // We don't need these threads
    }
    // Load points into local memory, pad with zero points if out-of-bounds
    if (idx >= n) {
        WGLx[local_idx] = IDENTITY_LIMBS_256;
        WGLy[local_idx] = IDENTITY_LIMBS_256;
        WGLz[local_idx] = IDENTITY_LIMBS_256;
    }
    // Else we have a real point to load
    else {
        WGLx[local_idx] = WGGx[idx];
        WGLy[local_idx] = WGGy[idx];
        WGLz[local_idx] = WGGz[idx];
    }
    
    // Workgroup-local binary reduction
    var stride = WORKGROUP_SIZE / 2;
    while (stride >= 1u) {
        let half = stride >> 1u;
        if (local_idx < half) {
            let temp = point_add_proj_256(
                ProjectivePoint256(WGLx[local_idx], WGLy[local_idx], WGLz[local_idx]),
                ProjectivePoint256(WGLx[local_idx + half], WGLy[local_idx + half], WGLz[local_idx + half]),
                PALLAS_CURVE.r2,
                PALLAS_CURVE.mont_inv32,
                PALLAS_CURVE.p
            );
            WGLx[local_idx] = temp.x;
            WGLy[local_idx] = temp.y;
            WGLz[local_idx] = temp.z;
        }
        workgroupBarrier();
        stride = half;
    }
    
    // Local thread 0 writes the result to WGG post reduction
    if (local_idx == 0) {
        WGGx[workgroup_idx] = WGLx[local_idx];
        WGGy[workgroup_idx] = WGLy[local_idx];
        WGGz[workgroup_idx] = WGLz[local_idx];
    }
    
    // For the final pass, first global thread writes the fully reduced point to B_x,y,z
    if (idx == 0 && n <= WORKGROUP_SIZE) {
        Bx[bucket_idx] = WGGx[0];
        By[bucket_idx] = WGGy[0];
        Bz[bucket_idx] = WGGz[0];
    }
}
`;

const pippengerShaderPassCBucketAggregation = `
${importTypes}
${importArithmetic256}
${importPallas}

// Note that we might have multiple passes of different chunks of B_x,y,z if our point didnt fit into a single 4 million chunk
@group(1) @binding(0) var<storage, read> Bx: array<Limbs256>;
@group(1) @binding(1) var<storage, read> By: array<Limbs256>;
@group(1) @binding(2) var<storage, read> Bz: array<Limbs256>;

@group(2) @binding(0) var<storage, read_write> Fx: array<Limbs256>;
@group(2) @binding(1) var<storage, read_write> Fy: array<Limbs256>;
@group(2) @binding(2) var<storage, read_write> Fz: array<Limbs256>;

const WORKGROUP_SIZE: u32 = 64u;
var<workgroup> scaled: array<ProjectivePoint256, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wgid: vec3<u32>) {
    let idx = gid.x;
    let local_idx = lid.x;
    let workgroup_idx = wgid.x;

    let NUM_BUCKETS = arrayLength(&Bx);

    // Compute number of workgroups needed for current pass
    let workgroups_needed = (NUM_BUCKETS + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE; // Ceil(n/WORKGROUP_SIZE)
    // No threads needed beyond NUM_BUCKETS
    if (workgroup_idx >= workgroups_needed) { 
        return; // We don't need these threads
    }
    // Load points into local memory, pad with zero points if out-of-bounds
    if (idx >= NUM_BUCKETS) {
        scaled[local_idx] = ProjectivePoint256(
            IDENTITY_LIMBS_256,
            IDENTITY_LIMBS_256,
            IDENTITY_LIMBS_256
        );
    }
    // Else we have a real point to load
    else {
        scaled[local_idx] = ProjectivePoint256(Bx[idx], By[idx], Bz[idx]);
    }

    // Next we need the same accumulation technique to weight the points in scaled but this is a bit confusing

    // Step 1: Each thread scales its bucket by its weight
    // B_x,y,z[idx] gets weight (NUM_BUCKETS - idx) to match Pippenger's algorithm
    // This ensures: B[1] has highest weight, B[NUM_BUCKETS-1] has weight 1
    
    if (idx < NUM_BUCKETS) {
        var weight = NUM_BUCKETS - idx;
        var accumulator = ProjectivePoint256(IDENTITY_LIMBS_256, IDENTITY_LIMBS_256, IDENTITY_LIMBS_256);

        var temp = scaled[local_idx];
        
        // Binary scalar multiplication: compute weight * B_x,y,z[local_idx]
        // Process each bit of weight from LSB to MSB
        while (weight > 0u) {
            // Could make this more efficient for weight 1 and 2, doing directly assignment or point double without needing to do the below.
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
        
        scaled[local_idx] = accumulator;
    }
    
    workgroupBarrier();
    
    // Step 2: Tree reduction to sum all scaled buckets
    // Binary tree: stride goes 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (local_idx < stride) {
            scaled[local_idx] = point_add_proj_256(
                scaled[local_idx],
                scaled[local_idx + stride],
                PALLAS_CURVE.r2,
                PALLAS_CURVE.mont_inv32,
                PALLAS_CURVE.p
            );
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Step 3: Thread 0 writes the point to F_x,y,z[workgroup_idx] which will be reduced in the final pass?
    if (local_idx == 0u) {
        Fx[workgroup_idx] = scaled[0].x;
        Fy[workgroup_idx] = scaled[0].y;
        Fz[workgroup_idx] = scaled[0].z;
    }
}
`;

const pippengerShaderPassDTreeReduceFinalPoint = `
${importTypes}
${importArithmetic256}
${importPallas}

// Uniform: number of points to reduce in this dispatch
@group(0) @binding(0) var<uniform> n: u32;
@group(0) @binding(1) var<uniform> batch_idx: u32;

// Storage: partially reduced points from Pass C
@group(1) @binding(0) var<storage, read_write> Fx: array<Limbs256>;
@group(1) @binding(1) var<storage, read_write> Fy: array<Limbs256>;
@group(1) @binding(2) var<storage, read_write> Fz: array<Limbs256>;

// Storage for batch final points (projective) - one per batch processed
@group(2) @binding(0) var<storage, read_write> batch_final_points_x: array<Limbs256>;
@group(2) @binding(1) var<storage, read_write> batch_final_points_y: array<Limbs256>;
@group(2) @binding(2) var<storage, read_write> batch_final_points_z: array<Limbs256>;

const WORKGROUP_SIZE: u32 = 64u;

var<workgroup> WGLx: array<Limbs256, WORKGROUP_SIZE>;
var<workgroup> WGLy: array<Limbs256, WORKGROUP_SIZE>;
var<workgroup> WGLz: array<Limbs256, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wgid: vec3<u32>) {
    let idx = gid.x;
    let local_idx = lid.x;
    let workgroup_idx = wgid.x;

    // Number of workgroups needed
    let workgroups_needed = (n + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    if (workgroup_idx >= workgroups_needed) { 
        return; // We don't need these threads
    }

    // Load points into local workgroup memory
    if (idx >= n) {
        WGLx[local_idx] = IDENTITY_LIMBS_256;
        WGLy[local_idx] = IDENTITY_LIMBS_256;
        WGLz[local_idx] = IDENTITY_LIMBS_256;
    } else {
        WGLx[local_idx] = Fx[idx];
        WGLy[local_idx] = Fy[idx];
        WGLz[local_idx] = Fz[idx];
    }

    workgroupBarrier();

    // Tree reduction within workgroup
    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (local_idx < stride) {
            let temp = point_add_proj_256(
                ProjectivePoint256(WGLx[local_idx], WGLy[local_idx], WGLz[local_idx]),
                ProjectivePoint256(WGLx[local_idx + stride], WGLy[local_idx + stride], WGLz[local_idx + stride]),
                PALLAS_CURVE.r2,
                PALLAS_CURVE.mont_inv32,
                PALLAS_CURVE.p
            );
            WGLx[local_idx] = temp.x;
            WGLy[local_idx] = temp.y;
            WGLz[local_idx] = temp.z;
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Local thread 0 writes reduced result back to global Fx/y/z
    if (local_idx == 0u) {
        Fx[workgroup_idx] = WGLx[0];
        Fy[workgroup_idx] = WGLy[0];
        Fz[workgroup_idx] = WGLz[0];
    }

    // For the final pass (single workgroup), first global thread writes output
    if (idx == 0u && n <= WORKGROUP_SIZE) {
        batch_final_points_x[batch_idx] = Fx[0];
        batch_final_points_y[batch_idx] = Fy[0];
        batch_final_points_z[batch_idx] = Fz[0];
    }
}
`;

const pippengerShaderPassEFinalAccumulation = `
${importTypes}
${importArithmetic256}
${importPallas}

// Uniform: number of batch final points to reduce in this dispatch
@group(0) @binding(0) var<uniform> n: u32;

// Storage: batch final points from Pass D (projective)
@group(1) @binding(0) var<storage, read_write> batch_final_points_x: array<Limbs256>;
@group(1) @binding(1) var<storage, read_write> batch_final_points_y: array<Limbs256>;
@group(1) @binding(2) var<storage, read_write> batch_final_points_z: array<Limbs256>;

// Storage for final affine point
@group(2) @binding(0) var<storage, read_write> final_point_x: Limbs256;
@group(2) @binding(1) var<storage, read_write> final_point_y: Limbs256;

const WORKGROUP_SIZE: u32 = 64u;

var<workgroup> WGLx: array<Limbs256, WORKGROUP_SIZE>;
var<workgroup> WGLy: array<Limbs256, WORKGROUP_SIZE>;
var<workgroup> WGLz: array<Limbs256, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wgid: vec3<u32>) {
    let idx = gid.x;
    let local_idx = lid.x;
    let workgroup_idx = wgid.x;

    // Number of workgroups needed
    let workgroups_needed = (n + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    if (workgroup_idx >= workgroups_needed) { 
        return; // We don't need these threads
    }

    // Load batch final points into local workgroup memory
    if (idx >= n) {
        WGLx[local_idx] = IDENTITY_LIMBS_256;
        WGLy[local_idx] = IDENTITY_LIMBS_256;
        WGLz[local_idx] = IDENTITY_LIMBS_256;
    } else {
        WGLx[local_idx] = batch_final_points_x[idx];
        WGLy[local_idx] = batch_final_points_y[idx];
        WGLz[local_idx] = batch_final_points_z[idx];
    }

    workgroupBarrier();

    // Tree reduction within workgroup
    var stride = WORKGROUP_SIZE / 2u;
    while (stride > 0u) {
        if (local_idx < stride) {
            let temp = point_add_proj_256(
                ProjectivePoint256(WGLx[local_idx], WGLy[local_idx], WGLz[local_idx]),
                ProjectivePoint256(WGLx[local_idx + stride], WGLy[local_idx + stride], WGLz[local_idx + stride]),
                PALLAS_CURVE.r2,
                PALLAS_CURVE.mont_inv32,
                PALLAS_CURVE.p
            );
            WGLx[local_idx] = temp.x;
            WGLy[local_idx] = temp.y;
            WGLz[local_idx] = temp.z;
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Local thread 0 writes reduced result back to global batch_final_points
    if (local_idx == 0u) {
        batch_final_points_x[workgroup_idx] = WGLx[0];
        batch_final_points_y[workgroup_idx] = WGLy[0];
        batch_final_points_z[workgroup_idx] = WGLz[0];
    }

    // For the final pass (single workgroup), first global thread converts to affine and writes final output
    if (idx == 0u && n <= WORKGROUP_SIZE) {
        let finalProj = ProjectivePoint256(batch_final_points_x[0], batch_final_points_y[0], batch_final_points_z[0]);
        let finalAffine = to_affine_256(finalProj, PALLAS_CURVE.r2, PALLAS_CURVE.mont_inv32, PALLAS_CURVE.p, PALLAS_CURVE.p_minus_2);
        final_point_x = finalAffine.x;
        final_point_y = finalAffine.y;
    }
}
`;

export {
    pippengerShaderPassAProjectiveConversion,
    pippengerShaderPassBi1BucketScalarWeightedPointContribution,
    pippengerShaderPassBi2TreeReduceBucket,
    pippengerShaderPassCBucketAggregation,
    pippengerShaderPassDTreeReduceFinalPoint,
    pippengerShaderPassEFinalAccumulation
};

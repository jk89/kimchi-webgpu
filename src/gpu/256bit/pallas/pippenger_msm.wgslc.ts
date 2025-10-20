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
  - Initialize B[bucket_idx] = infinity (Z=0)
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
        P[idx] = to_projective_256(x[idx], y[idx], curve.r2, curve.mont_inv32, curve.p)

Input Buffers (global):

    - x — length N, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total N × 32 B

    - y — length N, array of Limbs256, each element = 8 × u32 limbs = 32 bytes → total N × 32 B

    - k — length N, array of Limbs256 (scalars mod r), each element = 8 × u32 limbs = 32 bytes → total N × 32 B

Output Buffers (global):

    - P — length N, array of ProjectivePoint256, each element = 3 × Limbs256 = 96 bytes → total N × 96 B

-----------------------------------------------------------------------------------------------------

Pass Bi_1:

Purpose: Decompose scalar contribution over all points/scalar pairs (N) for a particular bucket.

Method: 

    1. Zero out local (WGL) buffers:
       For each thread t:
         WGL[t].x.limbs[i] = 0u for all i
         WGL[t].y.limbs[i] = 0u for all i  
         WGL[t].z.limbs[i] = 0u for all i  (projective infinity)
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
           WGL[t] = P[idx]  (copy entire ProjectivePoint256)
       - Else:
           Leave WGL[t] as infinity (already zeroed in step 1)
       
       workgroupBarrier()
    
    3. Tree reduce WGL using binary halving:
       For stride = WORKGROUP_SIZE/2 down to 1:
         if t < stride:
           WGL[t] = point_add_proj_256(WGL[t], WGL[t + stride], curve.r2, curve.mont_inv32, curve.p)
         workgroupBarrier()
       After loop: WGL[0] contains the reduced workgroup point
    
    4. Thread 0 writes WGL[0] to WGG[workgroup_id]:
       if t == 0:
         WGG[workgroup_id] = WGL[0]  (copy entire ProjectivePoint256)

Input Buffers (global):

    - bucket_idx — uniform u32, updated per shader dispatch (1 to NUMBER_OF_BUCKETS)
    
    - BUCKET_WIDTH_BITS — uniform u32, number of bits per bucket (typically 8-16)
    
    - k — length N, array of Limbs256 (scalars mod r), each = 8 × u32 limbs = 32 bytes → total N × 32 B

    - P — length N, array of ProjectivePoint256, each = 3 × Limbs256 = 96 bytes → total N × 96 B

Workgroup buffers (local):

    - WGL — length WORKGROUP_SIZE, array of ProjectivePoint256, each = 3 × Limbs256 = 96 bytes → total WORKGROUP_SIZE × 96 B

Output Buffers (global):

    - WGG — length NUM_WORKGROUPS, array of ProjectivePoint256, each = 3 × Limbs256 = 96 bytes → total NUM_WORKGROUPS × 96 B

-----------------------------------------------------------------------------------------------------

Pass Bi_2:

Purpose: Tree reduce WGG global buffer to a single point and accumulate into B[bucket_idx].

Method:

    - Tree reduce global WGG buffers using binary halving:
      For stride = NUM_WORKGROUPS/2 down to 1:
        if workgroup_id < stride:
          WGG[workgroup_id] = point_add_proj_256(WGG[workgroup_id], WGG[workgroup_id + stride], curve.r2, curve.mont_inv32, curve.p)
        workgroupBarrier()
      After loop: WGG[0] contains the reduced point for all workgroups
    
    - Workgroup 0, thread 0 performs:
      if workgroup_id == 0 && thread_id == 0:
        B[bucket_idx] = point_add_proj_256(B[bucket_idx], WGG[0], curve.r2, curve.mont_inv32, curve.p)
      
      This accumulates contributions across multiple batches when N is too large to process at once.

Input Buffers (global):

    - WGG — length NUM_WORKGROUPS, array of ProjectivePoint256, each = 3 × Limbs256 = 96 bytes → total NUM_WORKGROUPS × 96 B

Output Buffers (global):

    - B — length NUMBER_OF_BUCKETS, array of ProjectivePoint256, each = 3 × Limbs256 = 96 bytes → total NUMBER_OF_BUCKETS × 96 B

Pass Bi_1 and Bi_2 continue until all scalar/point pairs and all buckets have been processed.

-----------------------------------------------------------------------------------------------------

Pass C:

Purpose: Aggregate all buckets using running sum technique, then convert to affine.

Method:
    
    running_sum = infinity (ProjectivePoint256 with z.limbs[i] = 0u for all i)
    result = infinity (ProjectivePoint256 with z.limbs[i] = 0u for all i)
    
    for bucket_idx from NUMBER_OF_BUCKETS-1 down to 1:
        running_sum = point_add_proj_256(running_sum, B[bucket_idx], curve.r2, curve.mont_inv32, curve.p)
        result = point_add_proj_256(result, running_sum, curve.r2, curve.mont_inv32, curve.p)
    
    final_point = to_affine_256(result, curve.r2, curve.mont_inv32, curve.p, curve.p_minus_2)
    
    Output final_point.x and final_point.y

Input Buffers (global):
    - B — length NUMBER_OF_BUCKETS, array of ProjectivePoint256

Output Buffers (global):
    - final_point — Point256 (affine coordinates)
      final_point.x — Limbs256, 8 × u32 limbs = 32 bytes
      final_point.y — Limbs256, 8 × u32 limbs = 32 bytes

*/
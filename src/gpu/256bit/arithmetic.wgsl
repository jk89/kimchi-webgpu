// Pallas curve MSM with Montgomery multiplication
// Curve equation: y² = x³ + 5

struct Limbs256 {
    limbs: array<u32, 8>
}

struct Point256 {
    x: Limbs256,
    y: Limbs256
}

struct ProjectivePoint256 {
    X: Limbs256,
    Y: Limbs256,
    Z: Limbs256
}

const PALLAS_P: array<u32, 8> = array<u32, 8>(
    0x00000001u, 0x992d30edu, 0x094cf91bu, 0x224698fcu,
    0x00000000u, 0x00000000u, 0x00000000u, 0x40000000u
);

// Montgomery constant: R = 2^256 mod p
const MONTGOMERY_R: array<u32, 8> = array<u32, 8>(
    0xfffffffe, 0x5588b13, 0x6730d2a0, 0xf4f63f58,
    0xffffffff, 0xffffffff, 0xffffffff, 0x0
);

// Montgomery constant: R^2 mod p (for converting to Montgomery form)
const MONTGOMERY_R2: array<u32, 8> = array<u32, 8>(
    0x8c46eb20, 0x748d9d99, 0x7523e5ce, 0x1a5f79f5,
    0xffd8ddee, 0x0, 0x0, 0x0
);

// p' = -p^(-1) mod 2^32 (used in Montgomery reduction)
const P_PRIME: u32 = 0xffffffffu;

const P_MINUS_2: array<u32, 8> = array<u32, 8>(
    0xFFFFFFFFu, 0x992d30ecu, 0x094cf91bu, 0x224698fcu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x3FFFFFFFu
);

fn gte_256(a: array<u32, 8>, b: array<u32, 8>) -> bool {
    var i: i32 = 7;
    loop {
        if (i < 0) { break; }
        if (a[i] > b[i]) { return true; }
        if (a[i] < b[i]) { return false; }
        i = i - 1;
    }
    return true;
}

fn sub_no_borrow_256(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var result: array<u32, 8>;
    var borrow: u32 = 0u;
    
    for (var i = 0u; i < 8u; i = i + 1u) {
        let ai = a[i];
        let bi = b[i];
        
        if (ai >= (bi + borrow)) {
            result[i] = ai - bi - borrow;
            borrow = 0u;
        } else {
            let temp = 0xFFFFFFFFu - bi - borrow + 1u;
            result[i] = temp + ai;
            borrow = 1u;
        }
    }
    
    return result;
}

fn add_mod_256(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var result: array<u32, 8>;
    var carry: u32 = 0u;
    
    for (var i = 0u; i < 8u; i = i + 1u) {
        let sum_low = a[i] + b[i];
        let carry_from_low = u32(sum_low < a[i]);
        
        let sum_with_carry = sum_low + carry;
        let carry_from_carry = u32(sum_with_carry < sum_low);
        
        result[i] = sum_with_carry;
        carry = carry_from_low + carry_from_carry;
    }
    
    if (carry != 0u || gte_256(result, PALLAS_P)) {
        result = sub_no_borrow_256(result, PALLAS_P);
    }
    
    return result;
}

fn sub_mod_256(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    if (gte_256(a, b)) {
        return sub_no_borrow_256(a, b);
    } else {
        let diff = sub_no_borrow_256(b, a);
        return sub_no_borrow_256(PALLAS_P, diff);
    }
}

fn mul_add_carry(a: u32, b: u32, acc: u32, carry: ptr<function, u32>) -> u32 {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    
    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;
    
    let mid = p1 + p2;
    let mid_carry = u32(mid < p1);
    
    let low = p0 + (mid << 16u);
    let low_carry = u32(low < p0);
    
    let high = p3 + (mid >> 16u) + mid_carry + low_carry;
    
    let result = low + acc;
    let result_carry = u32(result < low);
    
    let final_result = result + *carry;
    let final_carry = u32(final_result < result);
    
    *carry = high + result_carry + final_carry;
    return final_result;
}

// Montgomery reduction: compute (T * R^-1) mod p
// Input: 512-bit number T
// Output: 256-bit number T * R^-1 mod p
fn montgomery_reduce(t: array<u32, 16>) -> array<u32, 8> {
    var temp = t;
    
    // Montgomery reduction loop
    for (var i = 0u; i < 8u; i = i + 1u) {
        // m = temp[i] * P_PRIME mod 2^32
        let m = temp[i] * P_PRIME;
        
        // temp += m * p (this makes temp[i] = 0)
        var carry: u32 = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            temp[i + j] = mul_add_carry(m, PALLAS_P[j], temp[i + j], &carry);
        }
        
        // Propagate carry to high limbs
        var k = i + 8u;
        loop {
            if (k >= 16u || carry == 0u) { break; }
            let sum = temp[k] + carry;
            carry = u32(sum < temp[k]);
            temp[k] = sum;
            k = k + 1u;
        }
    }
    
    // Extract upper 256 bits (temp >> 256)
    var result: array<u32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        result[i] = temp[i + 8u];
    }
    
    // Final conditional subtraction
    if (gte_256(result, PALLAS_P)) {
        result = sub_no_borrow_256(result, PALLAS_P);
    }
    
    return result;
}

// Montgomery multiplication: (a * b * R^-1) mod p
fn mont_mul_256(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var product: array<u32, 16>;
    
    // Compute a * b
    for (var i = 0u; i < 8u; i = i + 1u) {
        var carry: u32 = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            product[i + j] = mul_add_carry(a[i], b[j], product[i + j], &carry);
        }
        product[i + 8u] = carry;
    }
    
    return montgomery_reduce(product);
}

// Convert to Montgomery form: a * R mod p
fn to_montgomery_256(a: array<u32, 8>) -> array<u32, 8> {
    return mont_mul_256(a, MONTGOMERY_R2);
}

// Convert from Montgomery form: a * R^-1 mod p
fn from_montgomery(a: array<u32, 8>) -> array<u32, 8> {
    var temp: array<u32, 16>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        temp[i] = a[i];
    }
    return montgomery_reduce(temp);
}

// Modular inverse in Montgomery form
fn mod_inverse_mont_256(a: array<u32, 8>) -> array<u32, 8> {
    var result: array<u32, 8>;
    result[0] = 1u;
    for (var i = 1u; i < 8u; i = i + 1u) {
        result[i] = 0u;
    }
    result = to_montgomery_256(result); // 1 in Montgomery form
    
    var base = a; // Already in Montgomery form
    
    for (var limb_idx = 0u; limb_idx < 8u; limb_idx = limb_idx + 1u) {
        var bits = P_MINUS_2[limb_idx];
        
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            if ((bits & 1u) == 1u) {
                result = mont_mul_256(result, base);
            }
            base = mont_mul_256(base, base);
            bits = bits >> 1u;
        }
    }
    
    return result;
}

fn is_infinity_proj_256(p: ProjectivePoint256) -> bool {
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (p.Z.limbs[i] != 0u) {
            return false;
        }
    }
    return true;
}

fn to_projective_256(point_x: Limbs256, point_y: Limbs256) -> ProjectivePoint256 {
    var result: ProjectivePoint256;
    // Convert to Montgomery form
    result.X.limbs = to_montgomery_256(point_x.limbs);
    result.Y.limbs = to_montgomery_256(point_y.limbs);
    result.Z.limbs[0] = 1u;
    for (var i = 1u; i < 8u; i = i + 1u) {
        result.Z.limbs[i] = 0u;
    }
    result.Z.limbs = to_montgomery_256(result.Z.limbs);
    return result;
}

fn to_affine_256(p: ProjectivePoint256) -> Point256 {
    if (is_infinity_proj_256(p)) {
        var inf: Point256;
        for (var i = 0u; i < 8u; i = i + 1u) {
            inf.x.limbs[i] = 0u;
            inf.y.limbs[i] = 0u;
        }
        return inf;
    }
    
    let z_inv = mod_inverse_mont_256(p.Z.limbs);
    
    var result: Point256;
    result.x.limbs = mont_mul_256(p.X.limbs, z_inv);
    result.y.limbs = mont_mul_256(p.Y.limbs, z_inv);
    
    // Convert back from Montgomery form
    result.x.limbs = from_montgomery(result.x.limbs);
    result.y.limbs = from_montgomery(result.y.limbs);
    
    return result;
}

fn point_double_proj_256(p: ProjectivePoint256) -> ProjectivePoint256 {
    if (is_infinity_proj_256(p)) { return p; }
    
    let XX = mont_mul_256(p.X.limbs, p.X.limbs);
    let YY = mont_mul_256(p.Y.limbs, p.Y.limbs);
    let YYYY = mont_mul_256(YY, YY);
    let ZZ = mont_mul_256(p.Z.limbs, p.Z.limbs);
    
    var S = add_mod_256(p.X.limbs, YY);
    S = mont_mul_256(S, S);
    S = sub_mod_256(S, XX);
    S = sub_mod_256(S, YYYY);
    S = add_mod_256(S, S);
    
    var M = add_mod_256(XX, XX);
    M = add_mod_256(M, XX);
    
    var T = mont_mul_256(M, M);
    T = sub_mod_256(T, add_mod_256(S, S));
    
    var result: ProjectivePoint256;
    result.X.limbs = T;
    
    var Y3 = sub_mod_256(S, T);
    Y3 = mont_mul_256(M, Y3);
    var YYYY8 = add_mod_256(YYYY, YYYY);
    YYYY8 = add_mod_256(YYYY8, YYYY8);
    YYYY8 = add_mod_256(YYYY8, YYYY8);
    result.Y.limbs = sub_mod_256(Y3, YYYY8);
    
    var Z3 = add_mod_256(p.Y.limbs, p.Z.limbs);
    Z3 = mont_mul_256(Z3, Z3);
    Z3 = sub_mod_256(Z3, YY);
    Z3 = sub_mod_256(Z3, ZZ);
    result.Z.limbs = Z3;
    
    return result;
}

fn point_add_proj_256(p: ProjectivePoint256, q: ProjectivePoint256) -> ProjectivePoint256 {
    if (is_infinity_proj_256(p)) { return q; }
    if (is_infinity_proj_256(q)) { return p; }
    
    let Z1Z1 = mont_mul_256(p.Z.limbs, p.Z.limbs);
    let Z2Z2 = mont_mul_256(q.Z.limbs, q.Z.limbs);
    
    let U1 = mont_mul_256(p.X.limbs, Z2Z2);
    let U2 = mont_mul_256(q.X.limbs, Z1Z1);
    
    let S1 = mont_mul_256(p.Y.limbs, mont_mul_256(q.Z.limbs, Z2Z2));
    let S2 = mont_mul_256(q.Y.limbs, mont_mul_256(p.Z.limbs, Z1Z1));
    
    var same_x = true;
    var same_y = true;
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (U1[i] != U2[i]) { same_x = false; }
        if (S1[i] != S2[i]) { same_y = false; }
    }
    if (same_x && same_y) {
        return point_double_proj_256(p);
    }
    
    let H = sub_mod_256(U2, U1);
    var I = add_mod_256(H, H);
    I = mont_mul_256(I, I);
    let J = mont_mul_256(H, I);
    
    var r = sub_mod_256(S2, S1);
    r = add_mod_256(r, r);
    
    let V = mont_mul_256(U1, I);
    
    var X3 = mont_mul_256(r, r);
    X3 = sub_mod_256(X3, J);
    X3 = sub_mod_256(X3, add_mod_256(V, V));
    
    var Y3 = sub_mod_256(V, X3);
    Y3 = mont_mul_256(r, Y3);
    let S1J = mont_mul_256(S1, J);
    let S1J2 = add_mod_256(S1J, S1J);
    Y3 = sub_mod_256(Y3, S1J2);
    
    var Z3 = add_mod_256(p.Z.limbs, q.Z.limbs);
    Z3 = mont_mul_256(Z3, Z3);
    Z3 = sub_mod_256(Z3, Z1Z1);
    Z3 = sub_mod_256(Z3, Z2Z2);
    Z3 = mont_mul_256(Z3, H);
    
    var result: ProjectivePoint256;
    result.X.limbs = X3;
    result.Y.limbs = Y3;
    result.Z.limbs = Z3;
    return result;
}

fn scalar_mul_256(scalar: Limbs256, point_x: Limbs256, point_y: Limbs256) -> Point256 {
    let p_proj = to_projective_256(point_x, point_y);
    
    var result: ProjectivePoint256;
    for (var i = 0u; i < 8u; i = i + 1u) {
        result.X.limbs[i] = 0u;
        result.Y.limbs[i] = 0u;
        result.Z.limbs[i] = 0u;
    }
    
    var base = p_proj;
    
    for (var limb_idx = 0u; limb_idx < 8u; limb_idx = limb_idx + 1u) {
        var bits = scalar.limbs[limb_idx];
        
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            if ((bits & 1u) == 1u) {
                result = point_add_proj_256(result, base);
            }
            base = point_double_proj_256(base);
            bits = bits >> 1u;
        }
    }
    
    return to_affine_256(result);
}

@group(0) @binding(0) var<storage, read> scalars: array<Limbs256>;
@group(0) @binding(1) var<storage, read> points_x: array<Limbs256>;
@group(0) @binding(2) var<storage, read> points_y: array<Limbs256>;
@group(0) @binding(3) var<storage, read_write> out_x: array<Limbs256>;
@group(0) @binding(4) var<storage, read_write> out_y: array<Limbs256>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    
    if (idx >= arrayLength(&scalars)) {
        return;
    }
    
    let out_point = scalar_mul_256(scalars[idx], points_x[idx], points_y[idx]);
    out_x[idx] = out_point.x;
    out_y[idx] = out_point.y;
}
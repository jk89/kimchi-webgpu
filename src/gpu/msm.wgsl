// Pallas curve MSM with Montgomery multiplication
// Curve equation: y² = x³ + 5

struct Limbs {
    limbs: array<u32, 8>
}

struct Point {
    x: Limbs,
    y: Limbs
}

struct ProjectivePoint {
    X: Limbs,
    Y: Limbs,
    Z: Limbs
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

fn gte(a: array<u32, 8>, b: array<u32, 8>) -> bool {
    var i: i32 = 7;
    loop {
        if (i < 0) { break; }
        if (a[i] > b[i]) { return true; }
        if (a[i] < b[i]) { return false; }
        i = i - 1;
    }
    return true;
}

fn sub_no_borrow(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
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

fn add_mod(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
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
    
    if (carry != 0u || gte(result, PALLAS_P)) {
        result = sub_no_borrow(result, PALLAS_P);
    }
    
    return result;
}

fn sub_mod(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    if (gte(a, b)) {
        return sub_no_borrow(a, b);
    } else {
        let diff = sub_no_borrow(b, a);
        return sub_no_borrow(PALLAS_P, diff);
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
    if (gte(result, PALLAS_P)) {
        result = sub_no_borrow(result, PALLAS_P);
    }
    
    return result;
}

// Montgomery multiplication: (a * b * R^-1) mod p
fn mont_mul(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
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
fn to_montgomery(a: array<u32, 8>) -> array<u32, 8> {
    return mont_mul(a, MONTGOMERY_R2);
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
fn mod_inverse_mont(a: array<u32, 8>) -> array<u32, 8> {
    var result: array<u32, 8>;
    result[0] = 1u;
    for (var i = 1u; i < 8u; i = i + 1u) {
        result[i] = 0u;
    }
    result = to_montgomery(result); // 1 in Montgomery form
    
    var base = a; // Already in Montgomery form
    
    for (var limb_idx = 0u; limb_idx < 8u; limb_idx = limb_idx + 1u) {
        var bits = P_MINUS_2[limb_idx];
        
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            if ((bits & 1u) == 1u) {
                result = mont_mul(result, base);
            }
            base = mont_mul(base, base);
            bits = bits >> 1u;
        }
    }
    
    return result;
}

fn is_infinity_proj(p: ProjectivePoint) -> bool {
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (p.Z.limbs[i] != 0u) {
            return false;
        }
    }
    return true;
}

fn to_projective(point_x: Limbs, point_y: Limbs) -> ProjectivePoint {
    var result: ProjectivePoint;
    // Convert to Montgomery form
    result.X.limbs = to_montgomery(point_x.limbs);
    result.Y.limbs = to_montgomery(point_y.limbs);
    result.Z.limbs[0] = 1u;
    for (var i = 1u; i < 8u; i = i + 1u) {
        result.Z.limbs[i] = 0u;
    }
    result.Z.limbs = to_montgomery(result.Z.limbs);
    return result;
}

fn to_affine(p: ProjectivePoint) -> Point {
    if (is_infinity_proj(p)) {
        var inf: Point;
        for (var i = 0u; i < 8u; i = i + 1u) {
            inf.x.limbs[i] = 0u;
            inf.y.limbs[i] = 0u;
        }
        return inf;
    }
    
    let z_inv = mod_inverse_mont(p.Z.limbs);
    
    var result: Point;
    result.x.limbs = mont_mul(p.X.limbs, z_inv);
    result.y.limbs = mont_mul(p.Y.limbs, z_inv);
    
    // Convert back from Montgomery form
    result.x.limbs = from_montgomery(result.x.limbs);
    result.y.limbs = from_montgomery(result.y.limbs);
    
    return result;
}

fn point_double_proj(p: ProjectivePoint) -> ProjectivePoint {
    if (is_infinity_proj(p)) { return p; }
    
    let XX = mont_mul(p.X.limbs, p.X.limbs);
    let YY = mont_mul(p.Y.limbs, p.Y.limbs);
    let YYYY = mont_mul(YY, YY);
    let ZZ = mont_mul(p.Z.limbs, p.Z.limbs);
    
    var S = add_mod(p.X.limbs, YY);
    S = mont_mul(S, S);
    S = sub_mod(S, XX);
    S = sub_mod(S, YYYY);
    S = add_mod(S, S);
    
    var M = add_mod(XX, XX);
    M = add_mod(M, XX);
    
    var T = mont_mul(M, M);
    T = sub_mod(T, add_mod(S, S));
    
    var result: ProjectivePoint;
    result.X.limbs = T;
    
    var Y3 = sub_mod(S, T);
    Y3 = mont_mul(M, Y3);
    var YYYY8 = add_mod(YYYY, YYYY);
    YYYY8 = add_mod(YYYY8, YYYY8);
    YYYY8 = add_mod(YYYY8, YYYY8);
    result.Y.limbs = sub_mod(Y3, YYYY8);
    
    var Z3 = add_mod(p.Y.limbs, p.Z.limbs);
    Z3 = mont_mul(Z3, Z3);
    Z3 = sub_mod(Z3, YY);
    Z3 = sub_mod(Z3, ZZ);
    result.Z.limbs = Z3;
    
    return result;
}

fn point_add_proj(p: ProjectivePoint, q: ProjectivePoint) -> ProjectivePoint {
    if (is_infinity_proj(p)) { return q; }
    if (is_infinity_proj(q)) { return p; }
    
    let Z1Z1 = mont_mul(p.Z.limbs, p.Z.limbs);
    let Z2Z2 = mont_mul(q.Z.limbs, q.Z.limbs);
    
    let U1 = mont_mul(p.X.limbs, Z2Z2);
    let U2 = mont_mul(q.X.limbs, Z1Z1);
    
    let S1 = mont_mul(p.Y.limbs, mont_mul(q.Z.limbs, Z2Z2));
    let S2 = mont_mul(q.Y.limbs, mont_mul(p.Z.limbs, Z1Z1));
    
    var same_x = true;
    var same_y = true;
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (U1[i] != U2[i]) { same_x = false; }
        if (S1[i] != S2[i]) { same_y = false; }
    }
    if (same_x && same_y) {
        return point_double_proj(p);
    }
    
    let H = sub_mod(U2, U1);
    var I = add_mod(H, H);
    I = mont_mul(I, I);
    let J = mont_mul(H, I);
    
    var r = sub_mod(S2, S1);
    r = add_mod(r, r);
    
    let V = mont_mul(U1, I);
    
    var X3 = mont_mul(r, r);
    X3 = sub_mod(X3, J);
    X3 = sub_mod(X3, add_mod(V, V));
    
    var Y3 = sub_mod(V, X3);
    Y3 = mont_mul(r, Y3);
    let S1J = mont_mul(S1, J);
    let S1J2 = add_mod(S1J, S1J);
    Y3 = sub_mod(Y3, S1J2);
    
    var Z3 = add_mod(p.Z.limbs, q.Z.limbs);
    Z3 = mont_mul(Z3, Z3);
    Z3 = sub_mod(Z3, Z1Z1);
    Z3 = sub_mod(Z3, Z2Z2);
    Z3 = mont_mul(Z3, H);
    
    var result: ProjectivePoint;
    result.X.limbs = X3;
    result.Y.limbs = Y3;
    result.Z.limbs = Z3;
    return result;
}

fn scalar_mul(scalar: Limbs, point_x: Limbs, point_y: Limbs) -> Point {
    let p_proj = to_projective(point_x, point_y);
    
    var result: ProjectivePoint;
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
                result = point_add_proj(result, base);
            }
            base = point_double_proj(base);
            bits = bits >> 1u;
        }
    }
    
    return to_affine(result);
}

@group(0) @binding(0) var<storage, read> scalars: array<Limbs>;
@group(0) @binding(1) var<storage, read> points_x: array<Limbs>;
@group(0) @binding(2) var<storage, read> points_y: array<Limbs>;
@group(0) @binding(3) var<storage, read_write> out_x: array<Limbs>;
@group(0) @binding(4) var<storage, read_write> out_y: array<Limbs>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    
    if (idx >= arrayLength(&scalars)) {
        return;
    }
    
    let out_point = scalar_mul(scalars[idx], points_x[idx], points_y[idx]);
    out_x[idx] = out_point.x;
    out_y[idx] = out_point.y;
}
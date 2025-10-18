// arithmetic.wgsl

// @import types.wgsl

// Compare two 256-bit integers
// Inputs: a, b : 256-bit integers
// Output: true if a >= b
// Path: compare most significant limb first, stop at first difference
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

// Subtract two 256-bit integers without underflow
// Inputs: a, b : 256-bit integers
// Output: result = a - b (wraparound prevented)
// Formula: result[i] = a[i] - b[i] - borrow_from_previous
// borrow = 1 if previous subtraction underflowed, 0 otherwise
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

// Modular addition: (a + b) mod p
// Inputs: a, b : 256-bit integers; p : modulus
// Output: (a + b) mod p
// Path: limb-wise addition with carry; subtract p if result >= p
fn add_mod_256(a: array<u32, 8>, b: array<u32, 8>, p: array<u32, 8>) -> array<u32, 8> {
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
    
    if (carry != 0u || gte_256(result, p)) {
        result = sub_no_borrow_256(result, p);
    }
    
    return result;
}

// Modular subtraction: (a - b) mod p
// Inputs: a, b : 256-bit integers; p : modulus
// Output: (a - b) mod p
// Path: if a >= b, result = a - b; else result = p - (b - a)
fn sub_mod_256(a: array<u32, 8>, b: array<u32, 8>, p: array<u32, 8>) -> array<u32, 8> {
    if (gte_256(a, b)) {
        return sub_no_borrow_256(a, b);
    } else {
        let diff = sub_no_borrow_256(b, a);
        return sub_no_borrow_256(p, diff);
    }
}

// Multiply two 32-bit integers and accumulate with carry
// Inputs: a, b : u32 integers; acc : current limb value; carry : propagated carry
// Output: updated limb = (a*b + acc + *carry) mod 2^32
// Path: split a, b into high/low 16-bit halves, compute cross-products, propagate carry
fn mul_add_carry(a: u32, b: u32, acc: u32, carry: ptr<function, u32>) -> u32 {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;

    // Combine middle terms
    let mid = p1 + p2;
    let mid_carry = u32(mid < p1);

    // Add lower 16 bits of mid to low part
    let low = p0 + (mid << 16u);
    let low_carry = u32(low < p0);

    // Compute high 32-bit including previous carries
    // mid_carry needs to be at bit position 32 (upper 16 bits of the high word)
    let high = p3 + (mid >> 16u) + (mid_carry << 16u) + low_carry;

    // Add acc
    let temp = low + acc;
    let temp_carry = u32(temp < low);

    // Add existing carry
    let final_res = temp + *carry;
    let final_carry = u32(final_res < temp);

    // Update carry
    *carry = high + temp_carry + final_carry;

    return final_res;
}

// Montgomery reduction: REDC(T) = T * R^-1 mod p
// Inputs:
//   t          : 512-bit integer (16 u32 limbs)
//   mont_inv32 : -p⁻¹ mod 2^32
//   p          : prime modulus
// Output: 256-bit integer = T * R^-1 mod p
// Path:
//   For each limb i of t:
//     m = t[i] * mont_inv32
//     temp += m * p -> t[i] becomes 0
//   Upper 256 bits = result
//   If result >= p, subtract p
fn montgomery_reduce_256(t: array<u32, 16>, mont_inv32: u32, p: array<u32, 8>) -> array<u32, 8> {
    var temp = t;
    
    // Montgomery reduction loop
    for (var i = 0u; i < 8u; i = i + 1u) {
        // m = temp[i] * mont_inv32 mod 2^32
        let m = temp[i] * mont_inv32;
        
        // temp += m * p (this makes temp[i] = 0)
        var carry: u32 = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            temp[i + j] = mul_add_carry(m, p[j], temp[i + j], &carry);
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
    if (gte_256(result, p)) {
        result = sub_no_borrow_256(result, p);
    }
    
    return result;
}

// Montgomery multiplication: (a * b * R^-1) mod p
// Inputs: a, b : 256-bit integers; mont_inv32, p : parameters
// Output: 256-bit integer = a * b * R^-1 mod p
// Path: compute 512-bit product, then Montgomery reduce
fn mont_mul_256(a: array<u32, 8>, b: array<u32, 8>, mont_inv32: u32, p: array<u32, 8>) -> array<u32, 8> {
    var product: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        product[i] = 0u;
    }
    
    // Compute a * b
    for (var i = 0u; i < 8u; i = i + 1u) {
        var carry: u32 = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            product[i + j] = mul_add_carry(a[i], b[j], product[i + j], &carry);
        }
        product[i + 8u] = carry;
    }
    
    return montgomery_reduce_256(product, mont_inv32, p);
}

// Convert a 256-bit integer to Montgomery form
// Inputs: a, r2, mont_inv32, p
// Output: a * R mod p
fn to_montgomery_256(a: array<u32, 8>, r2: array<u32, 8>, mont_inv32: u32, p: array<u32, 8>) -> array<u32, 8> {
    return mont_mul_256(a, r2, mont_inv32, p);
}

// Convert a 256-bit integer from Montgomery form
// Inputs: a : 256-bit integer in Montgomery form; mont_inv32, p : Montgomery parameters
// Output: a * R^-1 mod p
// Path: multiply by 1 using Montgomery multiplication
fn from_montgomery_256(a: array<u32, 8>, mont_inv32: u32, p: array<u32, 8>) -> array<u32, 8> {
    let one: array<u32, 8> = array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    return mont_mul_256(a, one, mont_inv32, p);
}

// Modular inverse in Montgomery form
// Inputs: a : 256-bit integer in Montgomery form
//         r2, mont_inv32, p : parameters
//         p_minus_2 : p-2 for Fermat's inverse
// Output: a^-1 mod p in Montgomery form
// Path: exponentiation by squaring: result = a^(p-2)
fn mod_inverse_mont_256(a: array<u32, 8>, r2: array<u32, 8>, mont_inv32: u32, p: array<u32, 8>, p_minus_2: array<u32, 8>) -> array<u32, 8> {
    var result: array<u32, 8>;
    result[0] = 1u;
    for (var i = 1u; i < 8u; i = i + 1u) {
        result[i] = 0u;
    }
    result = to_montgomery_256(result, r2, mont_inv32, p); // 1 in Montgomery form
    
    var base = a; // Already in Montgomery form
    
    for (var limb_idx = 0u; limb_idx < 8u; limb_idx = limb_idx + 1u) {
        var bits = p_minus_2[limb_idx];
        
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            if ((bits & 1u) == 1u) {
                result = mont_mul_256(result, base, mont_inv32, p);
            }
            base = mont_mul_256(base, base, mont_inv32, p);
            bits = bits >> 1u;
        }
    }
    
    return result;
}

// Check if projective point P is infinity (z == 0)
fn is_infinity_proj_256(P: ProjectivePoint256) -> bool {
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (P.z.limbs[i] != 0u) {
            return false;
        }
    }
    return true;
}

// Convert affine point (x, y) to projective coordinates (x:y:z)
// Inputs: x, y : Limbs256, r2, mont_inv32, p
// Output: ProjectivePoint256 P in Montgomery form, z = 1
fn to_projective_256(x: Limbs256, y: Limbs256, r2: array<u32, 8>, mont_inv32: u32, p: array<u32, 8>) -> ProjectivePoint256 {
    var P: ProjectivePoint256;
    // Convert to Montgomery form
    P.x.limbs = to_montgomery_256(x.limbs, r2, mont_inv32, p);
    P.y.limbs = to_montgomery_256(y.limbs, r2, mont_inv32, p);
    P.z.limbs[0] = 1u;
    for (var i = 1u; i < 8u; i = i + 1u) {
        P.z.limbs[i] = 0u;
    }
    P.z.limbs = to_montgomery_256(P.z.limbs, r2, mont_inv32, p);
    return P;
}

// Convert projective point to affine coordinates
// Inputs: P : ProjectivePoint256, r2, mont_inv32, p, p_minus_2
// Output: Q : Point256 (x, y) in affine coordinates
// Path:
//   If z == 0, return (0,0)
//   Compute z_inv = z^-1 mod p
//   x_affine = x * z_inv, y_affine = y * z_inv
//   Convert x_affine, y_affine from Montgomery form
fn to_affine_256(P: ProjectivePoint256, r2: array<u32, 8>, mont_inv32: u32, p: array<u32, 8>, p_minus_2: array<u32, 8>) -> Point256 {
    if (is_infinity_proj_256(P)) {
        var inf: Point256;
        for (var i = 0u; i < 8u; i = i + 1u) {
            inf.x.limbs[i] = 0u;
            inf.y.limbs[i] = 0u;
        }
        return inf;
    }
    
    let z_inv = mod_inverse_mont_256(P.z.limbs, r2, mont_inv32, p, p_minus_2);
    
    var Q: Point256;
    Q.x.limbs = mont_mul_256(P.x.limbs, z_inv, mont_inv32, p);
    Q.y.limbs = mont_mul_256(P.y.limbs, z_inv, mont_inv32, p);
    
    // Convert back from Montgomery form
    Q.x.limbs = from_montgomery_256(Q.x.limbs, mont_inv32, p);
    Q.y.limbs = from_montgomery_256(Q.y.limbs, mont_inv32, p);
    
    return Q;
}

// Point doubling in projective coordinates
// Inputs: P : ProjectivePoint256, r2, mont_inv32, p
// Output: Q = 2*P in projective coordinates
// Path:
//   xx = x^2, yy = y^2, yyyy = yy^2, zz = z^2
//   s  = 2*((x+yy)^2 - xx - yyyy)
//   m  = 3*xx
//   x3 = m^2 - 2*s
//   y3 = m*(s - x3) - 8*yyyy
//   z3 = (y+z)^2 - yy - zz
fn point_double_proj_256(P: ProjectivePoint256, r2: array<u32, 8>, mont_inv32: u32, p: array<u32, 8>) -> ProjectivePoint256 {
    if (is_infinity_proj_256(P)) { return P; }
    
    let xx = mont_mul_256(P.x.limbs, P.x.limbs, mont_inv32, p);
    let yy = mont_mul_256(P.y.limbs, P.y.limbs, mont_inv32, p);
    let yyyy = mont_mul_256(yy, yy, mont_inv32, p);
    let zz = mont_mul_256(P.z.limbs, P.z.limbs, mont_inv32, p);
    
    var s = add_mod_256(P.x.limbs, yy, p);
    s = mont_mul_256(s, s, mont_inv32, p);
    s = sub_mod_256(s, xx, p);
    s = sub_mod_256(s, yyyy, p);
    s = add_mod_256(s, s, p);
    
    var m = add_mod_256(xx, xx, p);
    m = add_mod_256(m, xx, p);
    
    var t = mont_mul_256(m, m, mont_inv32, p);
    t = sub_mod_256(t, add_mod_256(s, s, p), p);
    
    var Q: ProjectivePoint256;
    Q.x.limbs = t;
    
    var y3 = sub_mod_256(s, t, p);
    y3 = mont_mul_256(m, y3, mont_inv32, p);
    var yyyy8 = add_mod_256(yyyy, yyyy, p);
    yyyy8 = add_mod_256(yyyy8, yyyy8, p);
    yyyy8 = add_mod_256(yyyy8, yyyy8, p);
    Q.y.limbs = sub_mod_256(y3, yyyy8, p);
    
    var z3 = add_mod_256(P.y.limbs, P.z.limbs, p);
    z3 = mont_mul_256(z3, z3, mont_inv32, p);
    z3 = sub_mod_256(z3, yy, p);
    z3 = sub_mod_256(z3, zz, p);
    Q.z.limbs = z3;
    
    return Q;
}

// Point addition in projective coordinates
// Inputs: P, Q : ProjectivePoint256, r2, mont_inv32, p
// Output: R = P + Q in projective coordinates
// Path:
//   z1z1 = z1^2, z2z2 = z2^2
//   u1 = x1*z2z2, u2 = x2*z1z1
//   s1 = y1*z2*z2z2, s2 = y2*z1*z1z1
//   If u1 == u2 and s1 == s2, return 2*P
//   h = u2 - u1, i = (2*h)^2, j = h*i
//   r = 2*(s2 - s1), v = u1*i
//   x3 = r^2 - j - 2*v
//   y3 = r*(v - x3) - 2*s1*j
//   z3 = ((z1+z2)^2 - z1z1 - z2z2)*h
fn point_add_proj_256(P: ProjectivePoint256, Q: ProjectivePoint256, r2: array<u32, 8>, mont_inv32: u32, p: array<u32, 8>) -> ProjectivePoint256 {
    if (is_infinity_proj_256(P)) { return Q; }
    if (is_infinity_proj_256(Q)) { return P; }
    
    let z1z1 = mont_mul_256(P.z.limbs, P.z.limbs, mont_inv32, p);
    let z2z2 = mont_mul_256(Q.z.limbs, Q.z.limbs, mont_inv32, p);
    
    let u1 = mont_mul_256(P.x.limbs, z2z2, mont_inv32, p);
    let u2 = mont_mul_256(Q.x.limbs, z1z1, mont_inv32, p);
    
    let s1 = mont_mul_256(P.y.limbs, mont_mul_256(Q.z.limbs, z2z2, mont_inv32, p), mont_inv32, p);
    let s2 = mont_mul_256(Q.y.limbs, mont_mul_256(P.z.limbs, z1z1, mont_inv32, p), mont_inv32, p);
    
    var same_x = true;
    var same_y = true;
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (u1[i] != u2[i]) { same_x = false; }
        if (s1[i] != s2[i]) { same_y = false; }
    }
    if (same_x && same_y) {
        return point_double_proj_256(P, r2, mont_inv32, p);
    }
    
    let h = sub_mod_256(u2, u1, p);
    var i = add_mod_256(h, h, p);
    i = mont_mul_256(i, i, mont_inv32, p);
    let j = mont_mul_256(h, i, mont_inv32, p);
    
    var r = sub_mod_256(s2, s1, p);
    r = add_mod_256(r, r, p);
    
    let v = mont_mul_256(u1, i, mont_inv32, p);
    
    var x3 = mont_mul_256(r, r, mont_inv32, p);
    x3 = sub_mod_256(x3, j, p);
    x3 = sub_mod_256(x3, add_mod_256(v, v, p), p);
    
    var y3 = sub_mod_256(v, x3, p);
    y3 = mont_mul_256(r, y3, mont_inv32, p);
    let s1j = mont_mul_256(s1, j, mont_inv32, p);
    let s1j2 = add_mod_256(s1j, s1j, p);
    y3 = sub_mod_256(y3, s1j2, p);
    
    var z3 = add_mod_256(P.z.limbs, Q.z.limbs, p);
    z3 = mont_mul_256(z3, z3, mont_inv32, p);
    z3 = sub_mod_256(z3, z1z1, p);
    z3 = sub_mod_256(z3, z2z2, p);
    z3 = mont_mul_256(z3, h, mont_inv32, p);
    
    var R: ProjectivePoint256;
    R.x.limbs = x3;
    R.y.limbs = y3;
    R.z.limbs = z3;
    return R;
}

// Scalar multiplication: k*P
// Inputs: k : scalar (Limbs256), P : point (x, y coordinates), r2, mont_inv32, p, p_minus_2
// Output: Q = k*P in affine coordinates
// Path: double-and-add algorithm, process scalar bits from LSB to MSB
fn scalar_mul_256(k: Limbs256, P_x: Limbs256, P_y: Limbs256, r2: array<u32, 8>, mont_inv32: u32, p: array<u32, 8>, p_minus_2: array<u32, 8>) -> Point256 {
    let P = to_projective_256(P_x, P_y, r2, mont_inv32, p);
    
    var Q: ProjectivePoint256;
    for (var i = 0u; i < 8u; i = i + 1u) {
        Q.x.limbs[i] = 0u;
        Q.y.limbs[i] = 0u;
        Q.z.limbs[i] = 0u;
    }
    
    var base = P;
    
    for (var limb_idx = 0u; limb_idx < 8u; limb_idx = limb_idx + 1u) {
        var bits = k.limbs[limb_idx];
        
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            if ((bits & 1u) == 1u) {
                Q = point_add_proj_256(Q, base, r2, mont_inv32, p);
            }
            base = point_double_proj_256(base, r2, mont_inv32, p);
            bits = bits >> 1u;
        }
    }
    
    return to_affine_256(Q, r2, mont_inv32, p, p_minus_2);
}

// Scalar multiplication for arbitrary curve: k*P
// Inputs: k : scalar (Limbs256), P : point (x, y coordinates), curve : Curve256 parameters
// Output: Q = k*P in affine coordinates
// Path: uses curve parameters with generic scalar_mul_256
fn curve_scalar_mul(k: Limbs256, P_x: Limbs256, P_y: Limbs256, curve: Curve256) -> Point256 {
    return scalar_mul_256(
        k, P_x, P_y,
        curve.r2,
        curve.mont_inv32,
        curve.p,
        curve.p_minus_2
    );
}
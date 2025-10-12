// Pallas curve Multi-Scalar Multiplication (MSM)
// Curve equation: y² = x³ + 5
// Field modulus p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001

struct Limbs {
    limbs: array<u32, 8>
}

struct Point {
    x: Limbs,
    y: Limbs
}

@group(0) @binding(0) var<storage, read> scalars: array<Limbs>;
@group(0) @binding(1) var<storage, read> points: array<Point>;
@group(0) @binding(2) var<storage, read_write> out: array<Point>;

// Pallas base field modulus p (little-endian u32 limbs)
const PALLAS_P: array<u32, 8> = array<u32, 8>(
    0x00000001u, 0x992d30edu, 0x094cf91bu, 0x224698fcu,
    0x00000000u, 0x00000000u, 0x00000000u, 0x40000000u
);

// p - 2 for Fermat's little theorem (modular inverse)
const P_MINUS_2: array<u32, 8> = array<u32, 8>(
    0xFFFFFFFFu, 0x992d30ecu, 0x094cf91bu, 0x224698fcu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x3FFFFFFFu
);

// Compare two 256-bit numbers: returns true if a >= b
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

// Check if two 256-bit numbers are equal
fn eq(a: array<u32, 8>, b: array<u32, 8>) -> bool {
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (a[i] != b[i]) { return false; }
    }
    return true;
}

// Subtract b from a (assumes a >= b)
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

// Add two 256-bit numbers modulo p
fn add_mod(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var result: array<u32, 8>;
    var carry: u32 = 0u;
    
    for (var i = 0u; i < 8u; i = i + 1u) {
        let ai = a[i];
        let bi = b[i];
        
        let sum_low = ai + bi;
        let carry_from_low = u32(sum_low < ai);
        
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

// Subtract b from a modulo p
fn sub_mod(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    if (gte(a, b)) {
        return sub_no_borrow(a, b);
    } else {
        let diff = sub_no_borrow(b, a);
        return sub_no_borrow(PALLAS_P, diff);
    }
}

// Multiply a * b where b is single u32, add to accumulator with carry
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

// Multiply two 256-bit numbers modulo p
fn mul_mod(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var result: array<u32, 16>;
    
    // Schoolbook multiplication
    for (var i = 0u; i < 8u; i = i + 1u) {
        var carry: u32 = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            result[i + j] = mul_add_carry(a[i], b[j], result[i + j], &carry);
        }
        result[i + 8u] = carry;
    }
    
    // Reduce modulo p using repeated subtraction
    // Extract high and low parts
    var low: array<u32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        low[i] = result[i];
    }
    
    // Simple reduction (not efficient but correct)
    for (var iter = 0u; iter < 512u; iter = iter + 1u) {
        if (gte(low, PALLAS_P)) {
            low = sub_no_borrow(low, PALLAS_P);
        } else {
            break;
        }
    }
    
    return low;
}

// Modular inverse using Fermat's Little Theorem: a^(-1) = a^(p-2) mod p
fn mod_inverse(a: array<u32, 8>) -> array<u32, 8> {
    var result: array<u32, 8>;
    result[0] = 1u;
    for (var i = 1u; i < 8u; i = i + 1u) {
        result[i] = 0u;
    }
    
    var base = a;
    
    // Square-and-multiply algorithm
    for (var limb_idx = 0u; limb_idx < 8u; limb_idx = limb_idx + 1u) {
        var bits = P_MINUS_2[limb_idx];
        
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            if ((bits & 1u) == 1u) {
                result = mul_mod(result, base);
            }
            base = mul_mod(base, base);
            bits = bits >> 1u;
        }
    }
    
    return result;
}

// Check if point is at infinity (all zeros)
fn is_infinity(p: Point) -> bool {
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (p.x.limbs[i] != 0u || p.y.limbs[i] != 0u) {
            return false;
        }
    }
    return true;
}

// Check if two points are equal
fn points_equal(p: Point, q: Point) -> bool {
    return eq(p.x.limbs, q.x.limbs) && eq(p.y.limbs, q.y.limbs);
}

// Point doubling: 2*P for curve y² = x³ + 5
fn point_double(p: Point) -> Point {
    if (is_infinity(p)) { return p; }
    
    // λ = (3x²) / (2y)
    let x_squared = mul_mod(p.x.limbs, p.x.limbs);
    let three_x_squared = add_mod(add_mod(x_squared, x_squared), x_squared);
    
    let two_y = add_mod(p.y.limbs, p.y.limbs);
    let two_y_inv = mod_inverse(two_y);
    
    let lambda = mul_mod(three_x_squared, two_y_inv);
    
    // x' = λ² - 2x
    let lambda_squared = mul_mod(lambda, lambda);
    let two_x = add_mod(p.x.limbs, p.x.limbs);
    let x_new = sub_mod(lambda_squared, two_x);
    
    // y' = λ(x - x') - y
    let x_diff = sub_mod(p.x.limbs, x_new);
    let lambda_x_diff = mul_mod(lambda, x_diff);
    let y_new = sub_mod(lambda_x_diff, p.y.limbs);
    
    var result: Point;
    result.x.limbs = x_new;
    result.y.limbs = y_new;
    return result;
}

// Point addition: P + Q for curve y² = x³ + 5
fn point_add(p: Point, q: Point) -> Point {
    if (is_infinity(p)) { return q; }
    if (is_infinity(q)) { return p; }
    
    // Check if points are equal (use doubling instead)
    if (points_equal(p, q)) {
        return point_double(p);
    }
    
    // λ = (q.y - p.y) / (q.x - p.x)
    let dy = sub_mod(q.y.limbs, p.y.limbs);
    let dx = sub_mod(q.x.limbs, p.x.limbs);
    let dx_inv = mod_inverse(dx);
    let lambda = mul_mod(dy, dx_inv);
    
    // x' = λ² - p.x - q.x
    let lambda_squared = mul_mod(lambda, lambda);
    let x_sum = add_mod(p.x.limbs, q.x.limbs);
    let x_new = sub_mod(lambda_squared, x_sum);
    
    // y' = λ(p.x - x') - p.y
    let x_diff = sub_mod(p.x.limbs, x_new);
    let lambda_x_diff = mul_mod(lambda, x_diff);
    let y_new = sub_mod(lambda_x_diff, p.y.limbs);
    
    var result: Point;
    result.x.limbs = x_new;
    result.y.limbs = y_new;
    return result;
}

// Scalar multiplication using double-and-add
fn scalar_mul(scalar: Limbs, point: Point) -> Point {
    var result: Point;
    // Initialize to point at infinity
    for (var i = 0u; i < 8u; i = i + 1u) {
        result.x.limbs[i] = 0u;
        result.y.limbs[i] = 0u;
    }
    
    var temp = point;
    
    // Double-and-add algorithm (process bits from LSB to MSB)
    for (var limb_idx = 0u; limb_idx < 8u; limb_idx = limb_idx + 1u) {
        var bits = scalar.limbs[limb_idx];
        
        for (var bit = 0u; bit < 32u; bit = bit + 1u) {
            if ((bits & 1u) == 1u) {
                result = point_add(result, temp);
            }
            temp = point_double(temp);
            bits = bits >> 1u;
        }
    }
    
    return result;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    
    if (idx >= arrayLength(&scalars)) {
        return;
    }
    
    // Compute scalar multiplication: out[idx] = scalars[idx] * points[idx]
    out[idx] = scalar_mul(scalars[idx], points[idx]);
}
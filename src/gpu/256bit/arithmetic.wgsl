// src/gpu/256bit/arithmetic.wgsl

struct limbs256 {
    limbs: array<u32, 8>
}

// Compare a >= b
fn gte256(a: limbs256, b: limbs256) -> bool {
    var i: i32 = 7;
    loop {
        if (i < 0) { break; }
        if (a.limbs[i] > b.limbs[i]) { return true; }
        if (a.limbs[i] < b.limbs[i]) { return false; }
        i = i - 1;
    }
    return true;
}

// Subtraction without borrow
fn sub_no_borrow256(a: limbs256, b: limbs256) -> limbs256 {
    var result: limbs256;
    var borrow: u32 = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let ai = a.limbs[i];
        let bi = b.limbs[i];
        if (ai >= (bi + borrow)) {
            result.limbs[i] = ai - bi - borrow;
            borrow = 0u;
        } else {
            let temp = 0xFFFFFFFFu - bi - borrow + 1u;
            result.limbs[i] = temp + ai;
            borrow = 1u;
        }
    }
    return result;
}

// Modular addition: (a + b) mod p
fn add_mod256(a: limbs256, b: limbs256) -> limbs256 {
    var result: limbs256;
    var carry: u32 = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let sum_low = a.limbs[i] + b.limbs[i];
        let carry_from_low = u32(sum_low < a.limbs[i]);
        let sum_with_carry = sum_low + carry;
        let carry_from_carry = u32(sum_with_carry < sum_low);
        result.limbs[i] = sum_with_carry;
        carry = carry_from_low + carry_from_carry;
    }
    if (carry != 0u || gte256(result, PALLAS_P)) {
        result = sub_no_borrow256(result, PALLAS_P);
    }
    return result;
}

// Modular subtraction: (a - b) mod p
fn sub_mod256(a: limbs256, b: limbs256) -> limbs256 {
    if (gte256(a, b)) {
        return sub_no_borrow256(a, b);
    } else {
        let diff = sub_no_borrow256(b, a);
        return sub_no_borrow256(PALLAS_P, diff);
    }
}

// 32-bit multiply with carry
fn mul_add_carry256(a: u32, b: u32, acc: u32, carry: ptr<function, u32>) -> u32 {
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

// Montgomery reduction: (T * R^-1) mod p
fn montgomery_reduce256(t: array<u32, 16>) -> limbs256 {
    var temp = t;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let m = temp[i] * P_PRIME;
        var carry: u32 = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            temp[i + j] = mul_add_carry256(m, PALLAS_P.limbs[j], temp[i + j], &carry);
        }
        var k = i + 8u;
        loop {
            if (k >= 16u || carry == 0u) { break; }
            let sum = temp[k] + carry;
            carry = u32(sum < temp[k]);
            temp[k] = sum;
            k = k + 1u;
        }
    }

    var result: limbs256;
    for (var i = 0u; i < 8u; i = i + 1u) {
        result.limbs[i] = temp[i + 8u];
    }

    if (gte256(result, PALLAS_P)) {
        result = sub_no_borrow256(result, PALLAS_P);
    }
    return result;
}

// Montgomery multiplication: (a * b * R^-1) mod p
fn mont_mul256(a: limbs256, b: limbs256) -> limbs256 {
    var product: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { product[i] = 0u; }

    for (var i = 0u; i < 8u; i = i + 1u) {
        var carry: u32 = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            product[i + j] = mul_add_carry256(a.limbs[i], b.limbs[j], product[i + j], &carry);
        }
        product[i + 8u] = carry;
    }

    return montgomery_reduce256(product);
}

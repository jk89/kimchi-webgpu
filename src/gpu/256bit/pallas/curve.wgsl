// curve.wgsl

// @import types.wgsl

// Pallas curve equation: y² = x³ + 5

// Pallas prime modulus (field size)  
const PALLAS_P: array<u32, 8> = array<u32, 8>(
    0x00000001u, 0x992d30edu, 0x094cf91bu, 0x224698fcu,
    0x00000000u, 0x00000000u, 0x00000000u, 0x40000000u
);

// Pallas R² mod p, used to convert numbers into Montgomery form for fast arithmetic  
const PALLAS_R2: array<u32, 8> = array<u32, 8>(
    0x8c46eb20, 0x748d9d99, 0x7523e5ce, 0x1a5f79f5,
    0xffd8ddee, 0x0, 0x0, 0x0
);

// Pallas -p⁻¹ mod 2^32, required for Montgomery reduction in field operations  
const PALLAS_MONT_INV32: u32 = 0xffffffffu;

// Pallas curve coefficient 'a' in y² = x³ + a*x + b (here a = 0 for Pallas)  
const PALLAS_A: array<u32, 8> = array<u32, 8>(
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
);

// Pallas curve coefficient 'b' in y² = x³ + a*x + b (here b = 5 for Pallas)  
const PALLAS_B: array<u32, 8> = array<u32, 8>(
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 5u
);

// Pallas p - 2, used for modular inverse computation
const PALLAS_P_MINUS_2: array<u32, 8> = array<u32, 8>(
    0xFFFFFFFFu, 0x992d30ecu, 0x094cf91bu, 0x224698fcu,
    0x00000000u, 0x00000000u, 0x00000000u, 0x40000000u
);

// Complete Pallas curve parameters as a Curve256 instance  
const PALLAS_CURVE: Curve256 = Curve256(PALLAS_P, PALLAS_R2, PALLAS_MONT_INV32, PALLAS_A, PALLAS_B, PALLAS_P_MINUS_2);
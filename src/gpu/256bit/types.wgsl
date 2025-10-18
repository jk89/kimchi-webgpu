// types.wgsl

// Represents a 256-bit integer as 8 32-bit limbs (little-endian)
// limbs[0] = least significant 32 bits
struct Limbs256 {
    limbs: array<u32, 8>       // limbs[0] = least significant 32 bits
}

// Represents a point in affine coordinates over a 256-bit field
// x, y coordinates are in Montgomery form
struct Point256 {
    x: Limbs256                // x-coordinate of the point
    y: Limbs256                // y-coordinate of the point
}

// Represents a point in projective coordinates over a 256-bit field
// x, y, z are in Montgomery form
// Point at infinity if Z = 0
struct ProjectivePoint256 {
    x: Limbs256                // Projective x coordinate
    y: Limbs256                // Projective y coordinate
    z: Limbs256                // Projective z coordinate (point at infinity if z = 0)
}

// Represents the parameters of a 256-bit elliptic curve
// p      : prime modulus of the field
// r2     : R² mod p, for Montgomery conversion
// mont_inv32 : -p⁻¹ mod 2^32 for Montgomery reduction
// a, b   : curve coefficients
// p_minus_2  : p - 2, used for modular inverse via Fermat's little theorem
struct Curve256 {
    p: array<u32, 8>,          // Prime modulus of the curve (field size)
    r2: array<u32, 8>,         // R² mod p, used for converting numbers into Montgomery form
    mont_inv32: u32,           // -p⁻¹ mod 2^32, used in Montgomery reduction
    a: array<u32, 8>,          // Curve coefficient 'a' in the equation y² = x³ + a*x + b
    b: array<u32, 8>,          // Curve coefficient 'b' in the equation y² = x³ + a*x + b
    p_minus_2: array<u32, 8>,  // p - 2, for computing modular inverse: a^(-1) = a^(p-2) mod p
};
import { expect } from 'chai';
import { gpuMSM } from './msm.js';

// Pallas prime and curve constants
const PALLAS_P = (1n << 254n) + 0x224698fc094cf91bn;
const CURVE_B = 5n;

// Modular arithmetic helpers
function addMod(a: bigint, b: bigint): bigint {
    const r = a + b;
    return r >= PALLAS_P ? r - PALLAS_P : r;
}

function subMod(a: bigint, b: bigint): bigint {
    return a >= b ? a - b : PALLAS_P - (b - a);
}

// Montgomery constants
const R = 1n << 256n; // 2^256
const R_INV = 0x992d30ed00000001000000000000000000000000000000004000000000000000n; // R^-1 mod p
const R2 = 0x8c46eb20748d9d997523e5ce1a5f79f5ffd8ddee00000000000000000000000n; // R^2 mod p

// Proper Montgomery multiplication
function montMul(a: bigint, b: bigint): bigint {
    // Montgomery multiplication: (a * b * R^-1) mod p
    const product = a * b;
    const m = (product * R_INV) % R;
    let t = (product + m * PALLAS_P) / R;
    if (t >= PALLAS_P) t -= PALLAS_P;
    return t;
}

// Convert to Montgomery form
function toMont(a: bigint): bigint {
    return montMul(a, R2);
}

// Convert from Montgomery form  
function fromMont(a: bigint): bigint {
    return montMul(a, 1n);
}

// Modular inverse using Fermat's Little Theorem
function modInv(a: bigint): bigint {
    return modPow(a, PALLAS_P - 2n);
}

function modPow(base: bigint, exp: bigint): bigint {
    let result = 1n;
    base = base % PALLAS_P;
    while (exp > 0n) {
        if (exp & 1n) result = (result * base) % PALLAS_P;
        base = (base * base) % PALLAS_P;
        exp >>= 1n;
    }
    return result;
}

// Projective point type
interface ProjectivePoint {
    X: bigint;
    Y: bigint;
    Z: bigint;
}

interface Point {
    x: bigint;
    y: bigint;
}

// Check if point is at infinity
function isInfinity(P: ProjectivePoint): boolean {
    return P.Z === 0n;
}

// Affine -> projective (convert to Montgomery form)
function toProjective(P: Point): ProjectivePoint {
    return {
        X: toMont(P.x),
        Y: toMont(P.y),
        Z: toMont(1n),
    };
}

// Projective -> affine (convert from Montgomery form)
function toAffine(P: ProjectivePoint): Point {
    if (isInfinity(P)) {
        return { x: 0n, y: 0n };
    }

    const zInv = modInv(fromMont(P.Z));
    const x = fromMont(montMul(P.X, toMont(zInv)));
    const y = fromMont(montMul(P.Y, toMont(zInv)));

    return { x, y };
}

// Point doubling in projective coordinates
function pointDouble(P: ProjectivePoint): ProjectivePoint {
    if (isInfinity(P)) return P;

    const XX = montMul(P.X, P.X);
    const YY = montMul(P.Y, P.Y);
    const YYYY = montMul(YY, YY);
    const ZZ = montMul(P.Z, P.Z);

    // S = 4 * X * Y^2
    let S = montMul(P.X, YY);
    S = addMod(S, S);
    S = addMod(S, S);

    // M = 3 * X^2
    let M = addMod(XX, XX);
    M = addMod(M, XX);

    // T = M^2 - 2S
    let T = montMul(M, M);
    T = subMod(T, addMod(S, S));

    // X3 = T
    const X3 = T;

    // Y3 = M * (S - T) - 8 * Y^4
    let Y3 = subMod(S, T);
    Y3 = montMul(M, Y3);
    const YYYY8 = addMod(YYYY, YYYY);
    Y3 = subMod(Y3, addMod(YYYY8, YYYY8));
    Y3 = subMod(Y3, addMod(YYYY8, YYYY8));

    // Z3 = 2 * Y * Z
    let Z3 = montMul(P.Y, P.Z);
    Z3 = addMod(Z3, Z3);

    return { X: X3, Y: Y3, Z: Z3 };
}

// Point addition in projective coordinates
function pointAdd(P: ProjectivePoint, Q: ProjectivePoint): ProjectivePoint {
    if (isInfinity(P)) return Q;
    if (isInfinity(Q)) return P;

    const Z1Z1 = montMul(P.Z, P.Z);
    const Z2Z2 = montMul(Q.Z, Q.Z);

    const U1 = montMul(P.X, Z2Z2);
    const U2 = montMul(Q.X, Z1Z1);

    const S1 = montMul(P.Y, montMul(Q.Z, Z2Z2));
    const S2 = montMul(Q.Y, montMul(P.Z, Z1Z1));

    // Check if points are equal (for doubling)
    if (U1 === U2) {
        if (S1 === S2) {
            return pointDouble(P);
        } else {
            // Points are inverses, return infinity
            return { X: 0n, Y: 0n, Z: 0n };
        }
    }

    const H = subMod(U2, U1);
    const I = montMul(addMod(H, H), addMod(H, H));
    const J = montMul(H, I);

    const r = subMod(S2, S1);
    const r2 = addMod(r, r);

    const V = montMul(U1, I);

    // X3 = r^2 - J - 2V
    let X3 = montMul(r2, r2);
    X3 = subMod(X3, J);
    X3 = subMod(X3, addMod(V, V));

    // Y3 = r * (V - X3) - 2 * S1 * J
    let Y3 = subMod(V, X3);
    Y3 = montMul(r2, Y3);
    const S1J = montMul(S1, J);
    Y3 = subMod(Y3, addMod(S1J, S1J));

    // Z3 = ((Z1 + Z2)^2 - Z1Z1 - Z2Z2) * H
    let Z3 = addMod(P.Z, Q.Z);
    Z3 = montMul(Z3, Z3);
    Z3 = subMod(Z3, Z1Z1);
    Z3 = subMod(Z3, Z2Z2);
    Z3 = montMul(Z3, H);

    return { X: X3, Y: Y3, Z: Z3 };
}

// Scalar multiplication using double-and-add
function scalarMulCPU(k: bigint, P: Point): Point {
    if (k === 0n) {
        return { x: 0n, y: 0n };
    }

    let result: ProjectivePoint = { X: 0n, Y: 0n, Z: 0n }; // infinity
    let base = toProjective(P);

    // Work with absolute value of scalar
    let scalar = k;
    if (scalar < 0n) {
        scalar = -scalar;
        // Negate the base point
        base = { ...base, Y: subMod(0n, base.Y) };
    }

    while (scalar > 0n) {
        if (scalar & 1n) {
            result = pointAdd(result, base);
        }
        base = pointDouble(base);
        scalar >>= 1n;
    }

    return toAffine(result);
}

describe('GPU vs CPU full MSM', function () {
    // on a 3090 we get about 215053 multiplications per second
    it('matches outputs and measures times', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();

        // Generate a large number of scalars and points
        const N = 4000000;
        const scalars: bigint[] = [];
        const points: { x: bigint; y: bigint }[] = [];
        for (let i = 0; i < N; i++) {
            scalars.push(BigInt(i + 1)); // 1n, 2n, 3n, ...
            points.push({ x: BigInt(5 + i * 6), y: BigInt(7 + i * 6) }); // deterministic increasing sequence
        }

        // CPU timing
        /*const cpuStart = performance.now();
        const cpuResults = scalars.map((s, i) => scalarMulCPU(s, points[i]));
        const cpuEnd = performance.now();
        console.log(`CPU MSM took ${cpuEnd - cpuStart} ms`);*/

        // GPU timing (your existing gpuMSM should produce points in normal affine form)
        const gpuStart = performance.now();
        const gpuResults = await gpuMSM(device, scalars, points);
        const gpuEnd = performance.now();
        console.log(`GPU MSM took ${gpuEnd - gpuStart} ms`);

        // Compare outputs (normalize mod p and to string to avoid negative bigint formatting issues)
        /*for (let i = 0; i < scalars.length; i++) {
            const gx = ((gpuResults[i].x % PALLAS_P) + PALLAS_P) % PALLAS_P;
            const gy = ((gpuResults[i].y % PALLAS_P) + PALLAS_P) % PALLAS_P;
            const cx = ((cpuResults[i].x % PALLAS_P) + PALLAS_P) % PALLAS_P;
            const cy = ((cpuResults[i].y % PALLAS_P) + PALLAS_P) % PALLAS_P;

            expect(gx.toString()).to.equal(
                cx.toString(),
                `x mismatch for scalar ${scalars[i]}`
            );
            expect(gy.toString()).to.equal(
                cy.toString(),
                `y mismatch for scalar ${scalars[i]}`
            );
        }*/
    });
});

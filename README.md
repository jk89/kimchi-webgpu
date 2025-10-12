# Abstract

**Kimchi WebGPU** explores accelerating the Kimchi zero-knowledge prover in browsers by performing **Number-Theoretic Transforms (NTTs)** [NTT](https://en.wikipedia.org/wiki/Number-theoretic_transform) and **Multi-Scalar Multiplications (MSMs)** [MSM](https://en.wikipedia.org/wiki/Scalar_multiplication#Multiple_scalar_multiplication) over the Pallas and Vesta fields entirely on the GPU via WebGPU. The approach keeps all polynomial coefficients and intermediate results resident on-device, minimizing host-GPU synchronization, with the goal of improving prover throughput while preserving correctness and determinism.

# Requirements

- Brave Nightly installed (macOS or Linux) to run browser-based WebGPU tests and benchmarks.  
  - Ensure flags like `--enable-unsafe-webgpu` and `--ignore-gpu-blocklist` are enabled for full GPU support.e

# Installation

1. Install dependencies:  
   `npm install`  

# Build  

- Compile TypeScript source files into the `dist` folder:  
  `npm run build`  
  This runs the command: `tsc`  

## Test / Run Browser WebGPU Tests

- **Headless CLI Tests** (for CI or automated runs):  
  `npm run test:browser-cli`  
  This runs the compiled browser test runner in Node using Puppeteer + Brave in headless mode.

- **Interactive Browser Tests** (leave browser open for inspection):  
  `npm run test:browser`  
  This starts an Express server, opens Brave with WebGPU enabled, and displays test results in the browser page.  
  The page URL will be printed to stdout so you can open it manually if needed.

import { expect } from 'chai';

describe('WebGPU GPU unit tests', () => {
    it('navigator.gpu should exist', () => {
        expect(navigator.gpu).to.exist;
    });

    it('can request a GPU device', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        expect(adapter).to.exist;

        const device = await adapter?.requestDevice();
        expect(device).to.exist;
    });

    it('simple GPU buffer test', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter?.requestDevice();

        const buffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE,
        });

        await buffer.mapAsync(GPUMapMode.WRITE);
        const array = new Uint32Array(buffer.getMappedRange());
        array[0] = 42;
        buffer.unmap();

        const readback = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(buffer, 0, readback, 0, 4);
        device.queue.submit([encoder.finish()]);

        await readback.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(readback.getMappedRange())[0];
        expect(result).to.equal(42);
        readback.unmap();
    });

    it('print device limits info', async () => {
        const adapter = await navigator.gpu.requestAdapter();
        expect(adapter).to.exist;

        const device = await adapter?.requestDevice();
        expect(device).to.exist;

        const limitKeys = [
            'maxBindGroups',
            'maxBindGroupsPlusVertexBuffers',
            'maxBindingsPerBindGroup',
            'maxBufferSize',
            'maxColorAttachmentBytesPerSample',
            'maxColorAttachments',
            'maxComputeInvocationsPerWorkgroup',
            'maxComputeWorkgroupSizeX',
            'maxComputeWorkgroupSizeY',
            'maxComputeWorkgroupSizeZ',
            'maxComputeWorkgroupStorageSize',
            'maxComputeWorkgroupsPerDimension',
            'maxDynamicStorageBuffersPerPipelineLayout',
            'maxDynamicUniformBuffersPerPipelineLayout',
            'maxInterStageShaderComponents',
            'maxInterStageShaderVariables',
            'maxSampledTexturesPerShaderStage',
            'maxSamplersPerShaderStage',
            'maxStorageBufferBindingSize',
            'maxStorageBuffersPerShaderStage',
            'maxStorageTexturesPerShaderStage',
            'maxSubgroupSize',
            'maxTextureArrayLayers',
            'maxTextureDimension1D',
            'maxTextureDimension2D',
            'maxTextureDimension3D',
            'maxUniformBufferBindingSize',
            'maxUniformBuffersPerShaderStage',
            'maxVertexAttributes',
            'maxVertexBufferArrayStride',
            'maxVertexBuffers',
            'minStorageBufferOffsetAlignment',
            'minSubgroupSize',
            'minUniformBufferOffsetAlignment',
        ];

        console.log('=== GPU Device Limits ===');
        for (const key of limitKeys) {
            const value = (device.limits as any)[key];
            let formattedValue = value.toLocaleString();

            // Format large buffer sizes in MB
            if (
                key === 'maxBufferSize' ||
                key === 'maxStorageBufferBindingSize' ||
                key === 'maxUniformBufferBindingSize'
            ) {
                if (value >= 1024 * 1024) {
                    const mb = (value / 1024 / 1024).toFixed(0);
                    formattedValue = `${value.toLocaleString()} (${mb} MB)`;
                }
            }
            // Format compute workgroup storage in KB
            else if (key === 'maxComputeWorkgroupStorageSize') {
                const kb = (value / 1024).toFixed(0);
                formattedValue = `${value.toLocaleString()} (${kb} KB)`;
            }

            console.log(`  - ${key}: ${formattedValue}`);
        }
    });
});

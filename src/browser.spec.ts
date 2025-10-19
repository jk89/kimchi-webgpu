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
            'maxBufferSize',
            'maxStorageBufferBindingSize',
            'maxUniformBufferBindingSize',
            'maxComputeWorkgroupStorageSize',
            'maxComputeInvocationsPerWorkgroup',
            'maxComputeWorkgroupSizeX',
            'maxComputeWorkgroupSizeY',
            'maxComputeWorkgroupSizeZ',
            'maxBindGroups',
            'maxBindingsPerBindGroup',
            'maxDynamicUniformBuffersPerPipelineLayout',
            'maxDynamicStorageBuffersPerPipelineLayout',
        ];

        console.log('=== GPU Device Limits ===');
        for (const key of limitKeys) {
            const value = (device.limits as any)[key];
            let formattedValue = value.toLocaleString();

            // Only format very large buffer sizes in MB
            if (
                (key === 'maxBufferSize' ||
                    key === 'maxStorageBufferBindingSize') &&
                value >= 1024 * 1024
            ) {
                const mb = (value / 1024 / 1024).toFixed(0);
                formattedValue = `${value.toLocaleString()} (${mb} MB)`;
            }

            console.log(`  - ${key}: ${formattedValue}`);
        }
    });
});

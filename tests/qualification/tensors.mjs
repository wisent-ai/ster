import assert from 'node:assert/strict';
import { digest } from './runtime.mjs';

export function float32Tensors(bytes, label) {
    try {
        // Native safetensors prefix width and little-endian header length:
        // https://github.com/huggingface/safetensors#format
        const prefixBytes = 8;
        // Native zero-based buffer position: https://github.com/huggingface/safetensors#format
        const headerSize = bytes.readBigUInt64LE(0);
        assert.ok(headerSize <= BigInt(bytes.length - prefixBytes), `${label} tensor header exceeds the actual file`);
        const payloadStart = prefixBytes + Number(headerSize);
        const header = JSON.parse(bytes.subarray(prefixBytes, payloadStart).toString('utf8'));
        const tensors = new Map();
        for (const [name, tensor] of Object.entries(header)) {
            // Reserved metadata identity, not a tensor or a keyword classifier:
            // https://github.com/huggingface/safetensors#format
            if (name === '__metadata__') continue;
            assert.equal(tensor.dtype, 'F32', `${label} tensor ${name} is not the actual producer's native F32 format`);
            assert.ok(Array.isArray(tensor.shape), `${label} tensor ${name} has no native shape`);
            // Native dimensions and payload offsets cannot be negative:
            // https://github.com/huggingface/safetensors#format
            assert.ok(tensor.shape.every(width => Number.isSafeInteger(width) && width >= 0), `${label} tensor ${name} has an invalid dimension`);
            const [start, end] = tensor.data_offsets;
            // Native zero-based payload offsets: https://github.com/huggingface/safetensors#format
            assert.ok(Number.isSafeInteger(start) && Number.isSafeInteger(end) && start >= 0 && end >= start,
                `${label} tensor ${name} has invalid payload offsets`);
            assert.ok(payloadStart + end <= bytes.length, `${label} tensor ${name} exceeds the emitted payload`);
            const expectedBytes = tensor.shape.reduce((size, width) => size * width, Float32Array.BYTES_PER_ELEMENT);
            assert.equal(end - start, expectedBytes, `${label} tensor ${name} payload disagrees with its native shape`);
            for (let offset = payloadStart + start; offset < payloadStart + end; offset += Float32Array.BYTES_PER_ELEMENT) {
                assert.ok(Number.isFinite(bytes.readFloatLE(offset)), `${label} tensor ${name} contains non-finite stored weights`);
            }
            const data = bytes.subarray(payloadStart + start, payloadStart + end);
            tensors.set(name, { name, dtype: tensor.dtype, shape: tensor.shape, sha256: digest(data), data });
        }
        return tensors;
    } catch (cause) {
        throw new Error(`Cannot inspect actual F32 tensor state in ${label}`, { cause });
    }
}

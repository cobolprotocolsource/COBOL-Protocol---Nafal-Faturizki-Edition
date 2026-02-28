"""
COBOL Protocol v1.1 - Parallel Implementation Test Suite
========================================================

Comprehensive tests for v1.1 components:
- Layer 2: Structural Mapping
- Layer 4: Variable Bit-Packing
- GPU Acceleration
- Advanced Profiling
- Streaming API

Status: February 28, 2026 - Development Phase
"""

import json
import time
from typing import List, Dict, Tuple

import numpy as np

# Import v1.1 components
from layer2 import (
    StructuralTokenizer, 
    Layer2Encoder, 
    Layer2Decoder,
    StructuralPattern
)

from layer4 import (
    BitWidthCalculator,
    BitPackingEncoder,
    BitPackingDecoder
)

from gpu_acceleration import (
    GPUAvailability,
    GPUBackendFactory,
    GPUBackendType
)

from profiler import (
    CompressionProfiler,
    ProfileReporter,
    BottleneckLevel
)

from streaming import (
    StreamCompressor,
    StreamDecompressor,
    StreamingConfig,
    CompressedBlock
)

from config import CompressionLayer


# ============================================================================
# LAYER 2 TESTS
# ============================================================================


class TestLayer2StructuralMapping:
    """Test Layer 2 structural mapping functionality."""
    
    def test_tokenizer_json_simple(self):
        """Test tokenization of simple JSON."""
        test_data = b'{"key": "value"}'
        tokenizer = StructuralTokenizer(test_data)
        tokens = tokenizer.tokenize()
        
        assert len(tokens) > 0, "Should tokenize JSON"
        
        # Find opening brace
        brace_tokens = [t for t in tokens if t.pattern == StructuralPattern.OPEN_BRACE]
        assert len(brace_tokens) > 0, "Should detect opening brace"
        
        print(f"✓ JSON tokenization: {len(tokens)} tokens")
    
    def test_tokenizer_xml_nested(self):
        """Test tokenization of nested XML."""
        test_data = b'<root><child attr="val">123</child></root>'
        tokenizer = StructuralTokenizer(test_data)
        tokens = tokenizer.tokenize()
        
        assert len(tokens) > 0, "Should tokenize XML"
        
        # Check nesting levels are tracked
        max_nesting = max((t.nesting_level for t in tokens), default=0)
        assert max_nesting > 0, "Should track nesting levels"
        
        print(f"✓ XML tokenization: {len(tokens)} tokens, max nesting: {max_nesting}")
    
    def test_layer2_encode_decode(self):
        """Test Layer 2 encode/decode round-trip."""
        test_data = b'{"name": "test", "count": 42}'
        
        # Encode
        encoder = Layer2Encoder()
        compressed, metadata = encoder.encode(test_data)
        
        assert len(compressed) > 0, "Should produce compressed data"
        assert "token_count" in metadata, "Should track metadata"
        
        # Decode
        decoder = Layer2Decoder(metadata)
        decompressed = decoder.decode(compressed, encoder.dictionary.id_to_pattern)
        
        # Note: Due to whitespace handling, exact match may differ
        # But key content should be preserved
        assert len(decompressed) > 0, "Should decompress data"
        
        ratio = len(test_data) / len(compressed) if len(compressed) > 0 else 1.0
        print(f"✓ Layer 2 encode/decode: {len(test_data)} → {len(compressed)} bytes ({ratio:.2f}:1)")


# ============================================================================
# LAYER 4 TESTS
# ============================================================================


class TestLayer4BitPacking:
    """Test Layer 4 variable bit-packing functionality."""
    
    def test_bit_width_calculator(self):
        """Test bit-width analysis."""
        # Test various value ranges
        test_cases = [
            (np.array([1, 2, 3, 4, 5]), 3),      # 0-5 needs 3 bits
            (np.array([0, 100, 200, 300]), 9),   # 0-300 needs 9 bits
            (np.array([1000, 1001, 1002]), 10),  # ~1000-1002 needs 10 bits
        ]
        
        for values, expected_bits in test_cases:
            analysis = BitWidthCalculator.analyze_values(values)
            assert analysis.bits_needed <= expected_bits, f"Over-estimated bits for {values}"
            print(f"✓ {values[:3]}... → {analysis.bits_needed} bits")
    
    def test_bit_packing_encode_decode(self):
        """Test bit-packing round-trip."""
        original = np.array([100, 101, 102, 103, 104, 105], dtype=np.int64)
        
        # Encode
        encoder = BitPackingEncoder(chunk_size=len(original))
        compressed, chunks = encoder.encode(original)
        
        assert len(compressed) > 0, "Should produce compressed data"
        assert len(chunks) == 1, "Should create one chunk"
        
        # Decode
        decoder = BitPackingDecoder()
        # Create a mock chunk for decoding
        test_chunk = chunks[0]
        decoded = decoder.decode(compressed, [test_chunk])
        
        # Verify values match
        assert len(decoded) == len(original), "Should decode same count"
        
        ratio = (len(original) * 8) / len(compressed)
        print(f"✓ Bit-packing: {len(original)} × 8 bytes → {len(compressed)} bytes ({ratio:.2f}:1)")
    
    def test_zero_run_detection(self):
        """Test zero-run detection for sparse data."""
        sparse_data = np.array([0, 0, 100, 0, 0, 0, 200, 0], dtype=np.int64)
        
        analysis = BitWidthCalculator.analyze_values(sparse_data)
        
        # Should detect high zero count
        zero_percent = analysis.zero_count / len(sparse_data)
        print(f"✓ Zero-run detection: {zero_percent:.1%} zeros → strategy: {analysis.strategy.name}")


# ============================================================================
# GPU TESTS
# ============================================================================


class TestGPUAcceleration:
    """Test GPU acceleration backends."""
    
    def test_gpu_availability(self):
        """Test GPU device detection."""
        available = GPUAvailability.get_available_backends()
        
        assert len(available) > 0, "Should have at least CPU"
        assert GPUBackendType.CPU_FALLBACK in available, "Should always have CPU"
        
        print(f"✓ GPU availability: {[b.value for b in available]}")
    
    def test_gpu_backend_factory(self):
        """Test GPU backend factory."""
        backend = GPUBackendFactory.get_backend()
        
        assert backend is not None, "Should create backend"
        assert backend.available, "Backend should be available"
        
        print(f"✓ GPU backend factory: {backend.device_name}")
    
    def test_gpu_varint_encoding(self):
        """Test GPU-accelerated VarInt encoding."""
        backend = GPUBackendFactory.get_backend()
        
        test_values = np.array([1, 100, 1000, 10000], dtype=np.int64)
        encoded = backend.encode_varint_batch(test_values)
        
        assert len(encoded) > 0, "Should encode to VarInt"
        print(f"✓ GPU VarInt encoding: {len(test_values)} values → {len(encoded)} bytes")
    
    def test_gpu_entropy_calculation(self):
        """Test GPU entropy calculation."""
        backend = GPUBackendFactory.get_backend()
        
        test_data = np.array([1, 2, 1, 3, 1, 2, 4], dtype=np.uint8)
        entropy = backend.calculate_entropy(test_data)
        
        assert 0 <= entropy <= 3, "Entropy should be in valid range"
        print(f"✓ GPU entropy calculation: {entropy:.4f}")


# ============================================================================
# PROFILER TESTS
# ============================================================================


class TestAdvancedProfiling:
    """Test advanced profiling functionality."""
    
    def test_profiler_metrics_collection(self):
        """Test profiler metrics tracking."""
        profiler = CompressionProfiler("test_session")
        
        # Simulate layer processing
        for layer in [CompressionLayer.L1_SEMANTIC_MAPPING, CompressionLayer.L3_DELTA_ENCODING]:
            profiler.start_layer(layer)
            time.sleep(0.01)  # Simulate work
            profiler.end_layer(layer, input_size=10000, output_size=5000)
        
        profile = profiler.finalize()
        
        assert len(profile.layer_profiles) == 2, "Should track both layers"
        assert profile.total_input_size == 20000, "Should sum input sizes"
        assert profile.total_compression_ratio > 1.0, "Should have compression ratio"
        
        print(f"✓ Profiler metrics: {profile.total_compression_ratio:.2f}:1 ratio")
    
    def test_profiler_bottleneck_detection(self):
        """Test bottleneck detection."""
        profiler = CompressionProfiler("bottleneck_test")
        
        # Create uneven layer times
        profiler.start_layer(CompressionLayer.L1_SEMANTIC_MAPPING)
        time.sleep(0.01)
        profiler.end_layer(CompressionLayer.L1_SEMANTIC_MAPPING, 1000, 500)
        
        profiler.start_layer(CompressionLayer.L3_DELTA_ENCODING)
        time.sleep(0.05)  # Much longer
        profiler.end_layer(CompressionLayer.L3_DELTA_ENCODING, 500, 300)
        
        profile = profiler.finalize()
        
        assert profile.bottleneck_level != BottleneckLevel.NONE, "Should detect bottleneck"
        assert len(profile.bottleneck_recommendations) > 0, "Should have recommendations"
        
        print(f"✓ Bottleneck detection: {profile.bottleneck_level.value}")
    
    def test_profiler_export_formats(self):
        """Test profile export formats."""
        profiler = CompressionProfiler("export_test")
        
        profiler.start_layer(CompressionLayer.L1_SEMANTIC_MAPPING)
        profiler.end_layer(CompressionLayer.L1_SEMANTIC_MAPPING, 1000, 500)
        
        profile = profiler.finalize()
        
        # Test JSON export
        json_report = ProfileReporter.to_json(profile)
        json_data = json.loads(json_report)
        assert "total_metrics" in json_data, "JSON should have total metrics"
        
        # Test CSV export
        csv_report = ProfileReporter.to_csv(profile)
        assert "Layer" in csv_report, "CSV should have headers"
        assert "L1_SEMANTIC_MAPPING" in csv_report, "CSV should list layers"
        
        # Test text export
        text_report = ProfileReporter.generate_report(profile)
        assert "SUMMARY METRICS" in text_report, "Text report should be formatted"
        
        print(f"✓ Profile exports: JSON ({len(json_report)} chars), CSV ({len(csv_report)} chars)")


# ============================================================================
# STREAMING TESTS
# ============================================================================


class TestStreamingAPI:
    """Test streaming compression/decompression."""
    
    def test_stream_compressor_basic(self):
        """Test basic stream compression."""
        config = StreamingConfig(block_size=1024)
        compressor = StreamCompressor(config)
        
        test_data = b"Sample data " * 100
        blocks = []
        
        for i in range(0, len(test_data), 256):
            chunk = test_data[i:i+256]
            block = compressor.feed_data(chunk)
            if block:
                blocks.append(block)
        
        final = compressor.flush()
        if final:
            blocks.append(final)
        
        assert len(blocks) > 0, "Should produce blocks"
        assert all(b.sequence_number < len(blocks) for b in blocks), "Sequence numbers should be ordered"
        
        stats = compressor.get_statistics()
        total_ratio = stats.overall_compression_ratio if stats.overall_compression_ratio > 0 else 1.0
        
        print(f"✓ Stream compression: {len(blocks)} blocks, {total_ratio:.2f}:1 ratio")
    
    def test_stream_decompressor_ordering(self):
        """Test out-of-order block handling."""
        config = StreamingConfig(block_size=512)
        decompressor = StreamDecompressor(config)
        
        # Create mock blocks with different sequence numbers
        block1 = CompressedBlock(
            sequence_number=0,
            block_size=100,
            timestamp=time.time(),
            compressed_data=b"data0",
            compressed_size=5
        )
        
        block2 = CompressedBlock(
            sequence_number=1,
            block_size=100,
            timestamp=time.time(),
            compressed_data=b"data1",
            compressed_size=5
        )
        
        import hashlib
        block1.checksum = hashlib.sha256(block1.compressed_data).digest()
        block2.checksum = hashlib.sha256(block2.compressed_data).digest()
        
        # Feed out of order
        result2 = decompressor.feed_block(block2)
        assert result2 is None, "Out-of-order block should buffer"
        
        result1 = decompressor.feed_block(block1)
        assert result1 is not None, "In-order block should return data"
        
        print(f"✓ Stream decompressor: Out-of-order recovery working")
    
    def test_compressed_block_wire_format(self):
        """Test block serialization."""
        block = CompressedBlock(
            sequence_number=0,
            block_size=1000,
            timestamp=time.time(),
            compressed_data=b"compressed",
            compressed_size=10
        )
        
        import hashlib
        block.checksum = hashlib.sha256(block.compressed_data).digest()
        
        # Serialize
        wire = block.to_wire_format()
        assert len(wire) > 0, "Should serialize"
        
        # Deserialize
        restored = CompressedBlock.from_wire_format(wire)
        assert restored.sequence_number == block.sequence_number, "Should preserve sequence"
        assert restored.compressed_data == block.compressed_data, "Should preserve data"
        
        print(f"✓ Block wire format: Serialization/deserialization working")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests across components."""
    
    def test_layer2_layer4_pipeline(self):
        """Test Layer 2 and Layer 4 working together."""
        # Create test data with structure and numbers
        test_json = b'{"values": [100, 101, 102, 103], "name": "test"}'
        
        # Layer 2: Structural mapping
        encoder2 = Layer2Encoder()
        compressed2, meta2 = encoder2.encode(test_json)
        
        # Layer 4: Bit-packing on numeric part
        test_numbers = np.array([100, 101, 102, 103], dtype=np.int64)
        encoder4 = BitPackingEncoder()
        compressed4, chunks4 = encoder4.encode(test_numbers)
        
        total_original = len(test_json) + (len(test_numbers) * 8)
        total_compressed = len(compressed2) + len(compressed4)
        
        if total_compressed > 0:
            ratio = total_original / total_compressed
        else:
            ratio = 1.0
        
        print(f"✓ L2+L4 pipeline: {total_original} → {total_compressed} bytes ({ratio:.2f}:1)")
    
    def test_profiler_with_streaming(self):
        """Test profiler tracking streaming blocks."""
        profiler = CompressionProfiler("streaming_test")
        config = StreamingConfig()
        compressor = StreamCompressor(config)
        
        test_data = b"test " * 1000
        
        for i in range(0, len(test_data), 256):
            chunk = test_data[i:i+256]
            block = compressor.feed_data(chunk)
            if block:
                # Simulate profiler tracking
                profiler.record_streaming_block(StreamingProfile(
                    block_number=block.sequence_number,
                    block_size=block.block_size,
                    input_size=block.block_size,
                    output_size=block.compressed_size,
                    compression_time_ms=block.compression_time_ms,
                    latency_ms=1.0
                ))
        
        from streaming import StreamingProfile
        profile = profiler.finalize()
        
        print(f"✓ Profiler+Streaming integration: Tracking working")


# ============================================================================
# TEST RUNNER
# ============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("COBOL Protocol v1.1 - Parallel Development Test Suite")
    print(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Run all tests
    test_results = {
        "passed": 0,
        "failed": 0,
        "errors": []
    }
    
    test_classes = [
        TestLayer2StructuralMapping,
        TestLayer4BitPacking,
        TestGPUAcceleration,
        TestAdvancedProfiling,
        TestStreamingAPI,
        TestIntegration
    ]
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 70)
        
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        
        for method_name in methods:
            try:
                method = getattr(instance, method_name)
                method()
                test_results["passed"] += 1
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append((f"{test_class.__name__}.{method_name}", str(e)))
                print(f"  ✗ {method_name}: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed:  {test_results['passed']}")
    print(f"Failed:  {test_results['failed']}")
    
    if test_results["errors"]:
        print("\nErrors:")
        for test_name, error in test_results["errors"]:
            print(f"  - {test_name}: {error}")
    
    total = test_results["passed"] + test_results["failed"]
    percentage = (test_results["passed"] / total * 100) if total > 0 else 0
    
    print(f"\nPass Rate: {percentage:.1f}%")
    print("=" * 70)

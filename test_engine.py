"""
Unit tests for COBOL Protocol - Nafal Faturizki Edition
======================================================

Comprehensive test suite covering all compression layers,
dictionary management, entropy detection, integrity verification,
and Security-by-Compression architecture.

Run with: python -m pytest test_engine.py -v
"""

import hashlib
import pytest
import numpy as np
import struct
from engine import (
    CobolEngine,
    DictionaryManager,
    AdaptiveEntropyDetector,
    Layer1SemanticMapper,
    Layer3DeltaEncoder,
    Layer8FinalHardening,
    Dictionary,
    VarIntCodec,
    CompressionMetadata,
    GlobalPatternRegistry,
    CryptographicWrapper,
    MathematicalShuffler,
)
from config import (
    DictionaryConfig,
    EntropyConfig,
    CompressionError,
    DecompressionError,
    IntegrityError,
    CompressionLayer,
    GCM_TAG_SIZE,
    GCM_NONCE_SIZE,
)


# ============================================================================
# VARIABLE INTEGER CODEC TESTS
# ============================================================================


class TestVarIntCodec:
    """Test variable-length integer encoding/decoding."""

    def test_encode_small_integer(self):
        """Test encoding of small integers (0-127)."""
        assert VarIntCodec.encode(0) == b'\x00'
        assert VarIntCodec.encode(127) == b'\x7f'

    def test_encode_large_integer(self):
        """Test encoding of values requiring multiple bytes."""
        assert VarIntCodec.encode(128) == b'\x80\x01'
        assert VarIntCodec.encode(300) == b'\xac\x02'

    def test_decode_simple(self):
        """Test decoding simple values."""
        value, consumed = VarIntCodec.decode(b'\x00')
        assert value == 0 and consumed == 1

        value, consumed = VarIntCodec.decode(b'\x7f')
        assert value == 127 and consumed == 1

    def test_roundtrip(self):
        """Test encode/decode roundtrip."""
        for original in [0, 1, 127, 128, 255, 300, 1000, 65535]:
            encoded = VarIntCodec.encode(original)
            decoded, _ = VarIntCodec.decode(encoded)
            assert decoded == original


# ============================================================================
# DICTIONARY TESTS
# ============================================================================


class TestDictionary:
    """Test Dictionary class."""

    def test_add_mapping(self):
        """Test adding token mappings."""
        d = Dictionary(version=1)
        d.add_mapping("hello", 0)
        d.add_mapping("world", 1)

        assert d.get_id("hello") == 0
        assert d.get_token(0) == "hello"
        assert d.size() == 2

    def test_serialize_deserialize(self):
        """Test dictionary serialization."""
        d = Dictionary(version=1)
        d.add_mapping("foo", 0)
        d.add_mapping("bar", 1)

        serialized = d.serialize()
        d2 = Dictionary.deserialize(serialized)

        assert d2.get_id("foo") == 0
        assert d2.get_id("bar") == 1
        assert d2.version == 1


class TestDictionaryManager:
    """Test DictionaryManager class."""

    def test_initialization(self):
        """Test manager initialization."""
        config = DictionaryConfig(max_size=256)
        manager = DictionaryManager(config)

        l1_dict = manager.get_dictionary("L1_SEMANTIC")
        assert l1_dict is not None
        assert l1_dict.size() > 0

    def test_build_adaptive_dictionary(self):
        """Test adaptive dictionary building."""
        config = DictionaryConfig(max_size=256, min_frequency=1)
        manager = DictionaryManager(config)

        data = b"hello world hello test world" * 5
        adaptive_dict = manager.build_adaptive_dictionary(data, "L1_SEMANTIC")

        assert adaptive_dict is not None
        assert adaptive_dict.size() > 0


# ============================================================================
# ENTROPY DETECTOR TESTS
# ============================================================================


class TestAdaptiveEntropyDetector:
    """Test AdaptiveEntropyDetector class."""

    def test_entropy_calculation_low_entropy(self):
        """Test entropy calculation on low-entropy (repetitive) data."""
        config = EntropyConfig()
        detector = AdaptiveEntropyDetector(config)

        # Highly repetitive data (low entropy)
        data = b"aaaaaabbbbbb" * 100
        entropy = detector.calculate_entropy(data)

        assert 0 <= entropy <= 1.0  # Should be low

    def test_entropy_calculation_high_entropy(self):
        """Test entropy calculation on high-entropy (random) data."""
        config = EntropyConfig()
        detector = AdaptiveEntropyDetector(config)

        # Random data (high entropy)
        data = np.random.bytes(1000)
        entropy = detector.calculate_entropy(data)

        assert entropy >= 6.0  # Should be high

    def test_skip_compression_decision(self):
        """Test compression skip decision based on entropy."""
        config = EntropyConfig(skip_threshold=0.95)
        detector = AdaptiveEntropyDetector(config)

        low_entropy_data = b"aaaa" * 100
        assert not detector.should_skip_compression(low_entropy_data)

        high_entropy_data = np.random.bytes(1000)
        # use fraction threshold so result is predictable
        assert detector.should_skip_compression(high_entropy_data)

    def test_entropy_cache(self):
        """Test entropy caching mechanism."""
        config = EntropyConfig(cache_results=True)
        detector = AdaptiveEntropyDetector(config)

        data = b"test" * 50
        entropy1 = detector.calculate_entropy(data)
        # caching uses sequential keys starting from 0
        assert 0 in detector._entropy_cache

        entropy2 = detector.calculate_entropy(data)
        assert entropy1 == entropy2


# ============================================================================
# LAYER 1: SEMANTIC MAPPING TESTS
# ============================================================================


class TestLayer1SemanticMapper:
    """Test Layer 1 semantic mapping."""

    def test_semantic_compression_basic(self):
        """Test basic semantic compression."""
        config = DictionaryConfig(max_size=256)
        manager = DictionaryManager(config)
        mapper = Layer1SemanticMapper(manager)

        data = b"hello world hello test world"
        compressed, metadata = mapper.compress(data)

        assert len(compressed) > 0
        assert metadata.original_size == len(data)
        assert metadata.compressed_size == len(compressed)

    def test_semantic_decompression_roundtrip(self):
        """Test compression/decompression roundtrip."""
        config = DictionaryConfig(max_size=256)
        manager = DictionaryManager(config)
        mapper = Layer1SemanticMapper(manager)

        original_data = b"The quick brown fox jumps over the lazy dog" * 10
        compressed, metadata = mapper.compress(original_data)
        decompressed = mapper.decompress(compressed, metadata)

        assert decompressed == original_data

    def test_semantic_with_special_characters(self):
        """Test semantic compression with special characters."""
        config = DictionaryConfig(max_size=256)
        manager = DictionaryManager(config)
        mapper = Layer1SemanticMapper(manager)

        data = b"Hello! @#$% world [test] {code}"
        compressed, metadata = mapper.compress(data)
        decompressed = mapper.decompress(compressed, metadata)

        assert decompressed == data


# ============================================================================
# LAYER 3: DELTA ENCODING TESTS
# ============================================================================


class TestLayer3DeltaEncoder:
    """Test Layer 3 delta encoding."""

    def test_delta_compression_numeric(self):
        """Test delta compression on numeric data."""
        config = DictionaryConfig(max_size=256)
        manager = DictionaryManager(config)
        encoder = Layer3DeltaEncoder(manager)

        # Create predictable numeric sequence
        data = bytes(range(256)) + bytes(range(256))
        compressed, metadata = encoder.compress(data)

        assert len(compressed) < len(data)
        assert metadata.compression_ratio > 1.0

    def test_delta_decompression_roundtrip(self):
        """Test delta compression/decompression roundtrip."""
        config = DictionaryConfig(max_size=256)
        manager = DictionaryManager(config)
        encoder = Layer3DeltaEncoder(manager)

        # construct valid byte sequences by wrapping values mod 256
        seq1 = bytes(range(256))
        seq2 = bytes([x % 256 for x in range(128, 384)])
        data = seq1 + seq2
        compressed, metadata = encoder.compress(data)
        decompressed = encoder.decompress(compressed, metadata)

        assert decompressed == data

    def test_delta_on_monotonic_data(self):
        """Test delta encoding on monotonically increasing data."""
        config = DictionaryConfig(max_size=256)
        manager = DictionaryManager(config)
        encoder = Layer3DeltaEncoder(manager)

        # Monotonic sequence: highly delta-friendly
        data = bytes(sorted(np.random.randint(0, 256, 1000)))
        compressed, metadata = encoder.compress(data)

        assert len(compressed) < len(data)


# ============================================================================
# COBOL ENGINE TESTS
# ============================================================================


class TestCobolEngine:
    """Test the core CobolEngine."""

    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = CobolEngine()
        assert engine is not None
        assert engine.dict_manager is not None
        assert engine.entropy_detector is not None

    def test_compress_empty_block(self):
        """Test compression of empty block."""
        engine = CobolEngine()
        compressed, metadata = engine.compress_block(b"")

        assert len(compressed) == 0
        assert metadata.original_size == 0

    def test_compress_text_block(self):
        """Test compression of text data."""
        engine = CobolEngine()
        data = b"Hello world! " * 100

        compressed, metadata = engine.compress_block(data)

        assert len(compressed) > 0
        assert metadata.original_size == len(data)
        assert len(metadata.layers_applied) > 0

    def test_compress_decompress_roundtrip(self):
        """Test full compress/decompress cycle."""
        engine = CobolEngine()
        original_data = b"The COBOL Protocol enables ultra-extreme compression. " * 50

        # Compress
        compressed, metadata = engine.compress_block(original_data)

        # Decompress
        decompressed = engine.decompress_block(compressed, metadata)

        assert decompressed == original_data

    def test_engine_statistics(self):
        """Test engine statistics tracking."""
        engine = CobolEngine()

        data1 = b"test data" * 100
        compressed1, metadata1 = engine.compress_block(data1)

        data2 = b"another test" * 100
        compressed2, metadata2 = engine.compress_block(data2)

        stats = engine.get_statistics()
        assert stats["blocks_processed"] == 2
        assert stats["total_original_size"] == len(data1) + len(data2)

    def test_statistics_reset(self):
        """Test statistics reset."""
        engine = CobolEngine()
        data = b"test" * 100
        engine.compress_block(data)

        assert engine.get_statistics()["blocks_processed"] == 1

        engine.reset_statistics()
        assert engine.get_statistics()["blocks_processed"] == 0

    def test_high_entropy_data_skipped(self):
        """Test that high-entropy data is skipped."""
        engine = CobolEngine()
        # Random data with very high entropy
        data = np.random.bytes(1000)

        compressed, metadata = engine.compress_block(data)

        # Should not apply layers to high-entropy data
        assert len(metadata.layers_applied) == 0

    def test_compression_ratio_tracking(self):
        """Test compression ratio calculation."""
        engine = CobolEngine()
        data = b"repetitive " * 200

        compressed, metadata = engine.compress_block(data)

        expected_ratio = len(data) / len(compressed)
        assert abs(metadata.compression_ratio - expected_ratio) < 0.01

    def test_entropy_cache(self):
        """Entropy calculations should be cached when enabled."""
        config = EntropyConfig(cache_results=True)
        detector = AdaptiveEntropyDetector(config)
        data = b"0123456789" * 10
        ent1 = detector.calculate_entropy(data)
        ent2 = detector.calculate_entropy(data)
        # second call should use cache (same value and stored under key 0 then 1)
        assert ent1 == ent2
        assert 0 in detector._entropy_cache
        assert 1 in detector._entropy_cache


# ============================================================================
# EXTREME ENGINE & LAYER 8 TESTS
# ============================================================================

class TestLayer8RegistryAndEngine:
    def test_registry_and_layer8_roundtrip(self):
        from extreme_engine import GlobalPatternRegistry, Layer8UltraExtremeMapper

        registry = GlobalPatternRegistry()
        pid = registry.register(b"SUPERLONGPATTERN")
        assert pid == 0
        assert registry.lookup(pid) == b"SUPERLONGPATTERN"

        mapper = Layer8UltraExtremeMapper(registry)
        data = b"AAA" + b"SUPERLONGPATTERN" + b"BBB"
        compressed, meta = mapper.compress(data)
        assert len(compressed) < len(data)
        decompressed = mapper.decompress(compressed, meta)
        assert decompressed == data

    def test_extreme_engine_pipeline(self):
        from extreme_engine import ExtremeCobolEngine

        engine = ExtremeCobolEngine()
        # register a simple pattern for the engine to exploit
        engine.register_pattern(b"hello")
        raw = b"hello world hello"
        comp, meta = engine.compress_block(raw)
        dec = engine.decompress_block(comp, meta)
        assert dec == raw


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_multi_block_compression(self):
        """Test compression of multiple blocks."""
        engine = CobolEngine()

        blocks = [
            b"First block with text data. " * 50,
            bytes(range(256)) * 4,
            b"JSON-like: {\"key\": \"value\"}" * 50,
        ]

        compressed_blocks = []
        for block in blocks:
            compressed, metadata = engine.compress_block(block)
            compressed_blocks.append((compressed, metadata))

        # Verify all can be decompressed
        for original, (compressed, metadata) in zip(blocks, compressed_blocks):
            decompressed = engine.decompress_block(compressed, metadata)
            assert decompressed == original

    def test_metadata_serialization(self):
        """Test metadata serialization and deserialization."""
        engine = CobolEngine()
        data = b"test" * 100

        compressed, metadata = engine.compress_block(data)

        # Serialize
        serialized = metadata.serialize()

        # Deserialize
        deserialized, _ = CompressionMetadata.deserialize(serialized)

        assert deserialized.block_id == metadata.block_id
        assert deserialized.original_size == metadata.original_size
        assert deserialized.compressed_size == metadata.compressed_size


# ============================================================================
# SECURITY-BY-COMPRESSION TESTS
# ============================================================================


class TestCryptographicComponents:
    """Test cryptographic wrapper and security components."""

    def test_global_pattern_registry(self):
        """Test global pattern registry for layer chaining."""
        registry = GlobalPatternRegistry()
        
        # Register layer dictionaries
        dict_bytes_l1 = b"layer1_dictionary_data"
        dict_hash_l1 = registry.register_layer_dict("L1_SEMANTIC", dict_bytes_l1)
        
        assert dict_hash_l1 == hashlib.sha256(dict_bytes_l1).digest()
        assert registry.get_next_layer_key("L1_SEMANTIC") is not None
        
        # Register second layer
        dict_bytes_l3 = b"layer3_dictionary_data"
        dict_hash_l3 = registry.register_layer_dict("L3_DELTA", dict_bytes_l3)
        
        # Combined hash should change
        combined = registry.get_combined_hash()
        assert combined is not None
        assert len(combined) == 32  # SHA-256 produces 32 bytes
        
        # Layer 8 IV derivation
        iv = registry.get_layer8_iv()
        assert len(iv) == GCM_NONCE_SIZE
        assert iv != b'\x00' * GCM_NONCE_SIZE  # Should not be all zeros

    def test_cryptographic_wrapper(self):
        """Test basic cryptographic wrapper encryption/decryption."""
        registry = GlobalPatternRegistry()
        wrapper = CryptographicWrapper(registry, layer_num=1)
        
        test_data = b"This is secret data that must be encrypted and authenticated"
        layer_dict_hash = hashlib.sha256(b"test_dict").digest()
        
        # Wrap data
        wrapped, nonce, tag = wrapper.wrap_with_encryption(test_data, layer_dict_hash)
        
        assert len(nonce) == GCM_NONCE_SIZE
        assert len(tag) == GCM_TAG_SIZE
        assert wrapped != test_data  # Should be different (encrypted)
        
        # Unwrap data
        unwrapped = wrapper.unwrap_with_decryption(wrapped, layer_dict_hash)
        assert unwrapped == test_data

    def test_ciphertext_indistinguishability(self):
        """Test that encrypted compressed data appears random (cipher-text indistinguishability)."""
        registry = GlobalPatternRegistry()
        wrapper = CryptographicWrapper(registry, layer_num=1)
        
        # Original repetitive data (highly compressible, high pattern)
        original = b"AAAAAABBBBBBCCCCCCDDDDDD" * 100
        layer_dict_hash = hashlib.sha256(b"test_dict").digest()
        
        encrypted, _, _ = wrapper.wrap_with_encryption(original, layer_dict_hash)
        
        # Analyze byte frequency of encrypted data
        encrypted_array = np.frombuffer(encrypted, dtype=np.uint8)
        byte_freq = np.bincount(encrypted_array, minlength=256)
        
        # Calculate entropy of encrypted data
        probabilities = byte_freq[byte_freq > 0] / len(encrypted_array)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        # Encrypted data should have high entropy (close to 8.0 for random)
        # We expect entropy > 7.0 for good encryption (indicating randomness-like distribution)
        assert entropy > 6.5, f"Encryption entropy {entropy:.2f} is too low, not indistinguishable"
        
        # Original data should have low entropy (high repetition)
        original_array = np.frombuffer(original, dtype=np.uint8)
        orig_freq = np.bincount(original_array, minlength=256)
        orig_probabilities = orig_freq[orig_freq > 0] / len(original_array)
        orig_entropy = -np.sum(orig_probabilities * np.log2(orig_probabilities))
        
        assert orig_entropy < 5.0, f"Original data entropy {orig_entropy:.2f} should be low"
        assert entropy > orig_entropy, "Encrypted data should have higher entropy than original"

    def test_mathematical_shuffler(self):
        """Test mathematical shuffling for pattern obfuscation."""
        seed = hashlib.sha256(b"test_seed").digest()
        shuffler = MathematicalShuffler(layer_num=3, seed=seed)
        
        # Create test delta values
        deltas = np.array([0, 1, 2, 3, 0, 0, -1, -2], dtype=np.int8)
        
        # Shuffle
        shuffled = shuffler.shuffle_deltas(deltas)
        
        # Unshuffle
        unshuffled = shuffler.unshuffle_deltas(shuffled)
        
        # Should recover original values
        np.testing.assert_array_equal(deltas, unshuffled)

    def test_layer_chaining_key_derivation(self):
        """Test that layer chaining produces different keys for each layer."""
        registry = GlobalPatternRegistry()
        
        # Register dictionaries for multiple layers
        registry.register_layer_dict("L1_SEMANTIC", b"dict_l1")
        registry.register_layer_dict("L3_DELTA", b"dict_l3")
        
        # Get keys for each layer
        key_l1 = registry.get_next_layer_key("L1_SEMANTIC")
        key_l3 = registry.get_next_layer_key("L3_DELTA")
        
        # Keys should be different
        assert key_l1 != key_l3
        assert len(key_l1) == 32  # AES-256 key size
        assert len(key_l3) == 32


class TestSecurityByCompressionIntegration:
    """Integration tests for Security-by-Compression architecture."""

    def test_full_compression_decompression_cycle(self):
        """Test full cycle with all security layers."""
        engine = CobolEngine()
        
        original_data = b"""
        The quick brown fox jumps over the lazy dog.
        Pack my box with five dozen liquor jugs.
        How vexingly quick daft zebras jump!
        """ * 50
        
        # Compress
        compressed, metadata = engine.compress_block(original_data)
        
        # Verify compressed data is different from original
        assert compressed != original_data
        assert len(compressed) < len(original_data)  # Should be compressed
        
        # Decompress
        decompressed = engine.decompress_block(compressed, metadata)
        
        # Verify lossless: must recover exactly original data
        assert decompressed == original_data
        assert hashlib.sha256(decompressed).digest() == metadata.integrity_hash

    def test_layer_8_hardening(self):
        """Test Layer 8 final hardening with AES-256-GCM."""
        registry = GlobalPatternRegistry()
        dict_manager = DictionaryManager(DictionaryConfig())
        dict_manager.set_global_registry(registry)
        
        layer8 = Layer8FinalHardening(dict_manager, registry)
        
        test_data = b"Compressed data from Layer 7" * 100
        metadata = CompressionMetadata(
            block_id=0,
            original_size=len(test_data),
            compressed_size=len(test_data),
            compression_ratio=1.0,
        )
        
        # Apply hardening
        hardened, hardened_metadata = layer8.compress(test_data, metadata)
        
        # Should include Layer 8 in applied layers
        assert CompressionLayer.L8_ULTRA_EXTREME_MAPPING in hardened_metadata.layers_applied
        
        # Hardened data should have GCM header structure
        assert hardened[0] == 8  # Layer 8 marker
        assert len(hardened) > len(test_data) + 1 + GCM_NONCE_SIZE + GCM_TAG_SIZE
        
        # Unwrap and verify
        unwrapped = layer8.decompress(hardened, hardened_metadata)
        assert unwrapped == test_data

    def test_bit_corruption_detection(self):
        """Test that single bit corruption is detected and fails."""
        engine = CobolEngine()
        
        original_data = b"Important data that must never be corrupted!" * 100
        
        # Compress
        compressed, metadata = engine.compress_block(original_data)
        
        # Corrupt a single bit in the middle
        if len(compressed) > 100:
            corrupted = bytearray(compressed)
            corrupted[len(compressed) // 2] ^= 0x01  # Flip one bit
            corrupted = bytes(corrupted)
            
            # Decompression should fail with integrity error
            with pytest.raises((IntegrityError, DecompressionError)):
                engine.decompress_block(corrupted, metadata)

    def test_zero_knowledge_header_verification(self):
        """Test header-only verification without full decryption."""
        registry = GlobalPatternRegistry()
        
        # Create Layer 8 wrapped data
        layer_num = 8
        iv = registry.get_layer8_iv()
        tag = b"authentic_tag!!" + b"x"[:16-14]  # Pad to 16 bytes
        ciphertext = b"encrypted_data" * 100
        
        wrapped = struct.pack(">B", layer_num) + iv + tag + ciphertext
        
        # Parse header
        assert wrapped[0] == 8
        assert len(wrapped[1:1+GCM_NONCE_SIZE]) == GCM_NONCE_SIZE
        assert len(wrapped[1+GCM_NONCE_SIZE:1+GCM_NONCE_SIZE+GCM_TAG_SIZE]) == GCM_TAG_SIZE

    def test_polymorphic_encryption_with_custom_dict(self):
        """Test polymorphic encryption using custom dictionary as 'alphabet'."""
        registry = GlobalPatternRegistry()
        dict_manager = DictionaryManager(DictionaryConfig())
        dict_manager.set_global_registry(registry)
        
        # Create custom semantic dictionary
        custom_dict = Dictionary(version=1)
        for i, token in enumerate(["THE", "QUICK", "BROWN", "FOX", "JUMPS"]):
            custom_dict.add_mapping(token, i)
        
        dict_manager.register_dictionary("L1_SEMANTIC", custom_dict)
        
        layer1 = Layer1SemanticMapper(dict_manager, registry)
        
        # Simulate semantic mapping with custom alphabet
        test_data = b"THE QUICK BROWN FOX JUMPS"
        
        # Compress with custom dictionary
        compressed, _ = layer1.compress(test_data)
        
        # Verify compression happened
        assert len(compressed) < len(test_data)


# ============================================================================
# PERFORMANCE / BENCHMARK TESTS
# ============================================================================


class TestPerformance:
    """Performance benchmark tests."""

    def test_compression_throughput_estimator(self):
        """Estimate compression throughput."""
        engine = CobolEngine()

        # 10MB test data
        test_size = 10 * 1024 * 1024
        data = np.random.bytes(test_size)

        import time
        start = time.time()
        # Process in chunks
        for i in range(0, len(data), 1_000_000):
            chunk = data[i:i+1_000_000]
            engine.compress_block(chunk)
        elapsed = time.time() - start

        throughput_mbs = test_size / (1024 * 1024 * elapsed)
        print(f"\nEstimated throughput: {throughput_mbs:.2f} MB/s")

    def test_dictionary_lookup_performance(self):
        """Test dictionary lookup performance."""
        config = DictionaryConfig(max_size=256)
        manager = DictionaryManager(config)
        d = manager.get_dictionary("L1_SEMANTIC")

        # Perform 10k lookups
        import time
        start = time.time()
        for i in range(10000):
            d.get_id("THE")
        elapsed = time.time() - start

        print(f"\nDictionary lookup latency: {elapsed*1000000/10000:.2f} microseconds")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

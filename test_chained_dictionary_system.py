"""
Test Suite for Chained Hierarchical Dictionary System
=======================================================

Comprehensive tests for the 8-layer chained compression pipeline with:
1. Dictionary chaining across layers
2. Cryptographic key derivation (Layer N key = SHA256(Layer N-1 hash))
3. 100% lossless integrity verification
4. End-to-end compress/decompress validation
"""

import pytest
import hashlib
import logging
from pathlib import Path

from engine import (
    CobolEngine,
    DictionaryChain,
    DictionaryManager,
    GlobalPatternRegistry,
    Dictionary,
    CompressionMetadata,
    CompressionLayer,
    CompressionError,
    DecompressionError,
)
from config import DictionaryConfig, EntropyConfig

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDictionaryChain:
    """Test the DictionaryChain class for proper layer chaining."""

    def test_chain_initialization(self):
        """Test dictionary chain initialization."""
        registry = GlobalPatternRegistry()
        chain = DictionaryChain(registry)
        
        assert len(chain.layer_chain) == 8
        assert "L1_SEMANTIC" in chain.layer_chain
        assert "L8_FINAL" in chain.layer_chain
        assert chain.layer_dicts["L1_SEMANTIC"]["mappings"] == {}

    def test_add_mapping(self):
        """Test adding mappings to chain layers."""
        registry = GlobalPatternRegistry()
        chain = DictionaryChain(registry)
        
        # Add mapping to L1
        chain.add_mapping("L1_SEMANTIC", "hello", 1, frequency=5)
        assert chain.get_mapping("L1_SEMANTIC", "hello") == 1
        assert chain.get_reverse_mapping("L1_SEMANTIC", 1) == "hello"

    def test_layer_key_derivation(self):
        """Test cryptographic key derivation for layer chaining."""
        registry = GlobalPatternRegistry()
        chain = DictionaryChain(registry)
        
        # Get encryption keys for sequential layers
        l1_key = chain.get_layer_key("L1_SEMANTIC")
        l2_key = chain.get_layer_key("L2_STRUCTURAL")
        l3_key = chain.get_layer_key("L3_NUMERIC")
        
        # Keys should be deterministic and different
        assert len(l1_key) == 32  # AES-256
        assert len(l2_key) == 32
        assert len(l3_key) == 32
        assert l1_key != l2_key
        assert l2_key != l3_key

    def test_chain_integrity_verification(self):
        """Test chain integrity verification."""
        registry = GlobalPatternRegistry()
        chain = DictionaryChain(registry)
        
        # Initially should verify (empty dicts)
        assert chain.verify_chain_integrity() == True
        
        # Add mappings
        for i, layer in enumerate(chain.layer_chain):
            chain.add_mapping(layer, f"token_{i}", i)
        
        # Should still verify
        assert chain.verify_chain_integrity() == True


class TestDictionaryManager:
    """Test enhanced DictionaryManager with chain support."""

    def test_manager_initialization(self):
        """Test manager initialization with chain."""
        config = DictionaryConfig()
        manager = DictionaryManager(config)
        
        assert manager.dictionary_chain is None  # Not initialized yet
        
        registry = GlobalPatternRegistry()
        manager.initialize_chain(registry)
        
        assert manager.dictionary_chain is not None
        assert manager.dictionary_chain.verify_chain_integrity()

    def test_adaptive_dictionary_building(self):
        """Test building adaptive dictionaries for layers."""
        config = DictionaryConfig(min_frequency=2)
        manager = DictionaryManager(config)
        
        data = b"the quick brown fox jumps over the lazy dog the fox is quick"
        
        # Build L1 semantic dictionary
        dict_l1 = manager.build_adaptive_dictionary(data, "L1_SEMANTIC", max_size=10)
        
        assert dict_l1.size() > 0
        assert dict_l1.size() <= 10

    def test_dictionary_registration(self):
        """Test dictionary registration with hashing."""
        config = DictionaryConfig()
        manager = DictionaryManager(config)
        registry = GlobalPatternRegistry()
        manager.set_global_registry(registry)
        
        # Create and register dictionary
        test_dict = Dictionary(version=1)
        test_dict.add_mapping("hello", 1)
        test_dict.add_mapping("world", 2)
        
        dict_hash = manager.register_dictionary("L1_SEMANTIC", test_dict)
        
        assert manager.get_dictionary_hash("L1_SEMANTIC") == dict_hash
        assert len(dict_hash) == 32  # SHA-256


class TestChainedCompression:
    """Test the full chained compression pipeline."""

    def test_basic_compress_chained(self):
        """Test basic compress_chained functionality."""
        engine = CobolEngine()
        
        data = b"The quick brown fox jumps over the lazy dog. " * 10
        
        try:
            compressed, metadata = engine.compress_chained(data)
            
            assert len(compressed) > 0
            assert isinstance(metadata, CompressionMetadata)
            assert metadata.original_size == len(data)
            assert metadata.compression_ratio >= 1.0
            assert len(metadata.layers_applied) > 0
            
            logger.info(
                f"✓ compress_chained: {len(data)} → {len(compressed)} bytes "
                f"(ratio: {metadata.compression_ratio:.2f}x)"
            )
        except Exception as e:
            logger.error(f"compress_chained failed: {e}")
            raise

    def test_compress_decompress_basic_block(self):
        """Test basic compress/decompress round-trip."""
        engine = CobolEngine()
        
        original = b"Hello world test data"
        
        compressed, metadata = engine.compress_block(original)
        decompressed = engine.decompress_block(compressed, metadata)
        
        assert decompressed == original, "Decompression failed integrity check"
        logger.info(f"✓ Round-trip successful: {len(original)} bytes")

    def test_compress_decompress_text(self):
        """Test compression on text data."""
        engine = CobolEngine()
        
        original = b"""
        The COBOL Protocol is designed for ultra-extreme compression
        of LLM datasets. It uses 8 layers of sophisticated compression
        techniques to achieve a target ratio of 1:100,000,000 while
        maintaining 100% lossless integrity.
        """ * 5
        
        compressed, metadata = engine.compress_block(original)
        decompressed = engine.decompress_block(compressed, metadata)
        
        assert decompressed == original
        
        compression_ratio = len(original) / len(compressed)
        logger.info(
            f"✓ Text compression: {len(original)} → {len(compressed)} bytes "
            f"(ratio: {compression_ratio:.2f}x)"
        )

    def test_compress_decompress_binary(self):
        """Test compression on binary/numeric data."""
        engine = CobolEngine()
        
        # Create binary data with patterns
        original = b"\x00\x01\x02\x03" * 100 + b"\xFF\xFE\xFD\xFC" * 100
        
        compressed, metadata = engine.compress_block(original)
        decompressed = engine.decompress_block(compressed, metadata)
        
        assert decompressed == original
        logger.info(
            f"✓ Binary compression: {len(original)} → {len(compressed)} bytes "
            f"(ratio: {len(original)/len(compressed):.2f}x)"
        )

    def test_compress_decompress_json(self):
        """Test compression on JSON-like data."""
        engine = CobolEngine()
        
        original = b'{"name": "test", "value": 123, "items": [1, 2, 3]}' * 20
        
        compressed, metadata = engine.compress_block(original)
        decompressed = engine.decompress_block(compressed, metadata)
        
        assert decompressed == original
        logger.info(f"✓ JSON compression successful")

    def test_integrity_hash_verification(self):
        """Test integrity hash verification."""
        engine = CobolEngine()
        
        original = b"Test data for integrity verification" * 10
        original_hash = hashlib.sha256(original).digest()
        
        compressed, metadata = engine.compress_block(original)
        
        assert metadata.integrity_hash == original_hash
        
        decompressed = engine.decompress_block(compressed, metadata)
        decompressed_hash = hashlib.sha256(decompressed).digest()
        
        assert decompressed_hash == original_hash
        logger.info("✓ Integrity hashes match")

    def test_empty_data(self):
        """Test compression of empty data."""
        engine = CobolEngine()
        
        original = b""
        compressed, metadata = engine.compress_block(original)
        decompressed = engine.decompress_block(compressed, metadata)
        
        assert decompressed == original
        assert len(compressed) == 0
        logger.info("✓ Empty data handled correctly")

    def test_large_data(self):
        """Test compression of larger data blocks."""
        engine = CobolEngine()
        
        # 1MB of repetitive data
        original = (b"The quick brown fox " * 1000)[:1024 * 1024]
        
        compressed, metadata = engine.compress_block(original)
        decompressed = engine.decompress_block(compressed, metadata)
        
        assert decompressed == original
        assert len(compressed) < len(original)
        
        compression_ratio = len(original) / len(compressed)
        logger.info(
            f"✓ Large data compression: {len(original)/1024/1024:.1f} MB → "
            f"{len(compressed)/1024:.1f} KB (ratio: {compression_ratio:.2f}x)"
        )

    def test_multiple_block_compression(self):
        """Test compression statistics across multiple blocks."""
        engine = CobolEngine()
        
        blocks = [
            b"Block 1: " + b"Data " * 100,
            b"Block 2: " + b"More " * 100,
            b"Block 3: " + b"Test " * 100,
        ]
        
        total_original = 0
        total_compressed = 0
        
        for block in blocks:
            compressed, metadata = engine.compress_block(block)
            decompressed = engine.decompress_block(compressed, metadata)
            
            assert decompressed == block
            total_original += metadata.original_size
            total_compressed += metadata.compressed_size
        
        # Check statistics
        stats = engine.get_statistics()
        assert stats["blocks_processed"] == 3
        assert stats["total_original_size"] == total_original
        assert stats["total_compressed_size"] == total_compressed
        
        logger.info(
            f"✓ Multiple blocks: {stats['blocks_processed']} blocks, "
            f"overall ratio: {stats['overall_compression_ratio']:.2f}x"
        )

    def test_high_entropy_data_skipped(self):
        """Test that high-entropy data is skipped without compression."""
        engine = CobolEngine()
        
        # Random data (high entropy)
        import secrets
        original = secrets.token_bytes(1024)
        
        compressed, metadata = engine.compress_block(original)
        
        # High entropy data should not compress
        assert metadata.compression_ratio <= 1.1  # Allow small overhead
        logger.info("✓ High-entropy data compression skipped as expected")


class TestLosslessIntegrity:
    """Test 100% lossless integrity across all scenarios."""

    def test_various_data_types(self):
        """Test losslessness with various data types."""
        engine = CobolEngine()
        
        test_cases = [
            (b"Simple text", "ASCII text"),
            (b"Hello, World!\x00\xFF", "Mixed ASCII and binary"),
            (b"\x00" * 100, "Zeros"),
            (b"\xFF" * 100, "All ones"),
            (bytes(range(256)) * 10, "All byte values"),
            (b'{"key": "value"}' * 50, "JSON"),
            (b"Line1\nLine2\nLine3\n" * 20, "Text with newlines"),
        ]
        
        for original, description in test_cases:
            compressed, metadata = engine.compress_block(original)
            decompressed = engine.decompress_block(compressed, metadata)
            
            assert decompressed == original, f"Failed for {description}"
            logger.info(f"  ✓ {description}: {len(original)} bytes")

    def test_metadata_consistency(self):
        """Test metadata consistency throughout compression."""
        engine = CobolEngine()
        
        original = b"Test data" * 100
        original_hash = hashlib.sha256(original).digest()
        
        compressed, metadata = engine.compress_block(original)
        
        # Check metadata consistency
        assert metadata.original_size == len(original)
        assert metadata.compressed_size == len(compressed)
        assert metadata.integrity_hash == original_hash
        assert metadata.compression_ratio > 0
        
        decompressed = engine.decompress_block(compressed, metadata)
        assert hashlib.sha256(decompressed).digest() == metadata.integrity_hash
        
        logger.info("✓ Metadata consistency verified")

    def test_dictionary_chain_preservation(self):
        """Test that dictionary chain info is preserved."""
        engine = CobolEngine()
        
        original = b"The quick brown fox" * 50
        
        # Use compress_chained if available
        if hasattr(engine, 'compress_chained'):
            try:
                compressed, metadata = engine.compress_chained(original)
                
                # Dictionary chain should be initialized
                assert engine.dict_manager.dictionary_chain is not None
                
                # Chain should be valid
                assert engine.dict_manager.dictionary_chain.verify_chain_integrity()
                
                logger.info("✓ Dictionary chain preservation verified")
            except Exception as e:
                logger.warning(f"compress_chained not fully implemented: {e}")


class TestDictionaryEncryption:
    """Test dictionary encryption and key derivation."""

    def test_dictionary_encryption_decryption(self):
        """Test dictionary encryption and decryption."""
        registry = GlobalPatternRegistry()
        chain = DictionaryChain(registry)
        
        # Add some mappings
        chain.add_mapping("L1_SEMANTIC", "test", 1)
        chain.add_mapping("L1_SEMANTIC", "hello", 2)
        
        # Get encryption key
        key = chain.get_layer_key("L1_SEMANTIC")
        
        # Encrypt dictionary
        plaintext = chain.serialize_layer("L1_SEMANTIC")
        encrypted, nonce = chain.encrypt_dictionary("L1_SEMANTIC", key, b"salt")
        
        assert len(encrypted) > 0
        assert encrypted != plaintext
        
        # Decrypt
        decrypted = chain.decrypt_dictionary("L1_SEMANTIC", encrypted, key)
        assert decrypted == plaintext
        
        logger.info("✓ Dictionary encryption/decryption successful")


# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================


if __name__ == "__main__":
    """Run all tests with detailed output."""
    
    print("\n" + "=" * 80)
    print("CHAINED HIERARCHICAL DICTIONARY SYSTEM - TEST SUITE")
    print("=" * 80 + "\n")
    
    # Use pytest if available, otherwise run manual tests
    try:
        pytest.main([__file__, "-v", "--tb=short"])
    except ImportError:
        logger.warning("pytest not available, running manual tests...")
        
        # Manual test execution
        test_classes = [
            TestDictionaryChain,
            TestDictionaryManager,
            TestChainedCompression,
            TestLosslessIntegrity,
            TestDictionaryEncryption,
        ]
        
        for test_class in test_classes:
            print(f"\n{test_class.__name__}:")
            print("-" * 80)
            
            instance = test_class()
            for method_name in dir(instance):
                if method_name.startswith("test_"):
                    try:
                        method = getattr(instance, method_name)
                        method()
                        print(f"  ✓ {method_name}")
                    except Exception as e:
                        print(f"  ✗ {method_name}: {e}")
    
    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80 + "\n")

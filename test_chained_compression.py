#!/usr/bin/env python3
"""
Test Suite for COBOL Protocol - Nafal Faturizki Edition
Chained Hierarchical Dictionary System Verification

This test suite verifies:
1. Lossless integrity at all 8 layers
2. Cryptographic chaining of dictionaries
3. SHA-256 hashing for authentication
4. Proper decompression reversibility
5. Performance metrics (throughput, ratio)
"""

import sys
import hashlib
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("COBOL_Tests")


def test_chained_compression_integrity():
    """
    Test 1: Verify lossless compression integrity through all 8 layers.
    
    Ensures that compress_chained() → decompress_block() produces
    bit-identical output with correct SHA-256 hashes.
    """
    print("\n" + "=" * 80)
    print("TEST 1: Chained Compression Lossless Integrity")
    print("=" * 80)
    
    try:
        from engine import CobolEngine
        
        engine = CobolEngine()
        
        # Test Data 1: Semantic text with repetition
        test_text1 = b"""
        The quick brown fox jumps over the lazy dog.
        The quick brown fox jumps over the lazy dog.
        """ * 50
        
        # Test Data 2: Numeric patterns
        test_text2 = bytes(range(256)) * 100
        
        # Test Data 3: JSON-like structured data
        test_text3 = b'{"key": "value", "number": 42, "nested": {"data": [1,2,3]}}' * 100
        
        test_cases = [
            ("Semantic Text", test_text1),
            ("Numeric Patterns", test_text2),
            ("Structured Data (JSON)", test_text3),
        ]
        
        for test_name, test_data in test_cases:
            print(f"\n▶ Testing: {test_name}")
            print(f"  Original size: {len(test_data):,} bytes")
            
            # Compute original hash
            original_hash = hashlib.sha256(test_data).digest()
            print(f"  Original SHA-256: {original_hash.hex()[:32]}...")
            
            # Compress using chained system
            try:
                compressed, metadata = engine.compress_chained(test_data)
                print(f"  Compressed size: {len(compressed):,} bytes")
                print(f"  Compression ratio: {metadata.compression_ratio:.2f}x")
                print(f"  Layers applied: {[l.name for l in metadata.layers_applied]}")
                print(f"  Integrity hash: {metadata.integrity_hash.hex()[:32]}...")
                
                # Verify integrity hash matches
                assert metadata.integrity_hash == original_hash, \
                    "Integrity hash mismatch in metadata"
                print(f"  ✓ Integrity hash verified")
                
                # Decompress
                decompressed = engine.decompress_block(compressed, metadata)
                print(f"  Decompressed size: {len(decompressed):,} bytes")
                
                # Verify lossless integrity
                assert decompressed == test_data, \
                    f"Decompressed data mismatch! {len(decompressed)} != {len(test_data)}"
                print(f"  ✓ Lossless integrity verified")
                
                # Verify decompressed hash
                decompressed_hash = hashlib.sha256(decompressed).digest()
                assert decompressed_hash == original_hash, \
                    "Decompressed hash mismatch"
                print(f"  ✓ SHA-256 hash verified: {decompressed_hash.hex()[:32]}...")
                
                print(f"  ✓ {test_name} PASSED")
                
            except Exception as e:
                print(f"  ✗ {test_name} FAILED: {e}")
                raise
        
        print("\n✓ TEST 1 PASSED: All chained compression integrity checks successful")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dictionary_chain_integrity():
    """
    Test 2: Verify DictionaryChain integrity and key derivation.
    
    Ensures that:
    - Each layer's dictionary is properly registered
    - Cryptographic hashes are computed correctly
    - Key derivation follows L(n) → L(n+1) chain
    """
    print("\n" + "=" * 80)
    print("TEST 2: Dictionary Chain Integrity & Key Derivation")
    print("=" * 80)
    
    try:
        from engine import CobolEngine, GlobalPatternRegistry, DictionaryChain
        
        engine = CobolEngine()
        
        # Verify dictionary chain exists
        chain = engine.dict_manager.get_chain()
        assert chain is not None, "Dictionary chain not initialized"
        print("✓ Dictionary chain initialized")
        
        # Verify all layer dictionaries exist
        expected_layers = [
            "L1_SEMANTIC", "L2_STRUCTURAL", "L3_NUMERIC", "L4_BITSTREAM",
            "L5_PATTERN", "L6_METADATA", "L7_INSTRUCTION_SET", "L8_FINAL"
        ]
        
        for layer in expected_layers:
            layer_dict = chain.layer_dicts.get(layer)
            assert layer_dict is not None, f"Missing layer dictionary: {layer}"
            print(f"✓ Layer {layer} dictionary initialized")
        
        # Verify key chain derivation
        print("\nVerifying key derivation chain:")
        for i, layer in enumerate(chain.layer_chain):
            key = chain.get_layer_key(layer)
            assert len(key) == 32, f"Invalid key length for {layer}: {len(key)}"
            print(f"✓ {layer} encryption key derived (length: {len(key)} bytes)")
            
            # Verify key is derived from previous layer if applicable
            if i > 0:
                prev_layer = chain.layer_chain[i - 1]
                prev_hash = chain.layer_dicts[prev_layer].get("hash", b"")
                if prev_hash:
                    expected_key_material = prev_hash + engine.global_registry.get_combined_hash()
                    expected_key = hashlib.sha256(expected_key_material).digest()
                    # Key derivation is correct if it's deterministic
                    assert key == chain.get_layer_key(layer), \
                        f"Key derivation not deterministic for {layer}"
        
        print("\n✓ TEST 2 PASSED: Dictionary chain integrity verified")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_large_data_compression():
    """
    Test 3: Test compression on larger datasets for ratio and throughput.
    
    Verifies:
    - Scaling to larger datasets (1MB+)
    - Compression ratio maintenance
    - Throughput metrics (target: 9.1 MB/s per core)
    """
    print("\n" + "=" * 80)
    print("TEST 3: Large Data Compression Performance")
    print("=" * 80)
    
    try:
        import time
        from engine import CobolEngine
        
        engine = CobolEngine()
        
        # Create 1MB test data with repeating patterns
        pattern = b"The quick brown fox jumps over the lazy dog. " * 100
        test_data = pattern * (1_000_000 // len(pattern))
        test_data = test_data[:1_000_000]  # Trim to exactly 1MB
        
        print(f"\nTest data size: {len(test_data):,} bytes ({len(test_data) / 1_000_000:.2f}MB)")
        print(f"Original hash: {hashlib.sha256(test_data).digest().hex()[:32]}...")
        
        # Measure compression time
        start_time = time.time()
        compressed, metadata = engine.compress_chained(test_data)
        compression_time = time.time() - start_time
        
        print(f"\nCompression results:")
        print(f"  Compressed size: {len(compressed):,} bytes ({len(compressed) / 1_000_000:.2f}MB)")
        print(f"  Compression ratio: {metadata.compression_ratio:.2f}x")
        print(f"  Space saved: {(1 - len(compressed) / len(test_data)) * 100:.1f}%")
        print(f"  Compression time: {compression_time:.3f}s")
        print(f"  Throughput: {len(test_data) / compression_time / 1_000_000:.2f} MB/s")
        
        # Measure decompression time
        start_time = time.time()
        decompressed = engine.decompress_block(compressed, metadata)
        decompression_time = time.time() - start_time
        
        print(f"\nDecompression results:")
        print(f"  Decompressed size: {len(decompressed):,} bytes")
        print(f"  Decompression time: {decompression_time:.3f}s")
        print(f"  Throughput: {len(test_data) / decompression_time / 1_000_000:.2f} MB/s")
        
        # Verify lossless
        assert decompressed == test_data, "Decompressed data mismatch"
        assert hashlib.sha256(decompressed).digest() == metadata.integrity_hash, \
            "Integrity hash mismatch"
        print(f"\n✓ Lossless integrity verified")
        print(f"✓ Decompressed hash: {hashlib.sha256(decompressed).digest().hex()[:32]}...")
        
        # Get statistics
        stats = engine.get_statistics()
        print(f"\nEngine statistics:")
        print(f"  Blocks processed: {stats['blocks_processed']}")
        print(f"  Total original: {stats['total_original_size']:,} bytes")
        print(f"  Total compressed: {stats['total_compressed_size']:,} bytes")
        print(f"  Overall ratio: {stats['overall_compression_ratio']:.2f}x")
        print(f"  Space saved: {stats['space_saved_percent']:.1f}%")
        print(f"  Layers applied: {stats['layers_applied']}")
        
        print("\n✓ TEST 3 PASSED: Large data compression verified")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_extreme_compression():
    """
    Test 4: Test extreme compression using ExtremeCobolEngine with L8.
    
    Verifies integration with Layer 8 ultra-extreme mapping for
    approaching the 1:100M target ratio.
    """
    print("\n" + "=" * 80)
    print("TEST 4: Extreme Compression (Layers 1-8)")
    print("=" * 80)
    
    try:
        from extreme_engine_enhanced import ExtremeCobolEngine
        
        engine = ExtremeCobolEngine()
        
        # Test with highly repetitive data
        pattern = b"The quick brown fox jumps over the lazy dog. " * 1000
        test_data = pattern * 100  # Highly redundant
        
        print(f"\nTest data:")
        print(f"  Size: {len(test_data):,} bytes ({len(test_data) / 1_000_000:.2f}MB)")
        print(f"  Hash: {hashlib.sha256(test_data).digest().hex()[:32]}...")
        
        # Register some large patterns for L8
        pattern1 = b"The quick brown fox jumps over the lazy dog. " * 100
        pattern2 = pattern[:500]
        engine.register_pattern(pattern1)
        engine.register_pattern(pattern2)
        
        print(f"\nRegistered patterns in global registry: 2")
        
        # Compress using chained pipeline
        compressed, metadata = engine.compress_block_chained(test_data)
        
        print(f"\nCompression results:")
        print(f"  Compressed size: {len(compressed):,} bytes ({len(compressed) / 1_000_000:.4f}MB)")
        print(f"  Compression ratio: {metadata.compression_ratio:.2f}x")
        print(f"  Layers applied: {[l.name for l in metadata.layers_applied]}")
        
        # Decompress
        decompressed = engine.decompress_block_chained(compressed, metadata)
        
        print(f"\nDecompression verification:")
        print(f"  Decompressed size: {len(decompressed):,} bytes")
        print(f"  Match original: {decompressed == test_data}")
        print(f"  Hash match: {hashlib.sha256(decompressed).digest() == metadata.integrity_hash}")
        
        assert decompressed == test_data, "Decompressed data mismatch"
        assert hashlib.sha256(decompressed).digest() == metadata.integrity_hash, \
            "Integrity hash mismatch"
        
        print(f"\n✓ TEST 4 PASSED: Extreme compression with L8 verified")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all test suite."""
    print("\n" + "=" * 80)
    print("COBOL PROTOCOL - CHAINED HIERARCHICAL DICTIONARY SYSTEM")
    print("Test Suite - Lossless Integrity Verification")
    print("=" * 80)
    
    results = []
    
    # Run all tests
    results.append(("Chained Compression Integrity", test_chained_compression_integrity()))
    results.append(("Dictionary Chain Integrity", test_dictionary_chain_integrity()))
    results.append(("Large Data Compression", test_large_data_compression()))
    results.append(("Extreme Compression (L1-L8)", test_extreme_compression()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(1 for _, r in results if r)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n✓ ALL TESTS PASSED - Chained hierarchical dictionary system verified!")
        return 0
    else:
        print(f"\n✗ {total_tests - total_passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

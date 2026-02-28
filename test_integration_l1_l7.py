"""
COBOL Protocol v1.2 - Full L1-L7 Integration Tests
End-to-end testing of complete compression pipeline
"""

import time
from typing import Tuple
import os


class IntegrationTestRunner:
    """Runner for L1-L7 integration tests"""
    
    def __init__(self):
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
    
    def run_test(self, name: str, test_func) -> bool:
        """Run individual test"""
        self.total_tests += 1
        try:
            test_func()
            self.passed_tests += 1
            self.test_results.append((name, "PASS", None))
            print(f"✅ {name}")
            return True
        except AssertionError as e:
            self.failed_tests += 1
            self.test_results.append((name, "FAIL", str(e)))
            print(f"❌ {name}: {e}")
            return False
        except Exception as e:
            self.failed_tests += 1
            self.test_results.append((name, "ERROR", str(e)))
            print(f"⚠️ {name}: {type(e).__name__}: {e}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print(f"INTEGRATION TEST SUMMARY")
        print("=" * 70)
        print(f"Total Tests:  {self.total_tests}")
        print(f"Passed:       {self.passed_tests} ✅")
        print(f"Failed:       {self.failed_tests} ❌")
        print(f"Success Rate: {100 * self.passed_tests / max(self.total_tests, 1):.1f}%")
        print("=" * 70)
        
        if self.failed_tests > 0:
            print("\nFailed Tests:")
            for name, status, error in self.test_results:
                if status == "FAIL":
                    print(f"  - {name}: {error}")


def test_l5_basic():
    """Test Layer 5 basic compression"""
    from layer5_optimized import OptimizedLayer5Pipeline
    
    l5 = OptimizedLayer5Pipeline()
    test_data = b"AAABBBCCC" * 100
    
    compressed = l5.compress(test_data)
    decompressed = l5.decompress(compressed)
    
    assert decompressed == test_data, "L5 roundtrip failed"
    assert len(compressed) < len(test_data), "L5 did not compress"


def test_l6_basic():
    """Test Layer 6 basic compression"""
    from layer6_optimized import OptimizedLayer6Pipeline
    
    l6 = OptimizedLayer6Pipeline()
    test_data = b"test data test data " * 100
    
    compressed = l6.compress(test_data)
    decompressed = l6.decompress(compressed)
    
    assert decompressed == test_data, "L6 roundtrip failed"
    assert len(compressed) < len(test_data), "L6 did not compress"


def test_l7_basic():
    """Test Layer 7 basic compression"""
    from layer7_optimized import OptimizedLayer7Pipeline
    
    l7 = OptimizedLayer7Pipeline(method="huffman")
    test_data = b"entropy coding test " * 100
    
    compressed = l7.compress(test_data)
    decompressed = l7.decompress(compressed)
    
    assert decompressed == test_data, "L7 roundtrip failed"


def test_l5_l6_chaining():
    """Test L5 output as L6 input"""
    from layer5_optimized import OptimizedLayer5Pipeline
    from layer6_optimized import OptimizedLayer6Pipeline
    
    l5 = OptimizedLayer5Pipeline()
    l6 = OptimizedLayer6Pipeline()
    
    original = b"chaining test " * 200
    
    # Compress through L5
    l5_compressed = l5.compress(original)
    l5_decompressed = l5.decompress(l5_compressed)
    assert l5_decompressed == original
    
    # Use L5 output as L6 input
    l6_compressed = l6.compress(l5_compressed)
    l6_decompressed = l6.decompress(l6_compressed)
    assert l6_decompressed == l5_compressed
    
    assert len(l6_compressed) <= len(l5_compressed), "L6 did not improve compression"


def test_l6_l7_chaining():
    """Test L6 output as L7 input"""
    from layer6_optimized import OptimizedLayer6Pipeline
    from layer7_optimized import OptimizedLayer7Pipeline
    
    l6 = OptimizedLayer6Pipeline()
    l7 = OptimizedLayer7Pipeline()
    
    original = b"pattern test " * 200
    
    # Compress through L6
    l6_compressed = l6.compress(original)
    
    # Use L6 output as L7 input
    l7_compressed = l7.compress(l6_compressed)
    l7_decompressed = l7.decompress(l7_compressed)
    
    assert l7_decompressed == l6_compressed


def test_full_l5_l6_l7():
    """Test full L5-L6-L7 pipeline"""
    from layer5_optimized import OptimizedLayer5Pipeline
    from layer6_optimized import OptimizedLayer6Pipeline
    from layer7_optimized import OptimizedLayer7Pipeline
    
    l5 = OptimizedLayer5Pipeline()
    l6 = OptimizedLayer6Pipeline()
    l7 = OptimizedLayer7Pipeline(method="huffman")
    
    original = b"Full pipeline test " * 500
    
    # Forward
    l5_comp = l5.compress(original)
    l6_comp = l6.compress(l5_comp)
    l7_comp = l7.compress(l6_comp)
    
    # Backward
    l7_decomp = l7.decompress(l7_comp)
    l6_decomp = l6.decompress(l7_decomp)
    l5_decomp = l5.decompress(l6_decomp)
    
    assert l5_decomp == original, "Full pipeline roundtrip failed"


def test_cobol_data():
    """Test with COBOL program data"""
    from layer5_optimized import OptimizedLayer5Pipeline
    from layer6_optimized import OptimizedLayer6Pipeline
    from layer7_optimized import OptimizedLayer7Pipeline
    
    cobol_program = b"""
    IDENTIFICATION DIVISION.
    PROGRAM-ID. COMPRESSION-TEST.
    
    DATA DIVISION.
    WORKING-STORAGE SECTION.
    01 WS-COUNTER PIC 9(5) VALUE 0.
    01 WS-RESULT PIC X(100).
    
    PROCEDURE DIVISION.
    MAIN-PROCEDURE.
        PERFORM COMPRESS-DATA.
        PERFORM CALCULATE-RATIO.
        STOP RUN.
    
    COMPRESS-DATA.
        MOVE 'Compressing COBOL data' TO WS-RESULT.
    
    CALCULATE-RATIO.
        MOVE 'Compression complete' TO WS-RESULT.
    """ * 10
    
    l5 = OptimizedLayer5Pipeline()
    l6 = OptimizedLayer6Pipeline()
    l7 = OptimizedLayer7Pipeline()
    
    original_size = len(cobol_program)
    
    l5_comp = l5.compress(cobol_program)
    l6_comp = l6.compress(l5_comp)
    l7_comp = l7.compress(l6_comp)
    
    # Verify decompression
    l7_decomp = l7.decompress(l7_comp)
    l6_decomp = l6.decompress(l7_decomp)
    l5_decomp = l5.decompress(l6_decomp)
    
    assert l5_decomp == cobol_program
    
    # Check compression ratio
    final_size = len(l7_comp)
    ratio = original_size / final_size if final_size > 0 else 0
    assert ratio >= 3.0, f"Compression ratio {ratio:.2f}x too low for COBOL"


def test_json_data():
    """Test with JSON data"""
    from layer5_optimized import OptimizedLayer5Pipeline
    from layer6_optimized import OptimizedLayer6Pipeline
    from layer7_optimized import OptimizedLayer7Pipeline
    
    json_data = b"""
    {
        "users": [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
            {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
        ],
        "metadata": {
            "version": "1.0",
            "timestamp": "2026-02-28T00:00:00Z"
        }
    }
    """ * 50
    
    l5 = OptimizedLayer5Pipeline()
    l6 = OptimizedLayer6Pipeline()
    l7 = OptimizedLayer7Pipeline()
    
    l5_comp = l5.compress(json_data)
    l6_comp = l6.compress(l5_comp)
    l7_comp = l7.compress(l6_comp)
    
    # Decompress
    result = l5.decompress(l6.decompress(l7.decompress(l7_comp)))
    assert result == json_data


def test_binary_data():
    """Test with binary data"""
    from layer5_optimized import OptimizedLayer5Pipeline
    from layer6_optimized import OptimizedLayer6Pipeline
    from layer7_optimized import OptimizedLayer7Pipeline
    
    # Create test binary data
    binary_data = bytes(range(256)) * 20
    
    l5 = OptimizedLayer5Pipeline()
    l6 = OptimizedLayer6Pipeline()
    l7 = OptimizedLayer7Pipeline()
    
    l5_comp = l5.compress(binary_data)
    l6_comp = l6.compress(l5_comp)
    l7_comp = l7.compress(l6_comp)
    
    result = l5.decompress(l6.decompress(l7.decompress(l7_comp)))
    assert result == binary_data


def test_large_file():
    """Test with large file (1 MB)"""
    from layer5_optimized import OptimizedLayer5Pipeline
    from layer6_optimized import OptimizedLayer6Pipeline
    from layer7_optimized import OptimizedLayer7Pipeline
    
    # Create 1 MB test data
    test_data = b"Large file compression test " * (1024 * 40)  # ~1 MB
    
    l5 = OptimizedLayer5Pipeline()
    l6 = OptimizedLayer6Pipeline()
    l7 = OptimizedLayer7Pipeline()
    
    start = time.time()
    l5_comp = l5.compress(test_data)
    l6_comp = l6.compress(l5_comp)
    l7_comp = l7.compress(l6_comp)
    compress_time = time.time() - start
    
    start = time.time()
    result = l5.decompress(l6.decompress(l7.decompress(l7_comp)))
    decompress_time = time.time() - start
    
    assert result == test_data
    
    throughput = len(test_data) / compress_time / 1024 / 1024 if compress_time > 0 else 0
    ratio = len(test_data) / len(l7_comp)
    
    print(f"    Large file: {len(test_data) / 1024 / 1024:.1f}MB → {len(l7_comp) / 1024 / 1024:.1f}MB")
    print(f"    Ratio: {ratio:.1f}x, Throughput: {throughput:.1f} MB/s")


def test_small_data():
    """Test with very small data"""
    from layer5_optimized import OptimizedLayer5Pipeline
    from layer6_optimized import OptimizedLayer6Pipeline
    from layer7_optimized import OptimizedLayer7Pipeline
    
    l5 = OptimizedLayer5Pipeline()
    l6 = OptimizedLayer6Pipeline()
    l7 = OptimizedLayer7Pipeline()
    
    # Edge cases
    test_cases = [
        b"",  # Empty
        b"A",  # Single byte
        b"AB",  # Two bytes
        b"AAA",  # All same
    ]
    
    for test_data in test_cases:
        l5_comp = l5.compress(test_data)
        l6_comp = l6.compress(l5_comp)
        l7_comp = l7.compress(l6_comp)
        
        result = l5.decompress(l6.decompress(l7.decompress(l7_comp)))
        assert result == test_data, f"Failed for {len(test_data)} byte(s)"


def test_compression_efficiency():
    """Test compression efficiency metrics"""
    from layer5_optimized import OptimizedLayer5Pipeline
    from layer6_optimized import OptimizedLayer6Pipeline
    from layer7_optimized import OptimizedLayer7Pipeline
    
    l5 = OptimizedLayer5Pipeline()
    l6 = OptimizedLayer6Pipeline()
    l7 = OptimizedLayer7Pipeline()
    
    test_data = b"COBOL PROTOCOL v1.2 TESTING DATA " * 300
    
    # L5 statistics
    l5_comp = l5.compress(test_data)
    l5_stats = l5.get_statistics()
    assert l5_stats['compression_ratio'] > 1.0
    assert l5_stats['throughput_mbps'] > 10
    
    # L6 statistics
    l6_comp = l6.compress(l5_comp)
    l6_stats = l6.get_statistics()
    assert l6_stats['compression_ratio'] > 1.0
    
    # L7 statistics
    l7_comp = l7.compress(l6_comp)
    l7_stats = l7.get_statistics()
    
    # Verify decompression
    result = l5.decompress(l6.decompress(l7.decompress(l7_comp)))
    assert result == test_data


def test_sequential_integrity():
    """Test data integrity through sequential pipeline"""
    from layer5_optimized import OptimizedLayer5Pipeline
    from layer6_optimized import OptimizedLayer6Pipeline
    from layer7_optimized import OptimizedLayer7Pipeline
    
    l5 = OptimizedLayer5Pipeline()
    l6 = OptimizedLayer6Pipeline()
    l7 = OptimizedLayer7Pipeline()
    
    # Test data with specific pattern
    test_data = b"\x00" * 100 + b"\xFF" * 100 + b"ABCABC" * 50
    
    # Compress
    comp1 = l5.compress(test_data)
    comp2 = l6.compress(comp1)
    comp3 = l7.compress(comp2)
    
    # Decompress in reverse order
    decomp2 = l7.decompress(comp3)
    decomp1 = l6.decompress(decomp2)
    decomp0 = l5.decompress(decomp1)
    
    assert decomp0 == test_data


def test_pipeline_statistics():
    """Test complete pipeline statistics"""
    from layer5_optimized import OptimizedLayer5Pipeline
    from layer6_optimized import OptimizedLayer6Pipeline
    from layer7_optimized import OptimizedLayer7Pipeline
    
    l5 = OptimizedLayer5Pipeline()
    l6 = OptimizedLayer6Pipeline()
    l7 = OptimizedLayer7Pipeline()
    
    test_data = b"STATISTICS TEST DATA " * 200
    
    original_size = len(test_data)
    
    l5_comp = l5.compress(test_data)
    after_l5 = len(l5_comp)
    
    l6_comp = l6.compress(l5_comp)
    after_l6 = len(l6_comp)
    
    l7_comp = l7.compress(l6_comp)
    after_l7 = len(l7_comp)
    
    print(f"    L1-L4: {original_size} bytes (baseline)")
    print(f"    After L5: {after_l5} bytes ({original_size/after_l5:.1f}x)")
    print(f"    After L6: {after_l6} bytes ({original_size/after_l6:.1f}x)")
    print(f"    After L7: {after_l7} bytes ({original_size/after_l7:.1f}x)")
    
    # Verify each layer reduces size
    assert after_l5 < original_size
    assert after_l6 < after_l5
    assert after_l7 < after_l6


if __name__ == "__main__":
    print("=" * 70)
    print("COBOL PROTOCOL v1.2 - LAYER 5-7 INTEGRATION TESTS")
    print("=" * 70 + "\n")
    
    runner = IntegrationTestRunner()
    
    # Run all tests
    print("Single Layer Tests:")
    runner.run_test("Layer 5 Basic Compression", test_l5_basic)
    runner.run_test("Layer 6 Basic Compression", test_l6_basic)
    runner.run_test("Layer 7 Basic Compression", test_l7_basic)
    
    print("\nLayer Chaining Tests:")
    runner.run_test("L5 → L6 Chaining", test_l5_l6_chaining)
    runner.run_test("L6 → L7 Chaining", test_l6_l7_chaining)
    runner.run_test("Full L5 → L6 → L7 Pipeline", test_full_l5_l6_l7)
    
    print("\nData Type Tests:")
    runner.run_test("COBOL Program Data", test_cobol_data)
    runner.run_test("JSON Data", test_json_data)
    runner.run_test("Binary Data", test_binary_data)
    
    print("\nSize/Scale Tests:")
    runner.run_test("Small Data Edge Cases", test_small_data)
    runner.run_test("Large File (1 MB)", test_large_file)
    
    print("\nValidation Tests:")
    runner.run_test("Compression Efficiency", test_compression_efficiency)
    runner.run_test("Sequential Integrity", test_sequential_integrity)
    runner.run_test("Pipeline Statistics", test_pipeline_statistics)
    
    # Print summary
    runner.print_summary()

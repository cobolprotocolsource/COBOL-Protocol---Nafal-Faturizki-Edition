"""
COBOL Protocol v1.2 - Layer 5-7 Comprehensive Test Suite
Integration tests, performance benchmarks, correctness verification
"""

import pytest
import time
from typing import Tuple
from layer5_optimized import OptimizedLayer5Pipeline
from layer6_optimized import OptimizedLayer6Pipeline
from layer7_optimized import OptimizedLayer7Pipeline


class TestLayer5Compression:
    """Layer 5 (Advanced RLE) tests"""
    
    def test_basic_rle_compression(self):
        """Test basic RLE compression"""
        pipeline = OptimizedLayer5Pipeline()
        test_data = b"AAABBBCCCC"
        
        compressed = pipeline.compress(test_data)
        decompressed = pipeline.decompress(compressed)
        
        assert decompressed == test_data, "Roundtrip failed"
        assert len(compressed) < len(test_data), "No compression achieved"
    
    def test_pattern_catalog(self):
        """Test pattern catalog functionality"""
        pipeline = OptimizedLayer5Pipeline()
        test_data = b"Hello World! Hello World! " * 100
        
        # Analyze patterns
        patterns = pipeline.encoder.analyze_patterns(test_data)
        assert len(patterns) > 0, "No patterns found"
        assert patterns[0].roi > 0, "Pattern has no ROI"
    
    def test_compressibility(self):
        """Test pattern-rich data"""
        pipeline = OptimizedLayer5Pipeline()
        # Highly compressible data
        test_data = b"ABC" * 1000
        
        compressed = pipeline.compress(test_data)
        decompressed = pipeline.decompress(compressed)
        
        assert decompressed == test_data
        ratio = len(test_data) / len(compressed)
        assert ratio >= 1.5, f"Compression ratio {ratio:.2f}x too low"
    
    def test_random_data(self):
        """Test incompressible data"""
        import os
        pipeline = OptimizedLayer5Pipeline()
        test_data = os.urandom(512)
        
        compressed = pipeline.compress(test_data)
        decompressed = pipeline.decompress(compressed)
        
        assert decompressed == test_data
        # Random data may expand, that's ok
    
    def test_empty_data(self):
        """Test empty input"""
        pipeline = OptimizedLayer5Pipeline()
        test_data = b""
        
        compressed = pipeline.compress(test_data)
        decompressed = pipeline.decompress(compressed)
        
        assert decompressed == test_data
    
    def test_large_data(self):
        """Test large data (10 MB)"""
        pipeline = OptimizedLayer5Pipeline()
        test_data = b"COBOL" * (1000000)  # ~5 MB
        
        start = time.time()
        compressed = pipeline.compress(test_data)
        compress_time = time.time() - start
        
        start = time.time()
        decompressed = pipeline.decompress(compressed)
        decompress_time = time.time() - start
        
        assert decompressed == test_data
        
        # Calculate throughput
        throughput = len(test_data) / compress_time / 1024 / 1024
        print(f"L5 compression throughput: {throughput:.1f} MB/s")
        assert throughput > 10, f"Throughput {throughput:.1f} MB/s too low"
    
    def test_statistics(self):
        """Test compression statistics"""
        pipeline = OptimizedLayer5Pipeline()
        test_data = b"Test data " * 1000
        
        compressed = pipeline.compress(test_data)
        stats = pipeline.get_statistics()
        
        assert stats['input_bytes'] == len(test_data)
        assert stats['output_bytes'] == len(compressed)
        assert stats['compression_ratio'] > 0
        assert stats['throughput_mbps'] > 0
    
    def test_pattern_priority(self):
        """Test pattern selection by ROI"""
        pipeline = OptimizedLayer5Pipeline()
        test_data = (
            b"COMMON_PATTERN_HERE_" * 500 +
            b"rare" * 2
        )
        
        patterns = pipeline.encoder.analyze_patterns(test_data)
        common_pattern = b"COMMON_PATTERN_HERE_"
        
        # Find common pattern in list
        found = False
        for p in patterns:
            if p.pattern == common_pattern:
                found = True
                assert p.roi > 0
                break
        assert found, "Common pattern not in top patterns"


class TestLayer6PatternDetection:
    """Layer 6 (Pattern Detection) tests"""
    
    def test_basic_pattern_detection(self):
        """Test pattern detection"""
        pipeline = OptimizedLayer6Pipeline()
        test_data = b"The quick brown fox " * 100
        
        compressed = pipeline.compress(test_data)
        decompressed = pipeline.decompress(compressed)
        
        assert decompressed == test_data, "Roundtrip failed"
    
    def test_trie_dictionary(self):
        """Test Trie dictionary operations"""
        pipeline = OptimizedLayer6Pipeline()
        
        # Add patterns
        pattern1 = b"hello"
        pattern2 = b"world"
        
        id1 = pipeline.dictionary.add_pattern(pattern1)
        id2 = pipeline.dictionary.add_pattern(pattern2)
        
        assert id1 >= 0
        assert id2 >= 0
        assert id1 != id2
        
        # Retrieve patterns
        assert pipeline.dictionary.get_pattern(id1) == pattern1
        assert pipeline.dictionary.get_pattern(id2) == pattern2
    
    def test_pattern_scorer(self):
        """Test pattern scoring"""
        pipeline = OptimizedLayer6Pipeline()
        test_data = b"test test test other other" * 100
        
        scored = pipeline.detector.score_patterns(test_data)
        assert len(scored) > 0
        assert scored[0][1] > scored[-1][1]  # Sorted by score
    
    def test_state_machine_tokenizer(self):
        """Test tokenizer performance"""
        pipeline = OptimizedLayer6Pipeline()
        test_data = b"ABCABCABC" * 100
        
        pipeline.dictionary.add_pattern(b"ABC")
        tokens = pipeline.encoder.tokenizer.tokenize(test_data)
        
        assert len(tokens) > 0
        # Detokenize should match original
        detokenized = pipeline.encoder.tokenizer.detokenize(tokens)
        assert detokenized == test_data
    
    def test_dictionary_serialization(self):
        """Test dictionary save/load"""
        pipeline1 = OptimizedLayer6Pipeline()
        pattern = b"test_pattern"
        id1 = pipeline1.dictionary.add_pattern(pattern)
        
        # Serialize
        dict_bytes = pipeline1.dictionary.to_bytes()
        
        # Load into new instance
        from layer6_optimized import StructuralPatternDictionary
        loaded = StructuralPatternDictionary.from_bytes(dict_bytes)
        
        assert loaded.get_pattern(id1) == pattern
    
    def test_compression_with_dictionary(self):
        """Test full compression with pattern dictionary"""
        pipeline = OptimizedLayer6Pipeline()
        test_data = (
            b"BEGIN COBOL PROGRAM. " +
            b"PERFORM UNTIL DONE. " * 500 +
            b"END PROGRAM."
        )
        
        compressed = pipeline.compress(test_data)
        decompressed = pipeline.decompress(compressed)
        
        assert decompressed == test_data
        ratio = len(test_data) / len(compressed)
        print(f"L6 compression ratio: {ratio:.2f}x")
    
    def test_large_file_l6(self):
        """Test 5 MB file"""
        pipeline = OptimizedLayer6Pipeline()
        test_data = b"Pattern matching test " * 100000  # ~2.2 MB
        
        start = time.time()
        compressed = pipeline.compress(test_data)
        compress_time = time.time() - start
        
        decompressed = pipeline.decompress(compressed)
        assert decompressed == test_data
        
        throughput = len(test_data) / compress_time / 1024 / 1024
        print(f"L6 compression throughput: {throughput:.1f} MB/s")


class TestLayer7EntropyCoding:
    """Layer 7 (Entropy Coding) tests"""
    
    def test_huffman_basic(self):
        """Test basic Huffman coding"""
        pipeline = OptimizedLayer7Pipeline(method="huffman")
        test_data = b"AAABBBCCCC"
        
        compressed = pipeline.compress(test_data)
        decompressed = pipeline.decompress(compressed)
        
        assert decompressed == test_data
    
    def test_frequency_analysis(self):
        """Test frequency analyzer"""
        analyzer = pipeline.analyzer
        pipeline = OptimizedLayer7Pipeline()
        test_data = b"AAABBBCCCC"
        
        freq = pipeline.analyzer.analyze(test_data)
        assert freq[ord('A')] == 3
        assert freq[ord('B')] == 3
        assert freq[ord('C')] == 4
    
    def test_entropy_calculation(self):
        """Test entropy calculation"""
        pipeline = OptimizedLayer7Pipeline()
        
        # Low entropy (highly patterned)
        low_entropy_data = b"AAAAAABBBB"
        entropy1 = pipeline.analyzer.entropy(low_entropy_data)
        
        # High entropy (random-like)
        import os
        high_entropy_data = os.urandom(256)
        entropy2 = pipeline.analyzer.entropy(high_entropy_data)
        
        assert entropy1 < entropy2, "Pattern has more entropy than random"
    
    def test_huffman_compression(self):
        """Test Huffman with text"""
        pipeline = OptimizedLayer7Pipeline(method="huffman")
        text = "the quick brown fox jumps over the lazy dog " * 100
        test_data = text.encode()
        
        compressed = pipeline.compress(test_data)
        decompressed = pipeline.decompress(compressed)
        
        assert decompressed == test_data
        ratio = len(test_data) / len(compressed)
        print(f"L7 Huffman ratio: {ratio:.2f}x")
    
    def test_optional_layer_skip(self):
        """Test optional L7 skip"""
        pipeline = OptimizedLayer7Pipeline(method="huffman", optional=True)
        
        # Already compressed data (high entropy)
        import os
        test_data = os.urandom(1024)
        
        compressed = pipeline.compress(test_data)
        stats = pipeline.get_statistics()
        
        # Should skip L7 for incompressible data
        decompressed = pipeline.decompress(compressed)
        assert decompressed == test_data
    
    def test_arithmetic_coding(self):
        """Test arithmetic coding"""
        pipeline = OptimizedLayer7Pipeline(method="arithmetic")
        test_data = b"test data test data" * 50
        
        compressed = pipeline.compress(test_data)
        decompressed = pipeline.decompress(compressed)
        
        assert decompressed == test_data
    
    def test_empty_input_l7(self):
        """Test empty input for L7"""
        pipeline = OptimizedLayer7Pipeline()
        test_data = b""
        
        compressed = pipeline.compress(test_data)
        decompressed = pipeline.decompress(compressed)
        
        assert decompressed == test_data
    
    def test_streaming_encoder(self):
        """Test streaming entropy encoder"""
        from layer7_optimized import StreamingEntropyEncoder
        encoder = StreamingEntropyEncoder(chunk_size=1024)
        
        test_data = b"Stream test data " * 500
        
        encoded = encoder.encode_streaming(test_data)
        decoded = encoder.decode_streaming(encoded)
        
        assert decoded == test_data


class TestIntegrationL5L6L7:
    """Integration tests for L5-L6-L7"""
    
    def test_l5_to_l6(self):
        """Test L5 output as L6 input"""
        l5 = OptimizedLayer5Pipeline()
        l6 = OptimizedLayer6Pipeline()
        
        original = b"test data test data test " * 100
        
        # Compress through L5
        l5_compressed = l5.compress(original)
        
        # Decompress L5
        l5_decompressed = l5.decompress(l5_compressed)
        assert l5_decompressed == original
        
        # Compress L5 output through L6
        l6_compressed = l6.compress(l5_compressed)
        l6_decompressed = l6.decompress(l6_compressed)
        assert l6_decompressed == l5_compressed
    
    def test_l6_to_l7(self):
        """Test L6 output as L7 input"""
        l6 = OptimizedLayer6Pipeline()
        l7 = OptimizedLayer7Pipeline()
        
        original = b"integration test " * 200
        
        l6_compressed = l6.compress(original)
        l7_compressed = l7.compress(l6_compressed)
        l7_decompressed = l7.decompress(l7_compressed)
        
        assert l7_decompressed == l6_compressed
    
    def test_full_l5_l6_l7_pipeline(self):
        """Test full L5-L6-L7 pipeline"""
        l5 = OptimizedLayer5Pipeline()
        l6 = OptimizedLayer6Pipeline()
        l7 = OptimizedLayer7Pipeline(method="huffman")
        
        original = b"COBOL PROTOCOL v1.2 " * 500
        
        # L5: RLE compression
        l5_compressed = l5.compress(original)
        
        # L6: Pattern detection
        l6_compressed = l6.compress(l5_compressed)
        
        # L7: Entropy coding
        l7_compressed = l7.compress(l6_compressed)
        
        # Full compression
        full_size = len(l7_compressed)
        original_size = len(original)
        ratio = original_size / full_size
        
        print(f"\nFull L5-L6-L7 compression: {ratio:.2f}x")
        print(f"Original: {original_size} bytes")
        print(f"Compressed: {full_size} bytes")
        
        # Decompress in reverse
        l7_decompressed = l7.decompress(l7_compressed)
        l6_decompressed = l6.decompress(l7_decompressed)
        l5_decompressed = l5.decompress(l6_decompressed)
        
        assert l5_decompressed == original, "Full decompression failed"
    
    def test_cobol_specific_data(self):
        """Test with COBOL-like data"""
        cobol_program = b"""
        IDENTIFICATION DIVISION.
        PROGRAM-ID. COMPRESSION-PROGRAM.
        
        DATA DIVISION.
        WORKING-STORAGE SECTION.
        01 WS-COUNTER PIC 9(5) VALUE 0.
        
        PROCEDURE DIVISION.
        MAIN-PROCEDURE.
            PERFORM TIMED-COMPRESSION.
            STOP RUN.
        """ * 50
        
        l5 = OptimizedLayer5Pipeline()
        l6 = OptimizedLayer6Pipeline()
        l7 = OptimizedLayer7Pipeline()
        
        compressed = l7.compress(l6.compress(l5.compress(cobol_program)))
        decompressed = l5.decompress(l6.decompress(l7.decompress(compressed)))
        
        assert decompressed == cobol_program
        ratio = len(cobol_program) / len(compressed)
        print(f"COBOL data compression: {ratio:.2f}x")
        assert ratio > 2.0, "COBOL compression below 2x"
    
    def test_throughput_measurement(self):
        """Measure combined throughput"""
        l5 = OptimizedLayer5Pipeline()
        l6 = OptimizedLayer6Pipeline()
        l7 = OptimizedLayer7Pipeline()
        
        test_data = b"performance test " * 50000  # ~800 KB
        
        start = time.time()
        step1 = l5.compress(test_data)
        step2 = l6.compress(step1)
        step3 = l7.compress(step2)
        total_time = time.time() - start
        
        throughput = len(test_data) / total_time / 1024 / 1024
        print(f"Combined L5-L6-L7 throughput: {throughput:.1f} MB/s")
        
        # Verify decompression
        verify1 = l7.decompress(step3)
        verify2 = l6.decompress(verify1)
        verify3 = l5.decompress(verify2)
        
        assert verify3 == test_data


class TestPerformanceBenchmarks:
    """Performance benchmarks"""
    
    def test_l5_benchmark(self):
        """L5 performance benchmark"""
        pipeline = OptimizedLayer5Pipeline()
        data_mb = 10
        test_data = b"ABC" * (1024 * 1024 * data_mb // 3)
        
        start = time.time()
        compressed = pipeline.compress(test_data)
        elapsed = time.time() - start
        
        throughput = data_mb / elapsed
        print(f"\nL5 Benchmark: {throughput:.1f} MB/s (Target: 100-150 MB/s)")
    
    def test_l6_benchmark(self):
        """L6 performance benchmark"""
        pipeline = OptimizedLayer6Pipeline()
        data_mb = 5
        test_data = b"Pattern test data " * (1024 * 1024 // 17)
        
        start = time.time()
        compressed = pipeline.compress(test_data)
        elapsed = time.time() - start
        
        throughput = len(test_data) / elapsed / 1024 / 1024
        print(f"L6 Benchmark: {throughput:.1f} MB/s (Target: 50-100 MB/s)")
    
    def test_l7_benchmark(self):
        """L7 performance benchmark"""
        pipeline = OptimizedLayer7Pipeline()
        test_data = b"entropy test data " * (100000)
        
        start = time.time()
        compressed = pipeline.compress(test_data)
        elapsed = time.time() - start
        
        throughput = len(test_data) / elapsed / 1024 / 1024
        print(f"L7 Benchmark: {throughput:.1f} MB/s (Target: 20-50 MB/s)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

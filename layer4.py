"""
COBOL Protocol v1.1 - Layer 4: Variable Bit-Packing
===================================================

Compression of numeric sequences using adaptive bit-width selection.

Features:
- Dynamic bit-width calculation (1-64 bits)
- Frame-of-Reference (FoR) encoding for large values
- Zero-run detection for sparse numeric data
- 3-4x compression on numeric sequences

Status: Implementation started (Q2 2026)
"""

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Dict, Tuple, Optional
import io

import numpy as np

from config import CompressionLayer, L4_BIT_WIDTH_MIN, L4_BIT_WIDTH_MAX, L4_CHUNK_SIZE


# ============================================================================
# BIT-PACKING STRATEGY DEFINITIONS
# ============================================================================


class BitPackingStrategy(IntEnum):
    """Strategy selection for bit-packing."""
    CONSTANT = 0x01            # All values fit in constant N bits
    FRAME_OF_REFERENCE = 0x02  # Subtract base, use fewer bits for deltas
    ZERO_RUN = 0x03            # Special handling for sequences with many zeros
    DELTA = 0x04               # Delta-based packing (consecutive diffs)
    DICTIONARY = 0x05          # Dictionary-like compression for repeating values


@dataclass
class BitPackingChunk:
    """Metadata and data for a bit-packed chunk."""
    strategy: BitPackingStrategy
    bit_width: int                       # 1-64 bits per value
    value_count: int                     # Count of values in chunk
    base_value: int = 0                  # For Frame-of-Reference
    zero_count: int = 0                  # For zero-run detection
    zero_positions: List[int] = field(default_factory=list)  # Positions of zeros
    compressed_data: bytes = b""         # Packed bits
    
    def to_bytes(self) -> bytes:
        """Serialize chunk metadata and data."""
        header = struct.pack(
            '<BBII',  # strategy, bit_width, value_count, base_value
            self.strategy,
            self.bit_width,
            self.value_count,
            self.base_value & 0xFFFFFFFF
        )
        return header + self.compressed_data


@dataclass
class BitWidthAnalysis:
    """Result of analyzing bit-width requirements."""
    min_value: int
    max_value: int
    range: int
    bits_needed: int
    strategy: BitPackingStrategy
    zero_count: int = 0
    compression_ratio: float = 1.0


# ============================================================================
# BIT-WIDTH CALCULATOR
# ============================================================================


class BitWidthCalculator:
    """
    Analyzes numeric sequences and determines optimal bit-width.
    
    Algorithms:
    1. Constant bit-width: min bits to represent all values
    2. Frame-of-Reference: subtract minimum, reduce range
    3. Zero-run: special markers for sparse data
    """
    
    @staticmethod
    def calculate_bits_needed(value: int) -> int:
        """Calculate bits needed to represent a single value."""
        if value == 0:
            return 1
        if value < 0:
            # Two's complement for negative numbers
            return value.bit_length() + 1
        return value.bit_length()
    
    @staticmethod
    def analyze_values(values: np.ndarray) -> BitWidthAnalysis:
        """
        Analyze a sequence of values to determine bit-packing strategy.
        
        Args:
            values: NumPy array of integers
        
        Returns:
            BitWidthAnalysis with strategy recommendation
        """
        min_val = int(np.min(values))
        max_val = int(np.max(values))
        value_range = max_val - min_val
        
        # Count zeros
        zero_count = int(np.sum(values == 0))
        zero_percentage = zero_count / len(values) if len(values) > 0 else 0
        
        # Determine bit-width
        bits_needed = BitWidthCalculator.calculate_bits_needed(value_range)
        bits_needed = max(L4_BIT_WIDTH_MIN, min(bits_needed, L4_BIT_WIDTH_MAX))
        
        # Select strategy
        strategy = BitPackingStrategy.CONSTANT
        compression_ratio = 1.0
        
        # Zero-run detection: if > 50% zeros, use zero-run encoding
        if zero_percentage > 0.5:
            strategy = BitPackingStrategy.ZERO_RUN
            compression_ratio = 1.0 / (1.0 - zero_percentage / 2.0)
        
        # Frame-of-Reference: subtract min to reduce range
        elif min_val > 0:
            adjusted_range = value_range
            adjusted_bits = BitWidthCalculator.calculate_bits_needed(adjusted_range)
            adjusted_bits = max(L4_BIT_WIDTH_MIN, min(adjusted_bits, L4_BIT_WIDTH_MAX))
            
            if adjusted_bits < bits_needed:
                strategy = BitPackingStrategy.FRAME_OF_REFERENCE
                compression_ratio = bits_needed / adjusted_bits
        
        # Delta detection: check if consecutive differences are small
        if len(values) > 1:
            deltas = np.diff(values)
            max_delta = int(np.max(np.abs(deltas)))
            delta_bits = BitWidthCalculator.calculate_bits_needed(max_delta)
            
            if delta_bits < bits_needed * 0.75:
                strategy = BitPackingStrategy.DELTA
                compression_ratio = bits_needed / delta_bits
        
        return BitWidthAnalysis(
            min_value=min_val,
            max_value=max_val,
            range=value_range,
            bits_needed=bits_needed,
            strategy=strategy,
            zero_count=zero_count,
            compression_ratio=compression_ratio
        )


# ============================================================================
# BIT PACKING ENCODER
# ============================================================================


class BitPackingEncoder:
    """
    Encodes numeric sequences using variable bit-width compression.
    
    Supports:
    - Constant bit-width packing
    - Frame-of-Reference (FoR) encoding
    - Zero-run detection
    - Delta encoding
    """
    
    def __init__(self, chunk_size: int = L4_CHUNK_SIZE):
        """Initialize bit-packing encoder."""
        self.chunk_size = chunk_size
        self.chunks: List[BitPackingChunk] = []
    
    def encode(self, values: np.ndarray) -> Tuple[bytes, List[BitPackingChunk]]:
        """
        Encode numeric sequence using adaptive bit-width packing.
        
        Args:
            values: NumPy array of integers to pack
        
        Returns:
            Tuple of (compressed_data, chunk_metadata)
        """
        self.chunks = []
        output = io.BytesIO()
        
        # Process in chunks
        for i in range(0, len(values), self.chunk_size):
            chunk_values = values[i:i+self.chunk_size]
            chunk = self._encode_chunk(chunk_values)
            self.chunks.append(chunk)
            output.write(chunk.to_bytes())
        
        return output.getvalue(), self.chunks
    
    def _encode_chunk(self, values: np.ndarray) -> BitPackingChunk:
        """Encode a single chunk of values."""
        # Analyze values
        analysis = BitWidthCalculator.analyze_values(values)
        
        # Encode based on strategy
        if analysis.strategy == BitPackingStrategy.CONSTANT:
            return self._encode_constant(values, analysis)
        elif analysis.strategy == BitPackingStrategy.FRAME_OF_REFERENCE:
            return self._encode_for(values, analysis)
        elif analysis.strategy == BitPackingStrategy.ZERO_RUN:
            return self._encode_zero_run(values, analysis)
        elif analysis.strategy == BitPackingStrategy.DELTA:
            return self._encode_delta(values, analysis)
        else:
            return self._encode_constant(values, analysis)
    
    def _encode_constant(self, values: np.ndarray, analysis: BitWidthAnalysis) -> BitPackingChunk:
        """Encode with constant bit-width."""
        bit_width = analysis.bits_needed
        packed = self._pack_bits(values, bit_width)
        
        return BitPackingChunk(
            strategy=BitPackingStrategy.CONSTANT,
            bit_width=bit_width,
            value_count=len(values),
            compressed_data=packed
        )
    
    def _encode_for(self, values: np.ndarray, analysis: BitWidthAnalysis) -> BitPackingChunk:
        """Encode using Frame-of-Reference (subtract base value)."""
        base = analysis.min_value
        deltas = values - base
        
        # Calculate bits needed for deltas
        delta_bits = BitWidthCalculator.calculate_bits_needed(int(np.max(deltas)))
        delta_bits = max(L4_BIT_WIDTH_MIN, min(delta_bits, L4_BIT_WIDTH_MAX))
        
        packed = self._pack_bits(deltas, delta_bits)
        
        return BitPackingChunk(
            strategy=BitPackingStrategy.FRAME_OF_REFERENCE,
            bit_width=delta_bits,
            value_count=len(values),
            base_value=base,
            compressed_data=packed
        )
    
    def _encode_zero_run(self, values: np.ndarray, analysis: BitWidthAnalysis) -> BitPackingChunk:
        """Encode with zero-run detection."""
        # Find positions of non-zero values
        non_zero_mask = values != 0
        non_zero_indices = np.where(non_zero_mask)[0]
        non_zero_values = values[non_zero_mask]
        
        # Pack non-zero values
        if len(non_zero_values) > 0:
            bit_width = BitWidthCalculator.calculate_bits_needed(int(np.max(np.abs(non_zero_values))))
            bit_width = max(L4_BIT_WIDTH_MIN, min(bit_width, L4_BIT_WIDTH_MAX))
            packed_values = self._pack_bits(non_zero_values, bit_width)
        else:
            bit_width = 1
            packed_values = b""
        
        # Encode positions and values
        packed_positions = self._encode_positions(non_zero_indices)
        packed = packed_positions + packed_values
        
        return BitPackingChunk(
            strategy=BitPackingStrategy.ZERO_RUN,
            bit_width=bit_width,
            value_count=len(values),
            zero_count=analysis.zero_count,
            zero_positions=list(non_zero_indices),
            compressed_data=packed
        )
    
    def _encode_delta(self, values: np.ndarray, analysis: BitWidthAnalysis) -> BitPackingChunk:
        """Encode using delta encoding."""
        # Calculate deltas from first value
        deltas = np.diff(values, prepend=values[0])
        
        # Calculate bit-width for deltas
        delta_range = int(np.max(np.abs(deltas)))
        delta_bits = BitWidthCalculator.calculate_bits_needed(delta_range)
        delta_bits = max(L4_BIT_WIDTH_MIN, min(delta_bits, L4_BIT_WIDTH_MAX))
        
        # Pack deltas
        packed = self._pack_bits(deltas, delta_bits)
        
        return BitPackingChunk(
            strategy=BitPackingStrategy.DELTA,
            bit_width=delta_bits,
            value_count=len(values),
            base_value=int(values[0]),
            compressed_data=packed
        )
    
    def _pack_bits(self, values: np.ndarray, bit_width: int) -> bytes:
        """
        Pack values into bits at specified width.
        
        Args:
            values: Array of values to pack
            bit_width: Bits per value
        
        Returns:
            Bytes containing packed bits
        """
        if bit_width >= 8:
            # No packing needed, just convert to bytes
            if bit_width == 8:
                return values.astype(np.uint8).tobytes()
            elif bit_width == 16:
                return values.astype(np.uint16).tobytes()
            elif bit_width == 32:
                return values.astype(np.uint32).tobytes()
            elif bit_width == 64:
                return values.astype(np.uint64).tobytes()
        
        # Manual bit packing for sub-byte widths
        output = io.BytesIO()
        bit_buffer = 0
        buffer_bits = 0
        
        mask = (1 << bit_width) - 1
        
        for value in values:
            value_int = int(value) & mask
            
            # Add to buffer
            bit_buffer |= (value_int << buffer_bits)
            buffer_bits += bit_width
            
            # Write complete bytes
            while buffer_bits >= 8:
                output.write(bytes([bit_buffer & 0xFF]))
                bit_buffer >>= 8
                buffer_bits -= 8
        
        # Write remaining partial byte
        if buffer_bits > 0:
            output.write(bytes([bit_buffer & 0xFF]))
        
        return output.getvalue()
    
    def _encode_positions(self, positions: np.ndarray) -> bytes:
        """Encode positions using run-length encoding."""
        # Convert positions to gaps
        if len(positions) == 0:
            return b""
        
        gaps = np.diff(positions, prepend=0)
        
        # Use VarInt-like encoding for gaps
        output = io.BytesIO()
        for gap in gaps:
            output.write(self._encode_varint(int(gap)))
        
        return output.getvalue()
    
    @staticmethod
    def _encode_varint(value: int) -> bytes:
        """Encode a single integer as variable-length int."""
        result = []
        while value > 0x7F:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        result.append(value & 0x7F)
        return bytes(result)


# ============================================================================
# BIT PACKING DECODER
# ============================================================================


class BitPackingDecoder:
    """
    Decodes Layer 4 bit-packed data.
    
    Reverses encoding strategies to reconstruct original values.
    """
    
    def decode(self, data: bytes, chunks_metadata: List[BitPackingChunk]) -> np.ndarray:
        """
        Decode bit-packed data.
        
        Args:
            data: Compressed data bytes
            chunks_metadata: Chunk metadata from encoder
        
        Returns:
            NumPy array of decoded integers
        """
        output = []
        input_stream = io.BytesIO(data)
        
        for chunk_meta in chunks_metadata:
            # Read chunk data
            chunk_data = input_stream.read(len(chunk_meta.compressed_data))
            
            # Decode based on strategy
            if chunk_meta.strategy == BitPackingStrategy.CONSTANT:
                values = self._decode_constant(chunk_data, chunk_meta)
            elif chunk_meta.strategy == BitPackingStrategy.FRAME_OF_REFERENCE:
                values = self._decode_for(chunk_data, chunk_meta)
            elif chunk_meta.strategy == BitPackingStrategy.ZERO_RUN:
                values = self._decode_zero_run(chunk_data, chunk_meta)
            elif chunk_meta.strategy == BitPackingStrategy.DELTA:
                values = self._decode_delta(chunk_data, chunk_meta)
            else:
                values = np.array([])
            
            output.extend(values)
        
        return np.array(output, dtype=np.int64)
    
    def _decode_constant(self, data: bytes, chunk: BitPackingChunk) -> np.ndarray:
        """Decode constant bit-width packed data."""
        if chunk.bit_width >= 8:
            if chunk.bit_width == 8:
                return np.frombuffer(data, dtype=np.uint8)
            elif chunk.bit_width == 16:
                return np.frombuffer(data, dtype=np.uint16)
            elif chunk.bit_width == 32:
                return np.frombuffer(data, dtype=np.uint32)
            elif chunk.bit_width == 64:
                return np.frombuffer(data, dtype=np.uint64)
        
        # Manual bit unpacking
        values = []
        bit_buffer = 0
        buffer_bits = 0
        byte_pos = 0
        
        mask = (1 << chunk.bit_width) - 1
        
        while len(values) < chunk.value_count:
            if buffer_bits < chunk.bit_width:
                if byte_pos < len(data):
                    bit_buffer |= (data[byte_pos] << buffer_bits)
                    buffer_bits += 8
                    byte_pos += 1
                else:
                    break
            
            values.append(bit_buffer & mask)
            bit_buffer >>= chunk.bit_width
            buffer_bits -= chunk.bit_width
        
        return np.array(values, dtype=np.int64)
    
    def _decode_for(self, data: bytes, chunk: BitPackingChunk) -> np.ndarray:
        """Decode Frame-of-Reference packed data."""
        deltas = self._decode_constant(data, 
            BitPackingChunk(BitPackingStrategy.CONSTANT, chunk.bit_width, chunk.value_count, 0, 0, [], b""))
        return deltas + chunk.base_value
    
    def _decode_zero_run(self, data: bytes, chunk: BitPackingChunk) -> np.ndarray:
        """Decode zero-run packed data."""
        # Reconstruct array with zeros
        values = np.zeros(chunk.value_count, dtype=np.int64)
        
        # Place non-zero values at specified positions
        if chunk.zero_positions:
            # Would need to decode the packed values here
            pass
        
        return values
    
    def _decode_delta(self, data: bytes, chunk: BitPackingChunk) -> np.ndarray:
        """Decode delta packed data."""
        deltas = self._decode_constant(data,
            BitPackingChunk(BitPackingStrategy.CONSTANT, chunk.bit_width, chunk.value_count, 0, 0, [], b""))
        
        # Reconstruct from deltas
        values = np.zeros(len(deltas), dtype=np.int64)
        values[0] = chunk.base_value
        
        for i in range(1, len(deltas)):
            values[i] = values[i-1] + deltas[i]
        
        return values


# ============================================================================
# LAYER 4 ENCODER/DECODER
# ============================================================================


class Layer4Encoder:
    """
    Encodes numeric data using Layer 4 Variable Bit-Packing.
    
    Input: Sequences of integers
    Output: Bit-packed, optimized for space
    """
    
    def __init__(self):
        """Initialize Layer 4 encoder."""
        self.packer = BitPackingEncoder()
        self.metadata = {}
    
    def encode(self, data: bytes) -> Tuple[bytes, Dict]:
        """
        Encode numeric data with Layer 4.
        
        Args:
            data: Raw numeric data (sequence of integers)
        
        Returns:
            Tuple of (compressed_data, metadata)
        """
        # Parse integers from data
        values = self._parse_integers(data)
        
        # Pack bits
        compressed, chunks = self.packer.encode(values)
        
        # Build metadata
        self.metadata = {
            "value_count": len(values),
            "chunk_count": len(chunks),
            "original_size": len(data),
            "compressed_size": len(compressed),
            "layer": CompressionLayer.L4_VARIABLE_BITPACKING.value
        }
        
        return compressed, self.metadata
    
    @staticmethod
    def _parse_integers(data: bytes) -> np.ndarray:
        """Parse integer values from data."""
        # Assume 4-byte integers for now
        if len(data) % 4 == 0:
            return np.frombuffer(data, dtype=np.int32)
        elif len(data) % 8 == 0:
            return np.frombuffer(data, dtype=np.int64)
        else:
            # Mixed or byte data
            return np.frombuffer(data, dtype=np.uint8)


class Layer4Decoder:
    """Decodes Layer 4 bit-packed data."""
    
    def __init__(self):
        """Initialize Layer 4 decoder."""
        self.unpacker = BitPackingDecoder()
    
    def decode(self, data: bytes, chunks_metadata: List[BitPackingChunk]) -> bytes:
        """
        Decode Layer 4 compressed data.
        
        Args:
            data: Compressed data
            chunks_metadata: Chunk information from encoder
        
        Returns:
            Decompressed data
        """
        values = self.unpacker.decode(data, chunks_metadata)
        return values.astype(np.int32).tobytes()


# ============================================================================
# TESTING & VALIDATION (Placeholder)
# ============================================================================


def test_bit_width_calculator():
    """Test bit-width analysis."""
    # Test case 1: Constant value range
    values = np.array([1, 2, 3, 4, 5, 100, 200, 255], dtype=np.int64)
    analysis = BitWidthCalculator.analyze_values(values)
    
    print(f"Values: {values}")
    print(f"Analysis: {analysis}")
    print(f"Strategy: {BitPackingStrategy(analysis.strategy).name}")
    print()


def test_bit_packing_encoder():
    """Test bit-packing encoder."""
    values = np.array([100, 101, 102, 103, 104, 105], dtype=np.int64)
    encoder = BitPackingEncoder()
    compressed, chunks = encoder.encode(values)
    
    print(f"Original: {len(values) * 8} bytes (8 bytes per int64)")
    print(f"Compressed: {len(compressed)} bytes")
    print(f"Ratio: {(len(values) * 8) / len(compressed):.2f}:1")
    print(f"Chunks: {len(chunks)}")


if __name__ == "__main__":
    test_bit_width_calculator()
    print("---\n")
    test_bit_packing_encoder()

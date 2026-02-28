"""
COBOL Protocol v1.1 - Layer 2: Structural Mapping
==================================================

Compression of structural patterns in semi-structured data.

Features:
- Pattern tokenization for HTML, XML, JSON structures
- Nesting level optimization with stack-based encoding
- Structural dictionary with 2-byte IDs
- 50-80% compression ratio on structured data

Status: Implementation started (Q2 2026)
"""

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Dict, Tuple, Optional, Set
from collections import deque
import io

import numpy as np

from config import CompressionLayer, L1_MAX_DICTIONARY_SIZE


# ============================================================================
# STRUCTURAL PATTERN DEFINITIONS
# ============================================================================


class StructuralPattern(IntEnum):
    """Enumeration of structural patterns."""
    # Bracket/Delimiter Patterns
    OPEN_ANGLE = 0x01          # <
    CLOSE_ANGLE = 0x02         # >
    OPEN_BRACE = 0x03          # {
    CLOSE_BRACE = 0x04         # }
    OPEN_BRACKET = 0x05        # [
    CLOSE_BRACKET = 0x06       # ]
    
    # Tag-related
    OPENING_TAG = 0x07         # <tag>
    CLOSING_TAG = 0x08         # </tag>
    SELF_CLOSING = 0x09        # />
    
    # Attribute-related
    ATTRIBUTE = 0x0A           # @attr
    EQUALS_SIGN = 0x0B         # =
    COLON = 0x0C               # :
    COMMA = 0x0D               # ,
    
    # Whitespace
    SPACE = 0x0E               # (single space)
    NEWLINE = 0x0F             # \n
    WHITESPACE_RUN = 0x10      # Multiple spaces
    
    # Numeric and special
    NUMERIC_BLOCK = 0x11       # Numeric sequence
    STRING_DELIMITER = 0x12    # Quote marks
    ESCAPE_SEQUENCE = 0x13     # Escape character
    
    # Special
    NESTING_LEVEL = 0x14       # Nesting depth encoding
    QUOTE_PAIR = 0x15          # " ... " or ' ... '
    EOF = 0xFF                 # End of structure


@dataclass
class StructuralToken:
    """Represents a single structural token."""
    pattern: StructuralPattern
    value: Optional[str] = None         # The actual text (if not pattern)
    nesting_level: int = 0               # Depth in structure
    dictionary_id: Optional[int] = None  # Reference to dictionary
    raw_position: int = 0                # Position in original data
    

# ============================================================================
# STRUCTURAL TOKENIZER
# ============================================================================


class StructuralTokenizer:
    """
    Tokenizes semi-structured data into structural patterns.
    
    Handles:
    - HTML/XML tags and attributes
    - JSON objects and arrays
    - Nested structures
    - Whitespace and escaping
    """
    
    # Regex patterns for structure detection
    TAG_START_CHARS = {'<', '{', '['}
    TAG_END_CHARS = {'>', '}', ']'}
    QUOTE_CHARS = {'"', "'"}
    ESCAPE_CHAR = '\\'
    
    def __init__(self, data: bytes):
        """Initialize tokenizer with input data."""
        self.data = data.decode('utf-8', errors='replace')
        self.pos = 0
        self.tokens: List[StructuralToken] = []
        self.nesting_stack: deque = deque()
        
    def tokenize(self) -> List[StructuralToken]:
        """
        Tokenize the input data.
        
        Returns:
            List of StructuralToken objects
        """
        self.tokens = []
        self.pos = 0
        
        while self.pos < len(self.data):
            self._process_character()
        
        # Add EOF marker
        self.tokens.append(StructuralToken(
            pattern=StructuralPattern.EOF,
            nesting_level=len(self.nesting_stack)
        ))
        
        return self.tokens
    
    def _process_character(self) -> None:
        """Process a single character and emit tokens."""
        char = self.data[self.pos]
        
        # Handle whitespace
        if char in {' ', '\t', '\n', '\r'}:
            self._handle_whitespace()
            return
        
        # Handle quoted strings (consume as single token)
        if char in self.QUOTE_CHARS:
            self._handle_quoted_string(char)
            return
        
        # Handle opening brackets
        if char == '<':
            self._handle_angle_bracket()
            return
        elif char == '{':
            self.tokens.append(StructuralToken(
                pattern=StructuralPattern.OPEN_BRACE,
                nesting_level=len(self.nesting_stack),
                raw_position=self.pos
            ))
            self.nesting_stack.append('{')
            self.pos += 1
            return
        elif char == '[':
            self.tokens.append(StructuralToken(
                pattern=StructuralPattern.OPEN_BRACKET,
                nesting_level=len(self.nesting_stack),
                raw_position=self.pos
            ))
            self.nesting_stack.append('[')
            self.pos += 1
            return
        
        # Handle closing brackets
        if char == '}':
            if self.nesting_stack and self.nesting_stack[-1] == '{':
                self.nesting_stack.pop()
            self.tokens.append(StructuralToken(
                pattern=StructuralPattern.CLOSE_BRACE,
                nesting_level=len(self.nesting_stack),
                raw_position=self.pos
            ))
            self.pos += 1
            return
        elif char == ']':
            if self.nesting_stack and self.nesting_stack[-1] == '[':
                self.nesting_stack.pop()
            self.tokens.append(StructuralToken(
                pattern=StructuralPattern.CLOSE_BRACKET,
                nesting_level=len(self.nesting_stack),
                raw_position=self.pos
            ))
            self.pos += 1
            return
        
        # Handle special characters
        if char == ':':
            self.tokens.append(StructuralToken(
                pattern=StructuralPattern.COLON,
                nesting_level=len(self.nesting_stack),
                raw_position=self.pos
            ))
            self.pos += 1
            return
        elif char == ',':
            self.tokens.append(StructuralToken(
                pattern=StructuralPattern.COMMA,
                nesting_level=len(self.nesting_stack),
                raw_position=self.pos
            ))
            self.pos += 1
            return
        elif char == '=':
            self.tokens.append(StructuralToken(
                pattern=StructuralPattern.EQUALS_SIGN,
                nesting_level=len(self.nesting_stack),
                raw_position=self.pos
            ))
            self.pos += 1
            return
        
        # Handle text content (attributes, tag names, values)
        self._handle_text_content()
    
    def _handle_whitespace(self) -> None:
        """Handle whitespace characters."""
        start_pos = self.pos
        ws_count = 0
        
        while self.pos < len(self.data) and self.data[self.pos] in {' ', '\t', '\n', '\r'}:
            if self.data[self.pos] == '\n':
                if ws_count > 0:
                    self.tokens.append(StructuralToken(
                        pattern=StructuralPattern.WHITESPACE_RUN,
                        value=self.data[start_pos:self.pos],
                        nesting_level=len(self.nesting_stack),
                        raw_position=start_pos
                    ))
                ws_count = 0
                self.tokens.append(StructuralToken(
                    pattern=StructuralPattern.NEWLINE,
                    nesting_level=len(self.nesting_stack),
                    raw_position=self.pos
                ))
            else:
                ws_count += 1
            self.pos += 1
        
        if ws_count > 0:
            self.tokens.append(StructuralToken(
                pattern=StructuralPattern.WHITESPACE_RUN,
                value=self.data[start_pos:self.pos],
                nesting_level=len(self.nesting_stack),
                raw_position=start_pos
            ))
    
    def _handle_quoted_string(self, quote_char: str) -> None:
        """Handle quoted string content."""
        start_pos = self.pos
        self.pos += 1  # Skip opening quote
        
        while self.pos < len(self.data):
            if self.data[self.pos] == self.ESCAPE_CHAR:
                self.pos += 2  # Skip escaped character
            elif self.data[self.pos] == quote_char:
                self.pos += 1  # Skip closing quote
                break
            else:
                self.pos += 1
        
        string_content = self.data[start_pos+1:self.pos-1]
        
        self.tokens.append(StructuralToken(
            pattern=StructuralPattern.QUOTE_PAIR,
            value=string_content,
            nesting_level=len(self.nesting_stack),
            raw_position=start_pos
        ))
    
    def _handle_angle_bracket(self) -> None:
        """Handle < and related tag structures."""
        if self.pos + 1 < len(self.data):
            next_char = self.data[self.pos + 1]
            
            # Closing tag: </
            if next_char == '/':
                self.pos += 2
                self._extract_tag_name()
                self.nesting_stack.pop() if self.nesting_stack else None
                
                # Consume >
                if self.pos < len(self.data) and self.data[self.pos] == '>':
                    self.pos += 1
                return
        
        # Opening tag: <
        self.pos += 1
        self._extract_tag_name()
        
        # Check for self-closing
        if self.pos + 1 < len(self.data) and self.data[self.pos:self.pos+2] == '/>':
            self.tokens.append(StructuralToken(
                pattern=StructuralPattern.SELF_CLOSING,
                nesting_level=len(self.nesting_stack),
                raw_position=self.pos
            ))
            self.pos += 2
        else:
            # Consume >
            if self.pos < len(self.data) and self.data[self.pos] == '>':
                self.pos += 1
            self.nesting_stack.append('<')
    
    def _extract_tag_name(self) -> None:
        """Extract tag name from current position."""
        start_pos = self.pos
        
        while self.pos < len(self.data) and self.data[self.pos] not in {'>', ' ', '\t', '\n'}:
            self.pos += 1
        
        tag_name = self.data[start_pos:self.pos]
        if tag_name:
            self.tokens.append(StructuralToken(
                pattern=StructuralPattern.OPENING_TAG,
                value=tag_name,
                nesting_level=len(self.nesting_stack),
                raw_position=start_pos
            ))
    
    def _handle_text_content(self) -> None:
        """Handle text content (not special characters)."""
        start_pos = self.pos
        
        # Collect digits as numeric block
        if self.data[self.pos].isdigit():
            while self.pos < len(self.data) and self.data[self.pos].isdigit():
                self.pos += 1
            
            self.tokens.append(StructuralToken(
                pattern=StructuralPattern.NUMERIC_BLOCK,
                value=self.data[start_pos:self.pos],
                nesting_level=len(self.nesting_stack),
                raw_position=start_pos
            ))
            return
        
        # Regular text/attribute names
        while self.pos < len(self.data) and self.data[self.pos] not in {
            ' ', '\t', '\n', '\r', '<', '>', '{', '}', '[', ']', ':', ',', '=', '"', "'"
        }:
            self.pos += 1
        
        text = self.data[start_pos:self.pos]
        if text:
            self.tokens.append(StructuralToken(
                pattern=StructuralPattern.ATTRIBUTE,
                value=text,
                nesting_level=len(self.nesting_stack),
                raw_position=start_pos
            ))


# ============================================================================
# NESTING LEVEL TRACKER
# ============================================================================


class NestingLevelTracker:
    """
    Tracks and optimizes nesting level encoding.
    
    Reduces nesting information from variable size to minimal bytes.
    """
    
    def __init__(self):
        """Initialize nesting tracker."""
        self.max_nesting_level = 0
        self.level_transitions: List[Tuple[int, int]] = []
        self.current_level = 0
    
    def process_token(self, token: StructuralToken) -> StructuralToken:
        """
        Process token nesting information.
        
        Returns modified token with optimized nesting encoding.
        """
        token.nesting_level = len(token.value or "") if token.pattern in {
            StructuralPattern.OPENING_TAG,
            StructuralPattern.CLOSING_TAG
        } else token.nesting_level
        
        # Track transitions
        if token.nesting_level != self.current_level:
            self.level_transitions.append((self.current_level, token.nesting_level))
            self.current_level = token.nesting_level
        
        self.max_nesting_level = max(self.max_nesting_level, token.nesting_level)
        
        return token
    
    def encode_nesting(self) -> bytes:
        """Encode nesting transitions as variable-length sequence."""
        # TODO: Implement VarInt encoding of level transitions
        pass
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about nesting."""
        return {
            "max_level": self.max_nesting_level,
            "transition_count": len(self.level_transitions),
            "avg_level": sum(t[1] for t in self.level_transitions) / len(self.level_transitions) if self.level_transitions else 0
        }


# ============================================================================
# STRUCTURAL DICTIONARY
# ============================================================================


class StructuralDictionary:
    """
    Dictionary for mapping common structural patterns to 2-byte IDs.
    
    Supports:
    - High-frequency pattern caching
    - Pattern sequence templates
    - Bidirectional encoding/decoding
    """
    
    def __init__(self, max_size: int = 65536):
        """Initialize structural dictionary."""
        self.max_size = max_size
        self.pattern_to_id: Dict[str, int] = {}
        self.id_to_pattern: Dict[int, str] = {}
        self.frequency: Dict[str, int] = {}
        self.next_id = 256  # Reserve 0-255 for single patterns
    
    def add_pattern(self, pattern: str) -> int:
        """
        Add a pattern to dictionary.
        
        Returns 2-byte ID (0-65535)
        """
        if pattern in self.pattern_to_id:
            self.frequency[pattern] += 1
            return self.pattern_to_id[pattern]
        
        if self.next_id >= self.max_size:
            # Dictionary full - could implement LRU eviction here
            return None
        
        pattern_id = self.next_id
        self.pattern_to_id[pattern] = pattern_id
        self.id_to_pattern[pattern_id] = pattern
        self.frequency[pattern] = 1
        self.next_id += 1
        
        return pattern_id
    
    def get_pattern(self, pattern_id: int) -> Optional[str]:
        """Retrieve pattern by ID."""
        return self.id_to_pattern.get(pattern_id)
    
    def get_id(self, pattern: str) -> Optional[int]:
        """Get ID for a pattern."""
        return self.pattern_to_id.get(pattern)
    
    def get_frequent_patterns(self, top_n: int = 100) -> List[Tuple[str, int]]:
        """Get top N most frequent patterns."""
        return sorted(
            self.frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]


# ============================================================================
# LAYER 2 ENCODER/DECODER
# ============================================================================


class Layer2Encoder:
    """
    Encodes structural patterns using Layer 2 compression.
    
    Input: Semi-structured data (JSON, XML, HTML)
    Output: Tokenized + dictionary-encoded + nesting-optimized
    """
    
    def __init__(self):
        """Initialize Layer 2 encoder."""
        self.tokenizer: Optional[StructuralTokenizer] = None
        self.dictionary = StructuralDictionary()
        self.nesting_tracker = NestingLevelTracker()
        self.metadata = {}
    
    def encode(self, data: bytes) -> Tuple[bytes, Dict]:
        """
        Encode structural data.
        
        Args:
            data: Raw semi-structured data
        
        Returns:
            Tuple of (compressed_data, metadata)
        """
        # Tokenize input
        self.tokenizer = StructuralTokenizer(data)
        tokens = self.tokenizer.tokenize()
        
        # Process tokens with nesting optimization
        processed_tokens = [
            self.nesting_tracker.process_token(token)
            for token in tokens
        ]
        
        # Encode tokens
        encoded = self._encode_tokens(processed_tokens)
        
        # Build metadata
        self.metadata = {
            "token_count": len(processed_tokens),
            "dictionary_size": len(self.dictionary.pattern_to_id),
            "nesting_stats": self.nesting_tracker.get_stats(),
            "layer": CompressionLayer.L2_STRUCTURAL_MAPPING.value
        }
        
        return encoded, self.metadata
    
    def _encode_tokens(self, tokens: List[StructuralToken]) -> bytes:
        """Encode token list to bytes."""
        output = io.BytesIO()
        
        for token in tokens:
            # Encode pattern byte
            output.write(bytes([token.pattern.value]))
            
            # Encode value if present
            if token.value is not None:
                # For patterns with values, store dictionary reference if available
                pattern_id = self.dictionary.add_pattern(token.value)
                if pattern_id is not None:
                    # Encode as 2-byte dictionary ID
                    output.write(struct.pack('<H', pattern_id))
                else:
                    # Fallback: encode as length-prefixed string
                    encoded_value = token.value.encode('utf-8')
                    output.write(struct.pack('<H', len(encoded_value)))
                    output.write(encoded_value)
        
        return output.getvalue()


class Layer2Decoder:
    """
    Decodes Layer 2 compressed data.
    
    Requires dictionary and metadata from encoder.
    """
    
    def __init__(self, encoder_metadata: Dict):
        """Initialize decoder with encoder metadata."""
        self.metadata = encoder_metadata
        # Would receive dictionary from encoder in practice
        self.dictionary = StructuralDictionary()
    
    def decode(self, data: bytes, dictionary: Dict[int, str]) -> bytes:
        """
        Decode Layer 2 compressed data.
        
        Args:
            data: Compressed data from Layer 2 encoder
            dictionary: Dictionary from encoder
        
        Returns:
            Decompressed semi-structured data
        """
        # Set dictionary
        self.dictionary.id_to_pattern = dictionary
        
        input_stream = io.BytesIO(data)
        output = io.BytesIO()
        
        while True:
            pattern_byte = input_stream.read(1)
            if not pattern_byte:
                break
            
            pattern = StructuralPattern(pattern_byte[0])
            
            # Reconstruct output based on pattern
            if pattern == StructuralPattern.EOF:
                break
            elif pattern in {
                StructuralPattern.QUOTE_PAIR,
                StructuralPattern.OPENING_TAG,
                StructuralPattern.ATTRIBUTE
            }:
                # Read dictionary ID
                dict_id_bytes = input_stream.read(2)
                if len(dict_id_bytes) == 2:
                    dict_id = struct.unpack('<H', dict_id_bytes)[0]
                    value = self.dictionary.id_to_pattern.get(dict_id, "")
                    output.write(value.encode('utf-8'))
            else:
                # Output pattern character/symbol
                output.write(bytes([pattern.value]))
        
        return output.getvalue()


# ============================================================================
# TESTING & VALIDATION (Placeholder)
# ============================================================================


def test_structural_tokenizer():
    """Test structural tokenization."""
    test_data = b'{"name": "test", "value": 123}'
    tokenizer = StructuralTokenizer(test_data)
    tokens = tokenizer.tokenize()
    
    print(f"Tokenized {len(tokens)} tokens:")
    for token in tokens[:10]:
        print(f"  {token.pattern.name}: {token.value}")


def test_layer2_encoding():
    """Test Layer 2 encoding."""
    test_data = b'<root><child attr="value">123</child></root>'
    encoder = Layer2Encoder()
    compressed, metadata = encoder.encode(test_data)
    
    print(f"Original: {len(test_data)} bytes")
    print(f"Compressed: {len(compressed)} bytes")
    print(f"Ratio: {len(test_data) / len(compressed):.2f}:1")
    print(f"Metadata: {metadata}")


if __name__ == "__main__":
    test_structural_tokenizer()
    print("\n---\n")
    test_layer2_encoding()

"""
COBOL Protocol v1.1 - Advanced Profiling Tools
==============================================

Comprehensive performance analytics for bottleneck identification.

Features:
- Per-layer timing and throughput metrics
- GPU utilization monitoring
- Dictionary analytics (hit/miss rates)
- Streaming latency analysis
- Bottleneck detection with recommendations
- Visualization exports (JSON, CSV, charts)

Status: Framework started (Q2 2026)
"""

import json
import time
import csv
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import io

import numpy as np

from config import CompressionLayer


# ============================================================================
# PROFILE DATA STRUCTURES
# ============================================================================


class BottleneckLevel(Enum):
    """Severity of performance bottleneck."""
    NONE = "none"              # No bottleneck
    MINOR = "minor"            # < 10% overhead
    MODERATE = "moderate"      # 10-30% overhead
    SIGNIFICANT = "significant"  # 30-60% overhead
    CRITICAL = "critical"      # > 60% overhead


@dataclass
class LayerProfile:
    """Performance profile for a single compression layer."""
    layer: CompressionLayer
    input_size: int
    output_size: int
    
    # Timing metrics
    start_time: float
    end_time: float
    duration_ms: float
    
    # Compression metrics
    compression_ratio: float          # input_size / output_size
    throughput_mbps: float            # MB/s
    
    # Dictionary metrics
    dictionary_lookups: int = 0
    dictionary_hits: int = 0
    dictionary_misses: int = 0
    hit_rate: float = 0.0
    
    # Entropy metrics
    entropy_value: float = 0.0
    entropy_skipped: bool = False
    
    # GPU metrics (if GPU used)
    gpu_used: bool = False
    gpu_device_name: str = ""
    gpu_transfer_time_ms: float = 0.0
    gpu_compute_time_ms: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    
    # Memory metrics
    memory_peak_mb: float = 0.0
    memory_allocated_mb: float = 0.0
    
    # Pattern statistics
    pattern_count: int = 0
    pattern_distribution: Dict[str, int] = field(default_factory=dict)
    
    def compute_metrics(self) -> None:
        """Compute derived metrics."""
        if self.output_size > 0:
            self.compression_ratio = self.input_size / self.output_size
        
        if self.duration_ms > 0:
            self.throughput_mbps = (self.input_size / (1024 * 1024)) / (self.duration_ms / 1000)
        
        if self.dictionary_lookups > 0:
            self.hit_rate = self.dictionary_hits / self.dictionary_lookups


@dataclass
class CompressionProfile:
    """Complete compression session profile."""
    session_id: str
    session_start_time: datetime
    session_end_time: Optional[datetime] = None
    
    # Total metrics
    total_input_size: int = 0
    total_output_size: int = 0
    total_duration_ms: float = 0.0
    total_compression_ratio: float = 0.0
    total_throughput_mbps: float = 0.0
    
    # Per-layer profiles
    layer_profiles: Dict[CompressionLayer, LayerProfile] = field(default_factory=dict)
    
    # GPU metrics
    gpu_enabled: bool = False
    gpu_backend: str = ""
    gpu_device_name: str = ""
    gpu_utilization_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    pcie_throughput_gbps: float = 0.0
    
    # Memory analysis
    peak_memory_mb: float = 0.0
    memory_allocations: int = 0
    memory_deallocations: int = 0
    
    # Pattern analysis
    total_patterns_detected: int = 0
    pattern_types: Dict[str, int] = field(default_factory=dict)
    
    # Bottleneck analysis
    bottleneck_level: BottleneckLevel = BottleneckLevel.NONE
    bottleneck_description: str = ""
    bottleneck_recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    data_type: str = ""  # "text", "json", "binary", etc.
    compression_config: Dict = field(default_factory=dict)
    
    def compute_metrics(self) -> None:
        """Compute total and derived metrics."""
        if self.layer_profiles:
            self.total_input_size = sum(p.input_size for p in self.layer_profiles.values())
            self.total_output_size = sum(p.output_size for p in self.layer_profiles.values())
            
            if self.total_output_size > 0:
                self.total_compression_ratio = self.total_input_size / self.total_output_size
            
            self.total_duration_ms = sum(p.duration_ms for p in self.layer_profiles.values())
            
            if self.total_duration_ms > 0:
                self.total_throughput_mbps = (
                    (self.total_input_size / (1024 * 1024)) / 
                    (self.total_duration_ms / 1000)
                )
            
            self.peak_memory_mb = max(p.memory_peak_mb for p in self.layer_profiles.values())


@dataclass
class StreamingProfile:
    """Profile for streaming compression operations."""
    block_number: int
    block_size: int
    input_size: int
    output_size: int
    
    # Timing
    compression_time_ms: float
    
    # QoS metrics
    latency_ms: float              # Time from input to output
    throughput_mbps: float
    jitter_ms: float = 0.0         # Variance from expected latency
    
    # Buffer metrics
    buffer_fill_percent: float = 0.0
    buffer_wait_time_ms: float = 0.0
    
    # Sequence info
    received_in_order: bool = True
    reordering_required: bool = False


# ============================================================================
# PROFILER
# ============================================================================


class CompressionProfiler:
    """
    Comprehensive profiler for compression operations.
    
    Tracks:
    - Per-layer performance metrics
    - GPU utilization
    - Memory usage
    - Bottlenecks
    - Estimates for optimization
    """
    
    def __init__(self, session_id: str = None):
        """Initialize profiler."""
        if session_id is None:
            session_id = f"session_{int(time.time() * 1000)}"
        
        self.session_id = session_id
        self.profile = CompressionProfile(
            session_id=session_id,
            session_start_time=datetime.now()
        )
        self.streaming_profiles: List[StreamingProfile] = []
        
        # Temporary tracking
        self._layer_stack: Dict[CompressionLayer, float] = {}
        self._memory_tracker = MemoryTracker()
        self._gpu_monitor: Optional[GPUMonitor] = None
    
    def start_layer(self, layer: CompressionLayer) -> None:
        """Mark start of layer processing."""
        profile = LayerProfile(
            layer=layer,
            input_size=0,
            output_size=0,
            start_time=time.time(),
            end_time=0.0,
            duration_ms=0.0,
            compression_ratio=1.0,
            throughput_mbps=0.0
        )
        
        self.profile.layer_profiles[layer] = profile
        self._layer_stack[layer] = time.time()
        
        # Track memory at start
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            profile.memory_allocated_mb = mem_info.rss / (1024 * 1024)
        except ImportError:
            pass
    
    def end_layer(self, layer: CompressionLayer, input_size: int, output_size: int) -> None:
        """Mark end of layer processing."""
        if layer not in self._layer_stack:
            return
        
        end_time = time.time()
        start_time = self._layer_stack.pop(layer)
        
        profile = self.profile.layer_profiles.get(layer)
        if profile:
            profile.end_time = end_time
            profile.duration_ms = (end_time - start_time) * 1000
            profile.input_size = input_size
            profile.output_size = output_size
            profile.compute_metrics()
            
            # Track peak memory
            try:
                import psutil
                process = psutil.Process()
                mem_info = process.memory_info()
                mem_mb = mem_info.rss / (1024 * 1024)
                profile.memory_peak_mb = max(profile.memory_peak_mb, mem_mb)
                self.profile.peak_memory_mb = max(self.profile.peak_memory_mb, mem_mb)
            except ImportError:
                pass
    
    def record_dictionary_access(self, layer: CompressionLayer, 
                                 hit: bool, lookups: int = 1) -> None:
        """Record dictionary access (hit or miss)."""
        if layer in self.profile.layer_profiles:
            profile = self.profile.layer_profiles[layer]
            profile.dictionary_lookups += lookups
            
            if hit:
                profile.dictionary_hits += lookups
            else:
                profile.dictionary_misses += lookups
            
            if profile.dictionary_lookups > 0:
                profile.hit_rate = profile.dictionary_hits / profile.dictionary_lookups
    
    def record_entropy(self, layer: CompressionLayer, entropy: float, skipped: bool = False) -> None:
        """Record entropy calculation result."""
        if layer in self.profile.layer_profiles:
            profile = self.profile.layer_profiles[layer]
            profile.entropy_value = entropy
            profile.entropy_skipped = skipped
    
    def record_gpu_metrics(self, layer: CompressionLayer, 
                           device_name: str, 
                           transfer_time_ms: float,
                           compute_time_ms: float,
                           memory_peak_mb: float) -> None:
        """Record GPU operation metrics."""
        if layer in self.profile.layer_profiles:
            profile = self.profile.layer_profiles[layer]
            profile.gpu_used = True
            profile.gpu_device_name = device_name
            profile.gpu_transfer_time_ms = transfer_time_ms
            profile.gpu_compute_time_ms = compute_time_ms
            profile.gpu_memory_peak_mb = memory_peak_mb
    
    def record_memory_peak(self, layer: CompressionLayer, peak_mb: float) -> None:
        """Record peak memory usage for layer."""
        if layer in self.profile.layer_profiles:
            self.profile.layer_profiles[layer].memory_peak_mb = peak_mb
        
        self.profile.peak_memory_mb = max(self.profile.peak_memory_mb, peak_mb)
    
    def record_streaming_block(self, profile: StreamingProfile) -> None:
        """Record streaming compression metrics for a block."""
        self.streaming_profiles.append(profile)
    
    def finalize(self) -> CompressionProfile:
        """Finalize profiling session."""
        self.profile.session_end_time = datetime.now()
        self.profile.compute_metrics()
        self._analyze_bottlenecks()
        return self.profile
    
    def _analyze_bottlenecks(self) -> None:
        """Analyze and identify bottlenecks."""
        if not self.profile.layer_profiles:
            return
        
        layer_times = {
            layer: profile.duration_ms
            for layer, profile in self.profile.layer_profiles.items()
        }
        
        if not layer_times:
            return
        
        max_time = max(layer_times.values())
        total_time = sum(layer_times.values())
        
        # Find slowest layer
        slowest_layer = max(layer_times.items(), key=lambda x: x[1])[0]
        slowest_percent = (layer_times[slowest_layer] / total_time * 100) if total_time > 0 else 0
        
        # Determine bottleneck level
        if slowest_percent > 60:
            self.profile.bottleneck_level = BottleneckLevel.CRITICAL
            self.profile.bottleneck_description = (
                f"{slowest_layer.name} is critical bottleneck ({slowest_percent:.1f}% of time)"
            )
            self.profile.bottleneck_recommendations = [
                f"Optimize {slowest_layer.name} algorithm",
                f"Consider GPU acceleration for {slowest_layer.name}",
                f"Profile {slowest_layer.name} with detailed tracing"
            ]
        elif slowest_percent > 30:
            self.profile.bottleneck_level = BottleneckLevel.SIGNIFICANT
            self.profile.bottleneck_description = (
                f"{slowest_layer.name} is significant bottleneck ({slowest_percent:.1f}% of time)"
            )
            self.profile.bottleneck_recommendations = [
                f"Consider optimizing {slowest_layer.name}",
                f"GPU acceleration may help with {slowest_layer.name}"
            ]
        elif slowest_percent > 10:
            self.profile.bottleneck_level = BottleneckLevel.MODERATE
        else:
            self.profile.bottleneck_level = BottleneckLevel.NONE


# ============================================================================
# MEMORY TRACKER
# ============================================================================


class MemoryTracker:
    """Tracks memory allocations and peak usage."""
    
    def __init__(self):
        """Initialize memory tracker."""
        self.allocations: Dict[int, Tuple[str, int]] = {}  # addr -> (name, size)
        self.peak_usage_bytes = 0
        self.current_usage_bytes = 0
    
    def allocate(self, name: str, size_bytes: int) -> int:
        """Track memory allocation."""
        addr = id(name)
        self.allocations[addr] = (name, size_bytes)
        self.current_usage_bytes += size_bytes
        self.peak_usage_bytes = max(self.peak_usage_bytes, self.current_usage_bytes)
        return addr
    
    def free(self, addr: int) -> None:
        """Track memory deallocation."""
        if addr in self.allocations:
            _, size = self.allocations.pop(addr)
            self.current_usage_bytes -= size


# ============================================================================
# GPU MONITOR
# ============================================================================


class GPUMonitor:
    """Monitors GPU utilization during compression."""
    
    def __init__(self, backend_name: str = "Unknown"):
        """Initialize GPU monitor."""
        self.backend_name = backend_name
        self.samples: List[Dict] = []
        self._monitoring = False
    
    def start_monitoring(self) -> None:
        """Start GPU monitoring."""
        self._monitoring = True
    
    def stop_monitoring(self) -> Dict:
        """Stop GPU monitoring and return statistics."""
        self._monitoring = False
        
        if not self.samples:
            return {
                "avg_utilization": 0.0,
                "peak_memory_mb": 0.0,
                "avg_memory_mb": 0.0
            }
        
        utilizations = [s.get("utilization", 0) for s in self.samples]
        memory_values = [s.get("memory_mb", 0) for s in self.samples]
        
        return {
            "avg_utilization": np.mean(utilizations),
            "peak_memory_mb": np.max(memory_values),
            "avg_memory_mb": np.mean(memory_values),
            "sample_count": len(self.samples)
        }
    
    def sample(self) -> None:
        """Take a GPU utilization sample."""
        if not self._monitoring:
            return
        
        # TODO: Implement GPU monitoring
        # For CUDA: use nvidia-ml-py3
        # For OpenCL: use time-based estimation
        
        self.samples.append({
            "timestamp": time.time(),
            "utilization": 0.0,
            "memory_mb": 0.0
        })


# ============================================================================
# PROFILING REPORTS
# ============================================================================


class ProfileReporter:
    """Generate reports from compression profiles."""
    
    @staticmethod
    def to_json(profile: CompressionProfile) -> str:
        """Export profile to JSON."""
        data = {
            "session_id": profile.session_id,
            "session_start": profile.session_start_time.isoformat(),
            "session_end": profile.session_end_time.isoformat() if profile.session_end_time else None,
            "total_metrics": {
                "input_size_bytes": profile.total_input_size,
                "output_size_bytes": profile.total_output_size,
                "compression_ratio": profile.total_compression_ratio,
                "duration_ms": profile.total_duration_ms,
                "throughput_mbps": profile.total_throughput_mbps
            },
            "layer_metrics": {}
        }
        
        for layer, layer_profile in profile.layer_profiles.items():
            data["layer_metrics"][layer.name] = asdict(layer_profile)
        
        # Add GPU metrics if available
        if profile.gpu_enabled:
            data["gpu_metrics"] = {
                "backend": profile.gpu_backend,
                "device_name": profile.gpu_device_name,
                "utilization_percent": profile.gpu_utilization_percent,
                "memory_mb": profile.gpu_memory_used_mb
            }
        
        # Add bottleneck analysis
        data["bottleneck_analysis"] = {
            "level": profile.bottleneck_level.value,
            "description": profile.bottleneck_description,
            "recommendations": profile.bottleneck_recommendations
        }
        
        return json.dumps(data, indent=2, default=str)
    
    @staticmethod
    def to_csv(profile: CompressionProfile) -> str:
        """Export layer metrics to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            "Layer",
            "Input Size (bytes)",
            "Output Size (bytes)",
            "Compression Ratio",
            "Duration (ms)",
            "Throughput (MB/s)",
            "Dictionary Hit Rate",
            "Entropy",
            "GPU Used"
        ])
        
        # Rows
        for layer, layer_profile in profile.layer_profiles.items():
            writer.writerow([
                layer.name,
                layer_profile.input_size,
                layer_profile.output_size,
                f"{layer_profile.compression_ratio:.2f}",
                f"{layer_profile.duration_ms:.2f}",
                f"{layer_profile.throughput_mbps:.2f}",
                f"{layer_profile.hit_rate:.2%}",
                f"{layer_profile.entropy_value:.4f}",
                "Yes" if layer_profile.gpu_used else "No"
            ])
        
        return output.getvalue()
    
    @staticmethod
    def generate_report(profile: CompressionProfile) -> str:
        """Generate human-readable text report."""
        lines = [
            "=" * 70,
            "COBOL Protocol Compression Profile Report",
            "=" * 70,
            f"Session: {profile.session_id}",
            f"Duration: {profile.session_start_time.isoformat()} -> {profile.session_end_time.isoformat()}",
            "",
            "SUMMARY METRICS",
            "-" * 70,
            f"Input Size:         {profile.total_input_size:,} bytes",
            f"Output Size:        {profile.total_output_size:,} bytes",
            f"Compression Ratio:  {profile.total_compression_ratio:.2f}:1",
            f"Total Duration:     {profile.total_duration_ms:.2f} ms",
            f"Throughput:         {profile.total_throughput_mbps:.2f} MB/s",
            "",
            "PER-LAYER ANALYSIS",
            "-" * 70,
        ]
        
        for layer, layer_profile in profile.layer_profiles.items():
            lines.extend([
                f"\n{layer.name}:",
                f"  Input:              {layer_profile.input_size:,} bytes",
                f"  Output:             {layer_profile.output_size:,} bytes",
                f"  Ratio:              {layer_profile.compression_ratio:.2f}:1",
                f"  Duration:           {layer_profile.duration_ms:.2f} ms",
                f"  Throughput:         {layer_profile.throughput_mbps:.2f} MB/s",
                f"  Dictionary Hit Rate:{layer_profile.hit_rate:.1%}",
                f"  Entropy:            {layer_profile.entropy_value:.4f}",
                f"  GPU Used:           {'Yes' if layer_profile.gpu_used else 'No'}"
            ])
        
        if profile.gpu_enabled:
            lines.extend([
                "",
                "GPU METRICS",
                "-" * 70,
                f"Backend:            {profile.gpu_backend}",
                f"Device:             {profile.gpu_device_name}",
                f"Utilization:        {profile.gpu_utilization_percent:.1f}%",
                f"Memory Used:        {profile.gpu_memory_used_mb:.1f} MB"
            ])
        
        lines.extend([
            "",
            "BOTTLENECK ANALYSIS",
            "-" * 70,
            f"Level:              {profile.bottleneck_level.value.upper()}",
            f"Description:        {profile.bottleneck_description}",
        ])
        
        if profile.bottleneck_recommendations:
            lines.append("Recommendations:")
            for rec in profile.bottleneck_recommendations:
                lines.append(f"  - {rec}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    profiler = CompressionProfiler("test_session")
    
    # Simulate layer processing
    profiler.start_layer(CompressionLayer.L1_SEMANTIC_MAPPING)
    time.sleep(0.01)  # Simulate work
    profiler.end_layer(CompressionLayer.L1_SEMANTIC_MAPPING, 10000, 5000)
    profiler.record_entropy(CompressionLayer.L1_SEMANTIC_MAPPING, 0.75)
    
    # Finalize and generate report
    profile = profiler.finalize()
    
    # Export
    print(ProfileReporter.generate_report(profile))
    print("\n" + ProfileReporter.to_csv(profile))

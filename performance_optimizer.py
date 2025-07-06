#!/usr/bin/env python3
"""
Performance Optimization Module
High-performance processing with GPU acceleration, multiprocessing, and memory optimization.
"""

import numpy as np
import librosa
import concurrent.futures
import multiprocessing as mp
from functools import partial
import psutil
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import time
import threading
import queue

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

try:
    import numba
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    numba = None
    jit = lambda x: x  # Dummy decorator

logger = logging.getLogger(__name__)

@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    use_gpu: bool = GPU_AVAILABLE
    use_multiprocessing: bool = True
    max_workers: int = None
    chunk_size: int = 48000 * 30  # 30 seconds at 48kHz
    overlap_size: int = 4800      # 0.1 second overlap
    memory_limit_gb: float = 8.0
    use_jit_compilation: bool = NUMBA_AVAILABLE
    prefetch_chunks: bool = True
    batch_processing: bool = True

class PerformanceOptimizer:
    """High-performance audio processing optimizer."""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        
        # Auto-configure based on system capabilities
        if self.config.max_workers is None:
            self.config.max_workers = min(mp.cpu_count(), 8)
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor(self.config.memory_limit_gb)
        
        # GPU setup
        if self.config.use_gpu and GPU_AVAILABLE:
            try:
                cp.cuda.Device(0).use()
                logger.info("GPU acceleration enabled")
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}")
                self.config.use_gpu = False
        
        # JIT compilation cache
        self._jit_cache = {}
        
        logger.info(f"Performance optimizer initialized: "
                   f"GPU={self.config.use_gpu}, "
                   f"Workers={self.config.max_workers}, "
                   f"JIT={self.config.use_jit_compilation}")
    
    def process_audio_parallel(self, audio: np.ndarray, sr: int, 
                             processing_func: Callable, 
                             func_args: Tuple = ()) -> np.ndarray:
        """Process audio in parallel chunks with optimal performance."""
        
        if len(audio) <= self.config.chunk_size:
            # Small audio, process directly
            return processing_func(audio, sr, *func_args)
        
        # Split into chunks
        chunks = self._split_into_chunks(audio)
        
        if self.config.use_multiprocessing and len(chunks) > 1:
            return self._process_chunks_multiprocessing(chunks, sr, processing_func, func_args)
        else:
            return self._process_chunks_sequential(chunks, sr, processing_func, func_args)
    
    def _split_into_chunks(self, audio: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """Split audio into overlapping chunks for parallel processing."""
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap_size
        
        start = 0
        while start < len(audio):
            end = min(start + chunk_size, len(audio))
            chunk_audio = audio[start:end]
            chunks.append((start, chunk_audio))
            
            if end >= len(audio):
                break
            
            start = end - overlap
        
        logger.debug(f"Split audio into {len(chunks)} chunks")
        return chunks
    
    def _process_chunks_multiprocessing(self, chunks: List[Tuple[int, np.ndarray]], 
                                      sr: int, processing_func: Callable, 
                                      func_args: Tuple) -> np.ndarray:
        """Process chunks using multiprocessing."""
        
        # Create partial function with fixed arguments
        worker_func = partial(self._chunk_worker, sr=sr, func=processing_func, args=func_args)
        
        # Process chunks in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_chunk = {
                executor.submit(worker_func, chunk_data): i 
                for i, chunk_data in enumerate(chunks)
            }
            
            processed_chunks = [None] * len(chunks)
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    processed_chunks[chunk_idx] = future.result()
                except Exception as e:
                    logger.error(f"Chunk {chunk_idx} processing failed: {e}")
                    # Use original chunk as fallback
                    processed_chunks[chunk_idx] = (chunks[chunk_idx][0], chunks[chunk_idx][1])
        
        # Reconstruct audio from processed chunks
        return self._reconstruct_from_chunks(processed_chunks)
    
    def _process_chunks_sequential(self, chunks: List[Tuple[int, np.ndarray]], 
                                 sr: int, processing_func: Callable, 
                                 func_args: Tuple) -> np.ndarray:
        """Process chunks sequentially with memory optimization."""
        processed_chunks = []
        
        for i, (start_pos, chunk_audio) in enumerate(chunks):
            try:
                # Monitor memory usage
                if self.memory_monitor.should_gc():
                    import gc
                    gc.collect()
                
                # Process chunk
                processed_audio = processing_func(chunk_audio, sr, *func_args)
                processed_chunks.append((start_pos, processed_audio))
                
                logger.debug(f"Processed chunk {i+1}/{len(chunks)}")
                
            except Exception as e:
                logger.error(f"Chunk {i} processing failed: {e}")
                processed_chunks.append((start_pos, chunk_audio))  # Fallback
        
        return self._reconstruct_from_chunks(processed_chunks)
    
    @staticmethod
    def _chunk_worker(chunk_data: Tuple[int, np.ndarray], sr: int, 
                     func: Callable, args: Tuple) -> Tuple[int, np.ndarray]:
        """Worker function for processing individual chunks."""
        start_pos, chunk_audio = chunk_data
        try:
            processed = func(chunk_audio, sr, *args)
            return (start_pos, processed)
        except Exception as e:
            logger.error(f"Worker failed: {e}")
            return (start_pos, chunk_audio)  # Return original on failure
    
    def _reconstruct_from_chunks(self, processed_chunks: List[Tuple[int, np.ndarray]]) -> np.ndarray:
        """Reconstruct audio from processed chunks with overlap handling."""
        if not processed_chunks:
            return np.array([])
        
        # Sort by start position
        processed_chunks.sort(key=lambda x: x[0])
        
        # Calculate total length
        last_start, last_chunk = processed_chunks[-1]
        total_length = last_start + len(last_chunk)
        
        # Determine if input was stereo
        sample_chunk = processed_chunks[0][1]
        if sample_chunk.ndim > 1:
            # Stereo
            reconstructed = np.zeros((total_length, sample_chunk.shape[1]))
        else:
            # Mono
            reconstructed = np.zeros(total_length)
        
        overlap = self.config.overlap_size
        
        for i, (start_pos, chunk) in enumerate(processed_chunks):
            end_pos = start_pos + len(chunk)
            
            if i == 0:
                # First chunk - use entirely
                reconstructed[start_pos:end_pos] = chunk
            else:
                # Subsequent chunks - handle overlap
                if start_pos < len(reconstructed):
                    overlap_start = start_pos
                    overlap_end = min(overlap_start + overlap, end_pos, len(reconstructed))
                    
                    if overlap_end > overlap_start:
                        # Cross-fade in overlap region
                        overlap_len = overlap_end - overlap_start
                        fade_in = np.linspace(0, 1, overlap_len)
                        fade_out = 1 - fade_in
                        
                        chunk_overlap = chunk[:overlap_len]
                        existing_overlap = reconstructed[overlap_start:overlap_end]
                        
                        if sample_chunk.ndim > 1:
                            # Stereo
                            fade_in = fade_in[:, np.newaxis]
                            fade_out = fade_out[:, np.newaxis]
                        
                        reconstructed[overlap_start:overlap_end] = (
                            existing_overlap * fade_out + chunk_overlap * fade_in
                        )
                        
                        # Add remaining part of chunk
                        if overlap_end < end_pos:
                            remaining_chunk = chunk[overlap_len:]
                            if overlap_end + len(remaining_chunk) <= len(reconstructed):
                                reconstructed[overlap_end:overlap_end + len(remaining_chunk)] = remaining_chunk
                    else:
                        # No overlap, just append
                        if end_pos <= len(reconstructed):
                            reconstructed[start_pos:end_pos] = chunk
        
        return reconstructed
    
    def gpu_accelerate_fft(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated FFT computation."""
        if not self.config.use_gpu or not GPU_AVAILABLE:
            # Fallback to CPU
            spectrum = np.fft.fft(audio)
            return np.abs(spectrum), np.angle(spectrum)
        
        try:
            # Transfer to GPU
            audio_gpu = cp.asarray(audio)
            
            # Compute FFT on GPU
            spectrum_gpu = cp.fft.fft(audio_gpu)
            magnitude_gpu = cp.abs(spectrum_gpu)
            phase_gpu = cp.angle(spectrum_gpu)
            
            # Transfer back to CPU
            magnitude = cp.asnumpy(magnitude_gpu)
            phase = cp.asnumpy(phase_gpu)
            
            return magnitude, phase
            
        except Exception as e:
            logger.warning(f"GPU FFT failed, falling back to CPU: {e}")
            spectrum = np.fft.fft(audio)
            return np.abs(spectrum), np.angle(spectrum)
    
    def gpu_accelerate_stft(self, audio: np.ndarray, n_fft: int = 2048, 
                          hop_length: int = 512) -> np.ndarray:
        """GPU-accelerated STFT computation."""
        if not self.config.use_gpu or not GPU_AVAILABLE:
            return librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        
        try:
            # For now, use CPU STFT as GPU STFT is complex
            # In a real implementation, would use cuPy's FFT with windowing
            return librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            
        except Exception as e:
            logger.warning(f"GPU STFT failed: {e}")
            return librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    
    @jit(nopython=True, cache=True) if NUMBA_AVAILABLE else lambda x: x
    def _jit_spectral_subtraction(self, magnitude: np.ndarray, noise_profile: np.ndarray, 
                                alpha: float = 2.0) -> np.ndarray:
        """JIT-compiled spectral subtraction for performance."""
        # Ensure same shape
        if noise_profile.shape != magnitude.shape:
            # Simple broadcasting - expand noise profile
            noise_expanded = np.zeros_like(magnitude)
            for i in range(magnitude.shape[1]):
                noise_expanded[:, i] = noise_profile[:, 0] if noise_profile.shape[1] > 0 else 0
            noise_profile = noise_expanded
        
        # Spectral subtraction
        result = magnitude - alpha * noise_profile
        
        # Ensure non-negative with floor
        floor_factor = 0.1
        result = np.maximum(result, floor_factor * magnitude)
        
        return result
    
    @jit(nopython=True, cache=True) if NUMBA_AVAILABLE else lambda x: x
    def _jit_apply_filter(self, audio: np.ndarray, filter_coeffs: np.ndarray) -> np.ndarray:
        """JIT-compiled filter application."""
        # Simple FIR filter implementation
        filtered = np.zeros_like(audio)
        filter_len = len(filter_coeffs)
        
        for i in range(len(audio)):
            for j in range(filter_len):
                if i - j >= 0:
                    filtered[i] += audio[i - j] * filter_coeffs[j]
        
        return filtered
    
    def batch_process_features(self, audio_list: List[np.ndarray], sr: int, 
                             feature_func: Callable) -> List[np.ndarray]:
        """Batch process multiple audio files for feature extraction."""
        if not self.config.batch_processing or len(audio_list) == 1:
            return [feature_func(audio, sr) for audio in audio_list]
        
        # Batch processing with multiprocessing
        worker_func = partial(feature_func, sr=sr)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            results = list(executor.map(worker_func, audio_list))
        
        return results
    
    def optimize_memory_usage(self, processing_func: Callable) -> Callable:
        """Decorator to optimize memory usage during processing."""
        def wrapper(*args, **kwargs):
            # Monitor memory before processing
            initial_memory = self.memory_monitor.get_memory_usage()
            
            try:
                result = processing_func(*args, **kwargs)
                
                # Check if memory usage is too high
                current_memory = self.memory_monitor.get_memory_usage()
                if current_memory > self.config.memory_limit_gb:
                    logger.warning(f"High memory usage: {current_memory:.1f}GB")
                    import gc
                    gc.collect()
                
                return result
                
            except MemoryError:
                logger.error("Out of memory - reducing chunk size")
                # Reduce chunk size and retry
                self.config.chunk_size //= 2
                if self.config.chunk_size < 4800:  # Minimum 0.1 seconds
                    raise
                return processing_func(*args, **kwargs)
        
        return wrapper

class MemoryMonitor:
    """Monitor and manage memory usage."""
    
    def __init__(self, limit_gb: float = 8.0):
        self.limit_gb = limit_gb
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            return self.process.memory_info().rss / (1024**3)
        except:
            return 0.0
    
    def should_gc(self) -> bool:
        """Check if garbage collection should be triggered."""
        return self.get_memory_usage() > self.limit_gb * 0.8
    
    def get_memory_percent(self) -> float:
        """Get memory usage as percentage of limit."""
        return (self.get_memory_usage() / self.limit_gb) * 100

class StreamingProcessor:
    """Real-time streaming audio processor."""
    
    def __init__(self, processing_func: Callable, sr: int = 48000, 
                 buffer_size: int = 4800):  # 0.1 second buffer
        self.processing_func = processing_func
        self.sr = sr
        self.buffer_size = buffer_size
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.running = False
    
    def start(self):
        """Start streaming processing."""
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        logger.info("Streaming processor started")
    
    def stop(self):
        """Stop streaming processing."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("Streaming processor stopped")
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Process a single audio chunk."""
        try:
            self.input_queue.put(audio_chunk, timeout=0.1)
            return self.output_queue.get(timeout=0.1)
        except queue.Empty:
            return None
        except queue.Full:
            logger.warning("Processing queue full, dropping frame")
            return None
    
    def _processing_loop(self):
        """Main processing loop for streaming."""
        while self.running:
            try:
                audio_chunk = self.input_queue.get(timeout=0.1)
                processed_chunk = self.processing_func(audio_chunk, self.sr)
                self.output_queue.put(processed_chunk, timeout=0.1)
            except queue.Empty:
                continue
            except queue.Full:
                logger.warning("Output queue full, dropping processed frame")
                continue
            except Exception as e:
                logger.error(f"Streaming processing error: {e}")

def benchmark_performance():
    """Benchmark performance optimizations."""
    print("Performance Benchmark")
    print("====================")
    
    # Create test audio
    sr = 48000
    duration = 30  # 30 seconds
    test_audio = np.random.randn(sr * duration)
    
    # Test function
    def dummy_processing(audio, sr):
        # Simulate processing
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        # Some processing
        processed_magnitude = magnitude * 0.95
        processed_stft = processed_magnitude * np.exp(1j * np.angle(stft))
        return librosa.istft(processed_stft, hop_length=512)
    
    optimizer = PerformanceOptimizer()
    
    # Benchmark sequential vs parallel
    start_time = time.time()
    result_sequential = dummy_processing(test_audio, sr)
    sequential_time = time.time() - start_time
    
    start_time = time.time()
    result_parallel = optimizer.process_audio_parallel(test_audio, sr, dummy_processing)
    parallel_time = time.time() - start_time
    
    print(f"Sequential processing: {sequential_time:.2f}s")
    print(f"Parallel processing: {parallel_time:.2f}s")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")
    
    # Memory usage
    memory_usage = optimizer.memory_monitor.get_memory_usage()
    print(f"Memory usage: {memory_usage:.2f}GB")
    
    # GPU availability
    print(f"GPU acceleration: {'Available' if GPU_AVAILABLE else 'Not available'}")
    print(f"JIT compilation: {'Available' if NUMBA_AVAILABLE else 'Not available'}")

def main():
    """Test performance optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Optimization Test")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--test-streaming", action="store_true", help="Test streaming processor")
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_performance()
    
    if args.test_streaming:
        print("Testing streaming processor...")
        
        def simple_processing(audio, sr):
            return audio * 0.95  # Simple gain reduction
        
        processor = StreamingProcessor(simple_processing)
        processor.start()
        
        # Test with some chunks
        for i in range(10):
            test_chunk = np.random.randn(4800)  # 0.1 second at 48kHz
            result = processor.process_chunk(test_chunk)
            if result is not None:
                print(f"Processed chunk {i+1}")
            time.sleep(0.1)
        
        processor.stop()

if __name__ == "__main__":
    main()
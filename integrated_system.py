#!/usr/bin/env python3
"""
Integrated Next-Generation AI Audio Fingerprint Removal System
Combines all advanced detection and removal techniques into a unified system.
"""

import numpy as np
import soundfile as sf
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time
import os
from pathlib import Path

# Import all our advanced modules
try:
    from advanced_steganography_detector import AdvancedSteganographyDetector
    from neural_watermark_detector import NeuralWatermarkDetector
    from next_gen_remover import NextGenWatermarkRemover
    from performance_optimizer import PerformanceOptimizer, PerformanceConfig
    from aggressive_watermark_remover import AggressiveWatermarkRemover
    from enhanced_suno_detector import SunoWatermarkDetector
except ImportError as e:
    logging.warning(f"Could not import advanced modules: {e}")
    # Fallback to basic functionality

logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    """Comprehensive processing statistics."""
    total_time: float
    detection_time: float
    removal_time: float
    optimization_time: float
    
    watermarks_detected: int
    steganography_detected: int
    neural_patterns_detected: int
    
    watermarks_removed: int
    quality_preserved: float  # 0-1 scale
    artifacts_introduced: float  # 0-1 scale
    
    memory_peak_gb: float
    cpu_usage_percent: float
    gpu_acceleration_used: bool
    
    processing_level: str
    file_size_mb: float
    audio_duration_seconds: float

class IntegratedWatermarkRemover:
    """
    Next-generation integrated AI audio fingerprint removal system.
    
    Combines:
    - Advanced steganography detection
    - Neural network-based watermark detection
    - Next-generation removal algorithms
    - Performance optimization
    - Quality preservation
    """
    
    def __init__(self, performance_config: PerformanceConfig = None):
        self.performance_config = performance_config or PerformanceConfig()
        
        # Initialize all subsystems
        try:
            self.steg_detector = AdvancedSteganographyDetector()
            self.neural_detector = NeuralWatermarkDetector()
            self.nextgen_remover = NextGenWatermarkRemover()
            self.performance_optimizer = PerformanceOptimizer(self.performance_config)
            self.aggressive_remover = AggressiveWatermarkRemover()
            self.suno_detector = SunoWatermarkDetector()
            
            self.systems_available = True
            logger.info("All advanced systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced systems: {e}")
            self.systems_available = False
            # Fallback to basic processing
        
        # Processing statistics
        self.stats = None
        
        # Quality thresholds
        self.quality_thresholds = {
            'gentle': {'min_quality': 0.95, 'max_artifacts': 0.05},
            'balanced': {'min_quality': 0.90, 'max_artifacts': 0.10},
            'aggressive': {'min_quality': 0.85, 'max_artifacts': 0.15},
            'maximum': {'min_quality': 0.75, 'max_artifacts': 0.25}
        }
    
    def process_file(self, input_path: str, output_path: str, 
                    processing_level: str = 'balanced',
                    enable_advanced_detection: bool = True,
                    enable_performance_optimization: bool = True,
                    generate_report: bool = False) -> ProcessingStats:
        """
        Process an audio file with next-generation watermark removal.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output audio file
            processing_level: 'gentle', 'balanced', 'aggressive', 'maximum'
            enable_advanced_detection: Use advanced detection algorithms
            enable_performance_optimization: Use performance optimizations
            generate_report: Generate detailed processing report
        
        Returns:
            ProcessingStats object with comprehensive statistics
        """
        start_time = time.time()
        
        # Validate inputs
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Get file info
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        
        logger.info(f"Starting integrated processing: {input_path}")
        logger.info(f"Level: {processing_level}, File size: {file_size_mb:.1f}MB")
        
        # Load audio
        try:
            audio, sr = sf.read(input_path)
            audio_duration = len(audio) / sr
            logger.info(f"Loaded audio: {len(audio)} samples at {sr}Hz ({audio_duration:.1f}s)")
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {e}")
        
        # Initialize statistics
        self.stats = ProcessingStats(
            total_time=0.0,
            detection_time=0.0,
            removal_time=0.0,
            optimization_time=0.0,
            watermarks_detected=0,
            steganography_detected=0,
            neural_patterns_detected=0,
            watermarks_removed=0,
            quality_preserved=1.0,
            artifacts_introduced=0.0,
            memory_peak_gb=0.0,
            cpu_usage_percent=0.0,
            gpu_acceleration_used=self.performance_config.use_gpu,
            processing_level=processing_level,
            file_size_mb=file_size_mb,
            audio_duration_seconds=audio_duration
        )
        
        try:
            # Phase 1: Advanced Detection
            detection_start = time.time()
            
            if enable_advanced_detection and self.systems_available:
                detections = self._run_advanced_detection(audio, sr)
            else:
                detections = self._run_basic_detection(audio, sr)
            
            self.stats.detection_time = time.time() - detection_start
            
            # Phase 2: Removal Processing
            removal_start = time.time()
            
            if enable_performance_optimization and self.systems_available:
                processed_audio = self._run_optimized_removal(audio, sr, detections, processing_level)
            else:
                processed_audio = self._run_basic_removal(audio, sr, detections, processing_level)
            
            self.stats.removal_time = time.time() - removal_start
            
            # Phase 3: Quality Validation
            optimization_start = time.time()
            
            processed_audio = self._validate_and_optimize_quality(processed_audio, audio, sr, processing_level)
            
            self.stats.optimization_time = time.time() - optimization_start
            
            # Save output
            sf.write(output_path, processed_audio, sr)
            
            # Finalize statistics
            self.stats.total_time = time.time() - start_time
            self.stats.quality_preserved = self._assess_quality_preservation(processed_audio, audio)
            self.stats.artifacts_introduced = self._assess_artifacts(processed_audio, audio)
            
            if self.systems_available:
                self.stats.memory_peak_gb = self.performance_optimizer.memory_monitor.get_memory_usage()
            
            logger.info(f"Processing complete: {self.stats.total_time:.2f}s")
            
            # Generate report if requested
            if generate_report:
                self._generate_processing_report(input_path, output_path)
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
    
    def _run_advanced_detection(self, audio: np.ndarray, sr: int) -> Dict[str, List]:
        """Run advanced detection algorithms."""
        detections = {
            'steganography': [],
            'neural': [],
            'suno': [],
            'traditional': []
        }
        
        try:
            # Advanced steganography detection
            steg_detections = self.steg_detector.detect_all_steganography(audio, sr)
            detections['steganography'] = steg_detections
            self.stats.steganography_detected = len(steg_detections)
            logger.info(f"Steganography detection: {len(steg_detections)} patterns found")
            
            # Neural watermark detection
            neural_detections = self.neural_detector.detect_neural_watermarks(audio, sr)
            detections['neural'] = neural_detections
            self.stats.neural_patterns_detected = len(neural_detections)
            logger.info(f"Neural detection: {len(neural_detections)} patterns found")
            
            # Enhanced Suno detection
            suno_detections = self.suno_detector.detect_suno_watermarks(audio, sr)
            detections['suno'] = suno_detections
            logger.info(f"Suno detection: {len(suno_detections)} patterns found")
            
            self.stats.watermarks_detected = (len(steg_detections) + 
                                            len(neural_detections) + 
                                            len(suno_detections))
            
        except Exception as e:
            logger.warning(f"Advanced detection failed: {e}")
            detections = self._run_basic_detection(audio, sr)
        
        return detections
    
    def _run_basic_detection(self, audio: np.ndarray, sr: int) -> Dict[str, List]:
        """Run basic detection as fallback."""
        detections = {
            'steganography': [],
            'neural': [],
            'suno': [],
            'traditional': []
        }
        
        try:
            # Basic Suno detection only
            suno_detections = self.suno_detector.detect_suno_watermarks(audio, sr)
            detections['suno'] = suno_detections
            self.stats.watermarks_detected = len(suno_detections)
            logger.info(f"Basic detection: {len(suno_detections)} patterns found")
            
        except Exception as e:
            logger.warning(f"Basic detection failed: {e}")
        
        return detections
    
    def _run_optimized_removal(self, audio: np.ndarray, sr: int, 
                             detections: Dict[str, List], 
                             processing_level: str) -> np.ndarray:
        """Run optimized removal with performance enhancements."""
        
        def combined_removal_func(audio_chunk, sr):
            """Combined removal function for parallel processing."""
            try:
                # Apply next-generation removal
                processed, results = self.nextgen_remover.remove_advanced_watermarks(
                    audio_chunk, sr, processing_level
                )
                
                # Update statistics
                self.stats.watermarks_removed += sum(r.watermarks_removed for r in results)
                
                return processed
                
            except Exception as e:
                logger.warning(f"Next-gen removal failed: {e}")
                # Fallback to aggressive removal
                return self.aggressive_remover.remove_watermarks_aggressive(
                    audio_chunk, sr, detections.get('suno', [])
                )
        
        # Use performance optimizer for parallel processing
        return self.performance_optimizer.process_audio_parallel(
            audio, sr, combined_removal_func
        )
    
    def _run_basic_removal(self, audio: np.ndarray, sr: int, 
                         detections: Dict[str, List], 
                         processing_level: str) -> np.ndarray:
        """Run basic removal as fallback."""
        try:
            # Use aggressive remover
            processed = self.aggressive_remover.remove_watermarks_aggressive(
                audio, sr, detections.get('suno', [])
            )
            self.stats.watermarks_removed = len(detections.get('suno', []))
            return processed
            
        except Exception as e:
            logger.error(f"Basic removal failed: {e}")
            return audio  # Return original audio as last resort
    
    def _validate_and_optimize_quality(self, processed_audio: np.ndarray, 
                                     original_audio: np.ndarray, sr: int,
                                     processing_level: str) -> np.ndarray:
        """Validate and optimize audio quality."""
        
        # Check quality thresholds
        thresholds = self.quality_thresholds[processing_level]
        quality = self._assess_quality_preservation(processed_audio, original_audio)
        artifacts = self._assess_artifacts(processed_audio, original_audio)
        
        # If quality is below threshold, apply quality restoration
        if quality < thresholds['min_quality'] or artifacts > thresholds['max_artifacts']:
            logger.warning(f"Quality below threshold: {quality:.3f} (min: {thresholds['min_quality']:.3f})")
            
            try:
                # Apply quality restoration techniques
                processed_audio = self._restore_audio_quality(processed_audio, original_audio, sr)
                
                # Re-assess quality
                new_quality = self._assess_quality_preservation(processed_audio, original_audio)
                new_artifacts = self._assess_artifacts(processed_audio, original_audio)
                
                logger.info(f"Quality after restoration: {new_quality:.3f}")
                
            except Exception as e:
                logger.warning(f"Quality restoration failed: {e}")
        
        # Final validation
        processed_audio = self._final_validation(processed_audio, original_audio)
        
        return processed_audio
    
    def _restore_audio_quality(self, processed: np.ndarray, original: np.ndarray, sr: int) -> np.ndarray:
        """Restore audio quality using advanced techniques."""
        # Simple quality restoration - could be much more sophisticated
        
        # Spectral envelope restoration
        try:
            import librosa
            
            # Get spectral envelopes
            orig_stft = librosa.stft(original, n_fft=2048, hop_length=512)
            proc_stft = librosa.stft(processed, n_fft=2048, hop_length=512)
            
            orig_mag = np.abs(orig_stft)
            proc_mag = np.abs(proc_stft)
            proc_phase = np.angle(proc_stft)
            
            # Restore spectral envelope while preserving modifications
            alpha = 0.3  # Blend factor
            restored_mag = (1 - alpha) * proc_mag + alpha * orig_mag
            
            # Reconstruct
            restored_stft = restored_mag * np.exp(1j * proc_phase)
            restored_audio = librosa.istft(restored_stft, hop_length=512)
            
            # Ensure same length
            if len(restored_audio) != len(original):
                if len(restored_audio) > len(original):
                    restored_audio = restored_audio[:len(original)]
                else:
                    restored_audio = np.pad(restored_audio, (0, len(original) - len(restored_audio)), mode='constant')
            
            return restored_audio
            
        except Exception as e:
            logger.warning(f"Spectral restoration failed: {e}")
            return processed
    
    def _final_validation(self, processed: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Final validation and safety checks."""
        # Check for silence
        if not np.any(np.abs(processed) > 1e-10):
            logger.error("Processed audio is silent, returning original")
            return original
        
        # Check for clipping
        if np.max(np.abs(processed)) > 1.0:
            processed = processed / np.max(np.abs(processed)) * 0.95
            logger.warning("Applied anti-clipping normalization")
        
        # Check for NaN/inf
        if np.any(~np.isfinite(processed)):
            processed = np.nan_to_num(processed, nan=0.0, posinf=0.95, neginf=-0.95)
            logger.warning("Cleaned NaN/inf values")
        
        # Ensure same length
        if len(processed) != len(original):
            if len(processed) > len(original):
                processed = processed[:len(original)]
            else:
                processed = np.pad(processed, (0, len(original) - len(processed)), mode='constant')
        
        return processed
    
    def _assess_quality_preservation(self, processed: np.ndarray, original: np.ndarray) -> float:
        """Assess quality preservation using multiple metrics."""
        try:
            # Simple correlation-based metric
            if len(processed) != len(original):
                min_len = min(len(processed), len(original))
                processed = processed[:min_len]
                original = original[:min_len]
            
            correlation = np.corrcoef(processed.flatten(), original.flatten())[0, 1]
            quality = max(0, correlation)
            
            return quality
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5
    
    def _assess_artifacts(self, processed: np.ndarray, original: np.ndarray) -> float:
        """Assess artifacts using spectral analysis."""
        try:
            # Simple RMS difference
            if len(processed) != len(original):
                min_len = min(len(processed), len(original))
                processed = processed[:min_len]
                original = original[:min_len]
            
            diff = processed - original
            rms_diff = np.sqrt(np.mean(diff**2))
            rms_original = np.sqrt(np.mean(original**2))
            
            artifacts = min(rms_diff / (rms_original + 1e-10), 1.0)
            
            return artifacts
            
        except Exception as e:
            logger.warning(f"Artifact assessment failed: {e}")
            return 0.0
    
    def _generate_processing_report(self, input_path: str, output_path: str):
        """Generate detailed processing report."""
        report_path = output_path.replace('.wav', '_report.txt').replace('.mp3', '_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("Next-Generation AI Audio Fingerprint Removal Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Input File: {input_path}\n")
            f.write(f"Output File: {output_path}\n")
            f.write(f"Processing Level: {self.stats.processing_level}\n\n")
            
            f.write("Processing Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Time: {self.stats.total_time:.2f}s\n")
            f.write(f"Detection Time: {self.stats.detection_time:.2f}s\n")
            f.write(f"Removal Time: {self.stats.removal_time:.2f}s\n")
            f.write(f"Optimization Time: {self.stats.optimization_time:.2f}s\n\n")
            
            f.write("Detection Results:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Total Watermarks Detected: {self.stats.watermarks_detected}\n")
            f.write(f"Steganographic Patterns: {self.stats.steganography_detected}\n")
            f.write(f"Neural Patterns: {self.stats.neural_patterns_detected}\n")
            f.write(f"Watermarks Removed: {self.stats.watermarks_removed}\n\n")
            
            f.write("Quality Assessment:\n")
            f.write("-" * 18 + "\n")
            f.write(f"Quality Preserved: {self.stats.quality_preserved:.3f}\n")
            f.write(f"Artifacts Introduced: {self.stats.artifacts_introduced:.3f}\n\n")
            
            f.write("Performance Metrics:\n")
            f.write("-" * 19 + "\n")
            f.write(f"Peak Memory Usage: {self.stats.memory_peak_gb:.2f}GB\n")
            f.write(f"GPU Acceleration: {'Yes' if self.stats.gpu_acceleration_used else 'No'}\n")
            f.write(f"File Size: {self.stats.file_size_mb:.1f}MB\n")
            f.write(f"Audio Duration: {self.stats.audio_duration_seconds:.1f}s\n")
        
        logger.info(f"Processing report saved to: {report_path}")

def main():
    """Test the integrated system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrated Next-Generation AI Audio Fingerprint Removal")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument("--level", choices=['gentle', 'balanced', 'aggressive', 'maximum'], 
                       default='balanced', help="Processing level")
    parser.add_argument("--no-advanced-detection", action="store_true", 
                       help="Disable advanced detection algorithms")
    parser.add_argument("--no-optimization", action="store_true", 
                       help="Disable performance optimizations")
    parser.add_argument("--report", action="store_true", help="Generate processing report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize integrated system
        system = IntegratedWatermarkRemover()
        
        # Process file
        stats = system.process_file(
            input_path=args.input,
            output_path=args.output,
            processing_level=args.level,
            enable_advanced_detection=not args.no_advanced_detection,
            enable_performance_optimization=not args.no_optimization,
            generate_report=args.report
        )
        
        # Display results
        print(f"\nNext-Generation Processing Complete!")
        print(f"=====================================")
        print(f"Processing Time: {stats.total_time:.2f}s")
        print(f"Watermarks Detected: {stats.watermarks_detected}")
        print(f"Watermarks Removed: {stats.watermarks_removed}")
        print(f"Quality Preserved: {stats.quality_preserved:.3f}")
        print(f"Artifacts: {stats.artifacts_introduced:.3f}")
        print(f"Memory Peak: {stats.memory_peak_gb:.2f}GB")
        print(f"GPU Acceleration: {'Yes' if stats.gpu_acceleration_used else 'No'}")
        
        if args.report:
            print(f"Report generated: {args.output.replace('.wav', '_report.txt').replace('.mp3', '_report.txt')}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
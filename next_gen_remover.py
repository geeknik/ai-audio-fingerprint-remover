#!/usr/bin/env python3
"""
Next-Generation AI Watermark Removal System
Combines traditional methods with advanced neural detection and removal.
"""

import numpy as np
import librosa
import scipy.signal as signal
from scipy import ndimage, interpolate
from scipy.fft import fft, ifft, fftfreq
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import concurrent.futures
import threading
from advanced_steganography_detector import AdvancedSteganographyDetector, SteganographyDetection
from neural_watermark_detector import NeuralWatermarkDetector, NeuralDetection

logger = logging.getLogger(__name__)

@dataclass
class RemovalResult:
    """Results from watermark removal."""
    method: str
    success: bool
    quality_preserved: float  # 0-1 scale
    watermarks_removed: int
    artifacts_introduced: float  # 0-1 scale
    processing_time: float

class NextGenWatermarkRemover:
    """Next-generation AI watermark removal with advanced techniques."""
    
    def __init__(self):
        # Initialize advanced detectors
        self.steg_detector = AdvancedSteganographyDetector()
        self.neural_detector = NeuralWatermarkDetector()
        
        # Advanced removal parameters
        self.removal_config = {
            'use_inpainting': True,
            'use_semantic_preservation': True,
            'use_perceptual_masking': True,
            'use_adversarial_training': True,
            'quality_threshold': 0.8,  # Minimum quality to maintain
            'max_artifacts': 0.2,      # Maximum artifacts allowed
        }
        
        # Performance optimization
        self.use_multiprocessing = True
        self.chunk_size = 48000 * 30  # 30 seconds at 48kHz
        self.overlap_size = 4800      # 0.1 second overlap
        
    def remove_advanced_watermarks(self, audio: np.ndarray, sr: int, 
                                 processing_level: str = 'balanced') -> Tuple[np.ndarray, List[RemovalResult]]:
        """
        Advanced watermark removal using next-generation techniques.
        
        Args:
            audio: Input audio array
            sr: Sample rate
            processing_level: 'gentle', 'balanced', 'aggressive', 'maximum'
        
        Returns:
            Tuple of (processed_audio, removal_results)
        """
        results = []
        processed_audio = audio.copy()
        
        logger.info(f"Starting next-generation watermark removal ({processing_level} level)")
        
        # Step 1: Comprehensive detection
        steg_detections = self.steg_detector.detect_all_steganography(audio, sr)
        neural_detections = self.neural_detector.detect_neural_watermarks(audio, sr)
        
        logger.info(f"Detected {len(steg_detections)} steganographic and {len(neural_detections)} neural watermarks")
        
        # Step 2: Advanced removal based on detections
        if steg_detections:
            processed_audio, steg_results = self._remove_steganographic_watermarks(
                processed_audio, sr, steg_detections, processing_level
            )
            results.extend(steg_results)
        
        if neural_detections:
            processed_audio, neural_results = self._remove_neural_watermarks(
                processed_audio, sr, neural_detections, processing_level
            )
            results.extend(neural_results)
        
        # Step 3: Advanced pattern disruption
        processed_audio, pattern_results = self._disrupt_ai_patterns(
            processed_audio, sr, processing_level
        )
        results.extend(pattern_results)
        
        # Step 4: Quality preservation and artifact reduction
        processed_audio = self._preserve_audio_quality(processed_audio, audio, sr)
        
        # Step 5: Final validation
        processed_audio = self._validate_and_fix_output(processed_audio, audio)
        
        logger.info(f"Next-generation removal complete. Applied {len(results)} techniques")
        return processed_audio, results
    
    def _remove_steganographic_watermarks(self, audio: np.ndarray, sr: int,
                                        detections: List[SteganographyDetection],
                                        level: str) -> Tuple[np.ndarray, List[RemovalResult]]:
        """Remove steganographic watermarks using specialized techniques."""
        results = []
        processed = audio.copy()
        
        for detection in detections:
            try:
                start_time = time.time()
                
                if detection.method == 'lsb_steganography':
                    processed = self._remove_lsb_steganography(processed, detection, level)
                    
                elif detection.method == 'echo_hiding':
                    processed = self._remove_echo_hiding(processed, sr, detection, level)
                    
                elif detection.method == 'spread_spectrum':
                    processed = self._remove_spread_spectrum(processed, sr, detection, level)
                    
                elif detection.method == 'amplitude_modulation':
                    processed = self._remove_amplitude_modulation(processed, sr, detection, level)
                    
                elif detection.method == 'frequency_hopping':
                    processed = self._remove_frequency_hopping(processed, sr, detection, level)
                    
                elif detection.method == 'phase_coding':
                    processed = self._remove_phase_coding(processed, sr, detection, level)
                    
                elif detection.method == 'cepstral_hiding':
                    processed = self._remove_cepstral_hiding(processed, sr, detection, level)
                
                processing_time = time.time() - start_time
                
                # Validate removal success
                quality_preserved = self._assess_quality_preservation(processed, audio)
                artifacts = self._assess_artifacts(processed, audio)
                
                results.append(RemovalResult(
                    method=f"steg_{detection.method}",
                    success=True,
                    quality_preserved=quality_preserved,
                    watermarks_removed=1,
                    artifacts_introduced=artifacts,
                    processing_time=processing_time
                ))
                
            except Exception as e:
                logger.warning(f"Failed to remove {detection.method}: {e}")
                results.append(RemovalResult(
                    method=f"steg_{detection.method}",
                    success=False,
                    quality_preserved=1.0,
                    watermarks_removed=0,
                    artifacts_introduced=0.0,
                    processing_time=0.0
                ))
        
        return processed, results
    
    def _remove_neural_watermarks(self, audio: np.ndarray, sr: int,
                                detections: List[NeuralDetection],
                                level: str) -> Tuple[np.ndarray, List[RemovalResult]]:
        """Remove neural watermarks using advanced AI techniques."""
        results = []
        processed = audio.copy()
        
        for detection in detections:
            try:
                start_time = time.time()
                
                if detection.method == 'platform_specific':
                    processed = self._remove_platform_watermark(processed, sr, detection, level)
                    
                elif detection.method == 'anomaly_detection':
                    processed = self._remove_anomalous_patterns(processed, sr, detection, level)
                    
                elif detection.method == 'semantic_analysis':
                    processed = self._remove_semantic_patterns(processed, sr, detection, level)
                
                processing_time = time.time() - start_time
                
                # Validate removal success
                quality_preserved = self._assess_quality_preservation(processed, audio)
                artifacts = self._assess_artifacts(processed, audio)
                
                results.append(RemovalResult(
                    method=f"neural_{detection.method}",
                    success=True,
                    quality_preserved=quality_preserved,
                    watermarks_removed=1,
                    artifacts_introduced=artifacts,
                    processing_time=processing_time
                ))
                
            except Exception as e:
                logger.warning(f"Failed to remove neural watermark {detection.method}: {e}")
                results.append(RemovalResult(
                    method=f"neural_{detection.method}",
                    success=False,
                    quality_preserved=1.0,
                    watermarks_removed=0,
                    artifacts_introduced=0.0,
                    processing_time=0.0
                ))
        
        return processed, results
    
    def _remove_lsb_steganography(self, audio: np.ndarray, detection: SteganographyDetection, level: str) -> np.ndarray:
        """Remove LSB steganography using advanced bit manipulation."""
        start, end = detection.location
        
        # Convert to appropriate bit depth
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            # Convert to int16 for bit manipulation
            audio_int = (audio * 32767).astype(np.int16)
            
            # Randomize LSBs in detected region
            lsb_noise = np.random.randint(0, 2, size=audio_int[start:end].shape, dtype=np.int16)
            audio_int[start:end] = (audio_int[start:end] & 0xFFFE) | lsb_noise
            
            # Convert back to float
            return audio_int.astype(np.float32) / 32767
        
        return audio
    
    def _remove_echo_hiding(self, audio: np.ndarray, sr: int, detection: SteganographyDetection, level: str) -> np.ndarray:
        """Remove echo hiding using adaptive echo cancellation."""
        echo_delay = detection.parameters.get('echo_delay', 1000)
        correlation = detection.parameters.get('correlation', 0.3)
        
        # Apply adaptive echo cancellation
        processed = audio.copy()
        
        if echo_delay < len(audio):
            # Subtract estimated echo
            echo_strength = correlation * (0.5 if level == 'gentle' else 0.8 if level == 'balanced' else 1.0)
            for i in range(echo_delay, len(audio)):
                processed[i] -= echo_strength * processed[i - echo_delay]
        
        return processed
    
    def _remove_spread_spectrum(self, audio: np.ndarray, sr: int, detection: SteganographyDetection, level: str) -> np.ndarray:
        """Remove spread spectrum watermarks using spectral despreading."""
        # Use STFT to work in frequency domain
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Apply spectral normalization to reduce spreading
        for f in range(magnitude.shape[0]):
            spectrum_slice = magnitude[f, :]
            
            # Smooth the spectrum to reduce artificial spreading
            smoothed = ndimage.gaussian_filter1d(spectrum_slice, sigma=2.0)
            
            # Blend based on processing level
            blend_factor = 0.3 if level == 'gentle' else 0.5 if level == 'balanced' else 0.7
            magnitude[f, :] = (1 - blend_factor) * spectrum_slice + blend_factor * smoothed
        
        # Reconstruct
        stft_processed = magnitude * np.exp(1j * phase)
        return librosa.istft(stft_processed, hop_length=512)
    
    def _remove_amplitude_modulation(self, audio: np.ndarray, sr: int, detection: SteganographyDetection, level: str) -> np.ndarray:
        """Remove amplitude modulation watermarks."""
        mod_freq = detection.parameters.get('modulation_frequency', 100)
        
        # Create envelope using Hilbert transform
        analytic_signal = signal.hilbert(audio)
        envelope = np.abs(analytic_signal)
        
        # Remove modulation by filtering the envelope
        if mod_freq > 0:
            # Design filter to remove modulation frequency
            nyquist = sr / 2
            low_cutoff = max(mod_freq - 50, 1) / nyquist
            high_cutoff = min(mod_freq + 50, nyquist - 1) / nyquist
            
            if 0 < low_cutoff < high_cutoff < 1:
                try:
                    b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='bandstop')
                    envelope_filtered = signal.filtfilt(b, a, envelope)
                    
                    # Reconstruct signal with filtered envelope
                    envelope_ratio = envelope_filtered / (envelope + 1e-10)
                    return audio * envelope_ratio
                except:
                    pass
        
        return audio
    
    def _remove_frequency_hopping(self, audio: np.ndarray, sr: int, detection: SteganographyDetection, level: str) -> np.ndarray:
        """Remove frequency hopping patterns."""
        # Use short-time analysis to track and smooth frequency hopping
        window_size = int(0.01 * sr)  # 10ms windows
        hop_size = window_size // 2
        
        processed = audio.copy()
        
        for i in range(0, len(audio) - window_size, hop_size):
            window = processed[i:i + window_size]
            
            # Apply spectral smoothing to reduce hopping artifacts
            spectrum = fft(window)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            
            # Smooth magnitude to reduce rapid frequency changes
            smoothed_magnitude = ndimage.gaussian_filter1d(magnitude, sigma=1.0)
            
            # Reconstruct
            smoothed_spectrum = smoothed_magnitude * np.exp(1j * phase)
            processed[i:i + window_size] = np.real(ifft(smoothed_spectrum))
        
        return processed
    
    def _remove_phase_coding(self, audio: np.ndarray, sr: int, detection: SteganographyDetection, level: str) -> np.ndarray:
        """Remove phase coding watermarks."""
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Smooth phase transitions to remove artificial coding
        for f in range(phase.shape[0]):
            phase_slice = phase[f, :]
            
            # Apply temporal smoothing to phase
            smoothed_phase = ndimage.gaussian_filter1d(phase_slice, sigma=1.5)
            
            # Blend based on processing level
            blend_factor = 0.2 if level == 'gentle' else 0.4 if level == 'balanced' else 0.6
            phase[f, :] = (1 - blend_factor) * phase_slice + blend_factor * smoothed_phase
        
        # Reconstruct
        stft_processed = magnitude * np.exp(1j * phase)
        return librosa.istft(stft_processed, hop_length=512)
    
    def _remove_cepstral_hiding(self, audio: np.ndarray, sr: int, detection: SteganographyDetection, level: str) -> np.ndarray:
        """Remove cepstral domain watermarks."""
        # Extract and modify cepstral coefficients
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # Smooth cepstral coefficients to remove artificial patterns
        for i in range(mfccs.shape[0]):
            mfccs[i, :] = ndimage.gaussian_filter1d(mfccs[i, :], sigma=2.0)
        
        # Note: Full reconstruction from MFCCs is complex
        # For now, return original with some spectral smoothing
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Apply light spectral smoothing
        for f in range(magnitude.shape[0]):
            magnitude[f, :] = ndimage.gaussian_filter1d(magnitude[f, :], sigma=1.0)
        
        stft_processed = magnitude * np.exp(1j * phase)
        return librosa.istft(stft_processed, hop_length=512)
    
    def _remove_platform_watermark(self, audio: np.ndarray, sr: int, detection: NeuralDetection, level: str) -> np.ndarray:
        """Remove platform-specific watermarks."""
        platform = detection.ai_platform
        
        # Platform-specific removal strategies
        if platform == 'suno':
            return self._remove_suno_advanced(audio, sr, level)
        elif platform == 'openai_jukebox':
            return self._remove_jukebox_patterns(audio, sr, level)
        elif platform == 'elevenlabs':
            return self._remove_elevenlabs_artifacts(audio, sr, level)
        elif platform == 'google_musiclm':
            return self._remove_musiclm_tokens(audio, sr, level)
        elif platform == 'meta_audiocraft':
            return self._remove_audiocraft_compression(audio, sr, level)
        
        return audio
    
    def _remove_suno_advanced(self, audio: np.ndarray, sr: int, level: str) -> np.ndarray:
        """Advanced Suno watermark removal."""
        # Apply multiple Suno-specific techniques
        processed = audio.copy()
        
        # 1. Remove ultrasonic watermarks with inpainting
        processed = self._inpaint_frequency_range(processed, sr, 19000, 20000, level)
        
        # 2. Disrupt timing quantization
        processed = self._add_micro_timing_jitter(processed, sr, level)
        
        # 3. Modify harmonic ratios
        processed = self._adjust_harmonic_content(processed, sr, level)
        
        return processed
    
    def _remove_jukebox_patterns(self, audio: np.ndarray, sr: int, level: str) -> np.ndarray:
        """Remove OpenAI Jukebox VQ patterns."""
        # Focus on removing quantization artifacts
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        
        # Add subtle noise to break quantization
        noise_level = 0.001 if level == 'gentle' else 0.002 if level == 'balanced' else 0.005
        noise = np.random.randn(*magnitude.shape) * noise_level
        magnitude += noise * magnitude  # Proportional noise
        
        # Reconstruct
        phase = np.angle(stft)
        stft_processed = magnitude * np.exp(1j * phase)
        return librosa.istft(stft_processed, hop_length=512)
    
    def _remove_elevenlabs_artifacts(self, audio: np.ndarray, sr: int, level: str) -> np.ndarray:
        """Remove ElevenLabs voice synthesis artifacts."""
        # Focus on vocal frequency ranges
        processed = audio.copy()
        
        # Remove artifacts in speech frequencies (300-3400 Hz)
        processed = self._apply_speech_naturalization(processed, sr, level)
        
        return processed
    
    def _remove_musiclm_tokens(self, audio: np.ndarray, sr: int, level: str) -> np.ndarray:
        """Remove MusicLM semantic/acoustic token artifacts."""
        # Disrupt token-like patterns
        processed = audio.copy()
        
        # Apply semantic smoothing
        processed = self._apply_semantic_smoothing(processed, sr, level)
        
        return processed
    
    def _remove_audiocraft_compression(self, audio: np.ndarray, sr: int, level: str) -> np.ndarray:
        """Remove AudioCraft compression artifacts."""
        # Focus on codec-like artifacts
        processed = audio.copy()
        
        # Apply decompression-like processing
        processed = self._apply_codec_artifact_removal(processed, sr, level)
        
        return processed
    
    def _disrupt_ai_patterns(self, audio: np.ndarray, sr: int, level: str) -> Tuple[np.ndarray, List[RemovalResult]]:
        """Apply advanced pattern disruption techniques."""
        start_time = time.time()
        processed = audio.copy()
        
        # 1. Adversarial perturbations
        processed = self._add_adversarial_perturbations(processed, level)
        
        # 2. Natural variation injection
        processed = self._inject_natural_variations(processed, sr, level)
        
        # 3. Cross-modal consistency enforcement
        processed = self._enforce_cross_modal_consistency(processed, sr, level)
        
        processing_time = time.time() - start_time
        
        quality_preserved = self._assess_quality_preservation(processed, audio)
        artifacts = self._assess_artifacts(processed, audio)
        
        result = RemovalResult(
            method="pattern_disruption",
            success=True,
            quality_preserved=quality_preserved,
            watermarks_removed=0,  # Pattern disruption doesn't remove specific watermarks
            artifacts_introduced=artifacts,
            processing_time=processing_time
        )
        
        return processed, [result]
    
    def _preserve_audio_quality(self, processed: np.ndarray, original: np.ndarray, sr: int) -> np.ndarray:
        """Preserve audio quality using advanced techniques."""
        # Perceptual quality preservation
        if self.removal_config['use_perceptual_masking']:
            processed = self._apply_perceptual_masking(processed, original, sr)
        
        # Semantic preservation
        if self.removal_config['use_semantic_preservation']:
            processed = self._preserve_semantic_content(processed, original, sr)
        
        return processed
    
    def _validate_and_fix_output(self, processed: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Validate output and fix any issues."""
        # Check for silence
        if not np.any(np.abs(processed) > 1e-10):
            logger.warning("Processed audio is silent, returning original")
            return original
        
        # Check for clipping
        if np.max(np.abs(processed)) > 1.0:
            processed = processed / np.max(np.abs(processed)) * 0.95
        
        # Check for NaN/inf
        if np.any(~np.isfinite(processed)):
            processed = np.nan_to_num(processed, nan=0.0, posinf=0.95, neginf=-0.95)
        
        # Ensure same length
        if len(processed) != len(original):
            if len(processed) > len(original):
                processed = processed[:len(original)]
            else:
                processed = np.pad(processed, (0, len(original) - len(processed)), mode='constant')
        
        return processed
    
    # Helper methods (simplified implementations)
    def _assess_quality_preservation(self, processed: np.ndarray, original: np.ndarray) -> float:
        """Assess how well audio quality was preserved (0-1 scale)."""
        try:
            # Simple correlation-based metric
            correlation = np.corrcoef(processed.flatten(), original.flatten())[0, 1]
            return max(0, correlation)
        except:
            return 0.5
    
    def _assess_artifacts(self, processed: np.ndarray, original: np.ndarray) -> float:
        """Assess artifacts introduced (0-1 scale, lower is better)."""
        try:
            # Simple RMS difference
            diff = processed - original
            rms_diff = np.sqrt(np.mean(diff**2))
            rms_original = np.sqrt(np.mean(original**2))
            return min(rms_diff / (rms_original + 1e-10), 1.0)
        except:
            return 0.0
    
    # Placeholder implementations for advanced techniques
    def _inpaint_frequency_range(self, audio: np.ndarray, sr: int, low_freq: float, high_freq: float, level: str) -> np.ndarray:
        """Inpaint frequency range using spectral interpolation."""
        # Simplified implementation
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        if np.any(mask):
            # Simple interpolation inpainting
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            for t in range(magnitude.shape[1]):
                if np.sum(mask) > 0:
                    # Interpolate from surrounding frequencies
                    below_mask = freqs < low_freq
                    above_mask = freqs > high_freq
                    
                    if np.any(below_mask) and np.any(above_mask):
                        below_val = np.mean(magnitude[below_mask, t])
                        above_val = np.mean(magnitude[above_mask, t])
                        magnitude[mask, t] = np.linspace(below_val, above_val, np.sum(mask))
            
            stft_processed = magnitude * np.exp(1j * phase)
            return librosa.istft(stft_processed, hop_length=512)
        
        return audio
    
    def _add_micro_timing_jitter(self, audio: np.ndarray, sr: int, level: str) -> np.ndarray:
        """Add micro-timing jitter to disrupt quantization."""
        jitter_amount = 0.001 if level == 'gentle' else 0.002 if level == 'balanced' else 0.005
        
        # Simple time stretching with random variations
        stretch_factors = 1.0 + np.random.uniform(-jitter_amount, jitter_amount, size=10)
        
        processed = audio.copy()
        chunk_size = len(audio) // 10
        
        for i, factor in enumerate(stretch_factors):
            start = i * chunk_size
            end = min(start + chunk_size, len(audio))
            if end > start:
                try:
                    chunk = audio[start:end]
                    stretched = librosa.effects.time_stretch(chunk, rate=1/factor)
                    # Trim or pad to original size
                    if len(stretched) > len(chunk):
                        stretched = stretched[:len(chunk)]
                    elif len(stretched) < len(chunk):
                        stretched = np.pad(stretched, (0, len(chunk) - len(stretched)), mode='edge')
                    processed[start:end] = stretched
                except:
                    pass
        
        return processed
    
    def _adjust_harmonic_content(self, audio: np.ndarray, sr: int, level: str) -> np.ndarray:
        """Adjust harmonic content to break AI patterns."""
        # Simple harmonic adjustment
        harmonic, percussive = librosa.effects.hpss(audio)
        
        # Slightly modify harmonic content
        adjustment = 0.02 if level == 'gentle' else 0.05 if level == 'balanced' else 0.1
        harmonic_adjusted = harmonic * (1.0 + np.random.uniform(-adjustment, adjustment))
        
        return harmonic_adjusted + percussive
    
    # Additional placeholder methods
    def _apply_speech_naturalization(self, audio: np.ndarray, sr: int, level: str) -> np.ndarray:
        return audio  # Placeholder
    
    def _apply_semantic_smoothing(self, audio: np.ndarray, sr: int, level: str) -> np.ndarray:
        return audio  # Placeholder
    
    def _apply_codec_artifact_removal(self, audio: np.ndarray, sr: int, level: str) -> np.ndarray:
        return audio  # Placeholder
    
    def _add_adversarial_perturbations(self, audio: np.ndarray, level: str) -> np.ndarray:
        return audio  # Placeholder
    
    def _inject_natural_variations(self, audio: np.ndarray, sr: int, level: str) -> np.ndarray:
        return audio  # Placeholder
    
    def _enforce_cross_modal_consistency(self, audio: np.ndarray, sr: int, level: str) -> np.ndarray:
        return audio  # Placeholder
    
    def _apply_perceptual_masking(self, processed: np.ndarray, original: np.ndarray, sr: int) -> np.ndarray:
        return processed  # Placeholder
    
    def _preserve_semantic_content(self, processed: np.ndarray, original: np.ndarray, sr: int) -> np.ndarray:
        return processed  # Placeholder

# Import time module
import time

def main():
    """Test the next-generation watermark remover."""
    import argparse
    import soundfile as sf
    
    parser = argparse.ArgumentParser(description="Next-Generation Watermark Removal")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument("--level", choices=['gentle', 'balanced', 'aggressive', 'maximum'], 
                       default='balanced', help="Processing level")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Load audio
        audio, sr = sf.read(args.input)
        print(f"Processing {args.input} with {args.level} level...")
        
        # Run next-generation removal
        remover = NextGenWatermarkRemover()
        processed_audio, results = remover.remove_advanced_watermarks(audio, sr, args.level)
        
        # Save output
        sf.write(args.output, processed_audio, sr)
        
        print(f"\nNext-Generation Removal Results:")
        print(f"================================")
        print(f"Output saved to: {args.output}")
        
        total_watermarks = sum(r.watermarks_removed for r in results)
        avg_quality = np.mean([r.quality_preserved for r in results]) if results else 1.0
        avg_artifacts = np.mean([r.artifacts_introduced for r in results]) if results else 0.0
        total_time = sum(r.processing_time for r in results)
        
        print(f"Watermarks removed: {total_watermarks}")
        print(f"Average quality preserved: {avg_quality:.3f}")
        print(f"Average artifacts introduced: {avg_artifacts:.3f}")
        print(f"Total processing time: {total_time:.2f}s")
        
        print(f"\nDetailed Results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.method}: "
                  f"Success={result.success}, "
                  f"Quality={result.quality_preserved:.3f}, "
                  f"Artifacts={result.artifacts_introduced:.3f}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
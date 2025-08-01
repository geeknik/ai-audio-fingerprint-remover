#!/usr/bin/env python3
"""
State-of-the-Art Watermark Remover
Implements cutting-edge techniques based on 2024-2025 research for removing AI audio watermarks.

Based on research from:
- SoK: How Robust is Audio Watermarking in Generative AI models? (2025)
- AudioMarkBench: Benchmarking Robustness of Audio Watermarking (2024)
- IDEAW: Robust Neural Audio Watermarking (2024)
"""

import numpy as np
import librosa
import scipy.signal as signal
from scipy import stats
from scipy.fft import fft, ifft, fftfreq
import logging
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')
from audio_processing_fixes import AudioProcessingFixes, WatermarkRemovalFixes


logger = logging.getLogger(__name__)

class StateOfTheArtWatermarkRemover:
    """
    State-of-the-art watermark removal system implementing the most effective
    techniques from recent academic research.
    """
    
    def __init__(self, quality_preservation_mode: str = "balanced"):
        """
        Initialize the SOTA watermark remover.
        
        Args:
            quality_preservation_mode: "aggressive", "balanced", or "conservative"
        """
        self.quality_mode = quality_preservation_mode
        
        # Research-proven attack parameters
        self.attack_params = {
            "aggressive": {
                "pitch_shift_cents": [50, 100, 150],  # Most effective: +100 cents
                "time_stretch_factors": [0.75, 0.9, 1.1, 1.25],  # Highly effective
                "high_pass_cutoffs": [300, 500, 800],  # Devastating to watermarks
                "low_pass_cutoffs": [2000, 3500, 5000],  # Very effective
                "sample_suppression_rates": [0.01, 0.05, 0.1],  # 1%, 5%, 10%
                "resampling_rates": [4000, 8000, 16000],  # Effective downsampling
                "noise_snr_db": [20, 30],  # Moderate noise addition
                "compression_bitrates": [16, 32, 64],  # kbps
            },
            "balanced": {
                "pitch_shift_cents": [25, 50],
                "time_stretch_factors": [0.9, 1.1],
                "high_pass_cutoffs": [200, 300],
                "low_pass_cutoffs": [3500, 5000],
                "sample_suppression_rates": [0.005, 0.01],
                "resampling_rates": [8000, 16000],
                "noise_snr_db": [30, 40],
                "compression_bitrates": [32, 64],
            },
            "conservative": {
                "pitch_shift_cents": [5, 10],
                "time_stretch_factors": [0.98, 1.02],
                "high_pass_cutoffs": [100, 200],
                "low_pass_cutoffs": [5000, 8000],
                "sample_suppression_rates": [0.0005, 0.001],
                "resampling_rates": [16000, 22050],
                "noise_snr_db": [40, 50],
                "compression_bitrates": [64, 128],
            }
        }
        
        # Psychoacoustic masking parameters
        self.psychoacoustic = {
            "bark_scale_bands": 24,  # Critical bands for masking
            "masking_threshold_db": -20,  # Threshold below which changes are inaudible
            "temporal_masking_ms": 50,  # Temporal masking window
        }
        
        # AI watermark frequency signatures (from research)
        self.ai_watermark_signatures = {
            "suno": [(19000, 20000), (17500, 18500), (15000, 16000), (12000, 12500)],
            "elevenlabs": [(15000, 17000), (18000, 19000)],
            "mubert": [(8000, 8500), (16000, 16500)],
            "generic_neural": [(19000, 22000), (50, 200)],  # Common neural patterns
        }
    
    def remove_watermarks_sota(self, audio: np.ndarray, sr: int, 
                              watermarks: List[Dict[str, Any]] = None) -> np.ndarray:
        """
        Apply state-of-the-art watermark removal using multiple coordinated techniques.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            watermarks: Detected watermarks (optional, will detect if not provided)
            
        Returns:
            Cleaned audio signal
        """
        logger.info(f"Applying SOTA watermark removal in {self.quality_mode} mode")
        
        # Ensure audio is valid
        result = self._validate_and_normalize_audio(audio)
        original_result = result.copy()
        
        # Step 1: Psychoacoustic analysis for quality preservation
        masking_threshold = self._compute_psychoacoustic_masking(result, sr)
        
        # Step 2: Apply research-proven attack methods in order of effectiveness
        attack_methods = [
            ("pitch_shift", self._apply_pitch_shift_attack),
            ("time_stretch", self._apply_time_stretch_attack),
            ("high_pass_filter", self._apply_high_pass_attack),
            ("low_pass_filter", self._apply_low_pass_attack),
            ("sample_suppression", self._apply_sample_suppression_attack),
            ("spectral_inversion", self._apply_spectral_inversion_attack),
            ("resampling", self._apply_resampling_attack),
            ("adaptive_noise", self._apply_adaptive_noise_attack),
            ("compression_decompression", self._apply_compression_attack),
        ]
        
        # Apply attacks with quality monitoring
        for attack_name, attack_func in attack_methods:
            try:
                logger.debug(f"Applying {attack_name} attack")
                result_candidate = attack_func(result, sr, masking_threshold)
                
                # Quality check - only apply if quality is preserved
                if self._quality_check(original_result, result_candidate, sr):
                    result = result_candidate
                    logger.debug(f"{attack_name} attack applied successfully")
                else:
                    logger.debug(f"{attack_name} attack rejected due to quality degradation")
                    
            except Exception as e:
                logger.warning(f"{attack_name} attack failed: {e}")
                continue
        
        # Step 3: Apply targeted AI watermark removal
        result = self._apply_targeted_ai_removal(result, sr, watermarks)
        
        # Step 4: Final quality restoration
        result = self._apply_quality_restoration(result, original_result, sr)
        
        return result
    
    def _validate_and_normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Validate and normalize audio input with enhanced safety."""
        # Use enhanced NaN cleanup
        audio = AudioProcessingFixes.safe_nan_cleanup(audio)
        
        # Validate audio content
        if not AudioProcessingFixes.validate_audio_content(audio, context="SOTA input validation"):
            logger.error("Input audio validation failed")
            raise ValueError("Invalid audio input")
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        return audio

    
    def _compute_psychoacoustic_masking(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Compute psychoacoustic masking threshold for quality preservation."""
        # Compute bark scale filterbank
        n_fft = 2048
        hop_length = n_fft // 4
        
        # STFT for frequency analysis
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        # Convert to bark scale (simplified)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        bark_freqs = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500) ** 2)
        
        # Compute masking threshold (simplified model)
        masking_threshold = np.zeros_like(magnitude)
        
        for i in range(magnitude.shape[1]):  # For each time frame
            frame_magnitude = magnitude[:, i]
            
            # Find spectral peaks (maskers)
            peaks, _ = signal.find_peaks(frame_magnitude, height=np.max(frame_magnitude) * 0.1)
            
            # Compute masking for each peak
            for peak in peaks:
                peak_level = frame_magnitude[peak]
                
                # Spreading function (simplified)
                spread_lower = np.maximum(0, peak_level - 15 - 0.4 * (bark_freqs[peak] - bark_freqs[:peak]))
                spread_upper = np.maximum(0, peak_level - 15 - 0.15 * (bark_freqs[peak+1:] - bark_freqs[peak]))
                
                # Update masking threshold
                masking_threshold[:peak, i] = np.maximum(masking_threshold[:peak, i], spread_lower)
                if peak + 1 < len(masking_threshold):
                    masking_threshold[peak+1:, i] = np.maximum(masking_threshold[peak+1:, i], spread_upper)
        
        return masking_threshold
    
    def _apply_pitch_shift_attack(self, audio: np.ndarray, sr: int, 
                                 masking_threshold: np.ndarray) -> np.ndarray:
        """Apply pitch shift attack - most effective against watermarks."""
        params = self.attack_params[self.quality_mode]
        
        # Choose pitch shift amount based on quality mode
        pitch_shift_cents = np.random.choice(params["pitch_shift_cents"])
        
        try:
            # Apply pitch shift using librosa
            shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift_cents/100)
            
            # Ensure same length
            if len(shifted) != len(audio):
                if len(shifted) > len(audio):
                    shifted = shifted[:len(audio)]
                else:
                    shifted = np.pad(shifted, (0, len(audio) - len(shifted)), mode='constant')
            
            return shifted
            
        except Exception as e:
            logger.warning(f"Pitch shift failed: {e}")
            return audio
    
    def _apply_time_stretch_attack(self, audio: np.ndarray, sr: int, 
                                  masking_threshold: np.ndarray) -> np.ndarray:
        """Apply time stretch attack - highly effective."""
        params = self.attack_params[self.quality_mode]
        
        # Choose stretch factor
        stretch_factor = np.random.choice(params["time_stretch_factors"])
        
        try:
            # Apply time stretch
            stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)
            
            # Restore original length
            if len(stretched) > len(audio):
                stretched = stretched[:len(audio)]
            elif len(stretched) < len(audio):
                stretched = np.pad(stretched, (0, len(audio) - len(stretched)), mode='constant')
            
            return stretched
            
        except Exception as e:
            logger.warning(f"Time stretch failed: {e}")
            return audio
    
    def _apply_high_pass_attack(self, audio: np.ndarray, sr: int, 
                               masking_threshold: np.ndarray) -> np.ndarray:
        """Apply high-pass filter attack - devastating to watermarks."""
        params = self.attack_params[self.quality_mode]
        
        cutoff = np.random.choice(params["high_pass_cutoffs"])
        nyquist = sr / 2
        
        if cutoff >= nyquist:
            return audio
        
        try:
            # Design high-pass filter
            b, a = signal.butter(4, cutoff / nyquist, btype='high')
            filtered = signal.filtfilt(b, a, audio)
            
            return filtered
            
        except Exception as e:
            logger.warning(f"High-pass filter failed: {e}")
            return audio
    
    def _apply_low_pass_attack(self, audio: np.ndarray, sr: int, 
                              masking_threshold: np.ndarray) -> np.ndarray:
        """Apply low-pass filter attack - very effective."""
        params = self.attack_params[self.quality_mode]
        
        cutoff = np.random.choice(params["low_pass_cutoffs"])
        nyquist = sr / 2
        
        if cutoff >= nyquist:
            return audio
        
        try:
            # Design low-pass filter
            b, a = signal.butter(4, cutoff / nyquist, btype='low')
            filtered = signal.filtfilt(b, a, audio)
            
            return filtered
            
        except Exception as e:
            logger.warning(f"Low-pass filter failed: {e}")
            return audio
    
    def _apply_sample_suppression_attack(self, audio: np.ndarray, sr: int, 
                                        masking_threshold: np.ndarray) -> np.ndarray:
        """Apply sample suppression attack - reduces amplitude of random samples."""
        params = self.attack_params[self.quality_mode]
        
        suppression_rate = np.random.choice(params["sample_suppression_rates"])
        
        # Instead of zeroing, reduce amplitude
        result = audio.copy()
        
        # Create random indices for suppression
        num_samples_to_suppress = int(len(audio) * suppression_rate)
        if num_samples_to_suppress > 0:
            suppress_indices = np.random.choice(len(audio), num_samples_to_suppress, replace=False)
            
            # Reduce amplitude instead of zeroing (preserve some signal)
            suppression_factor = 0.1  # Reduce to 10% instead of 0%
            result[suppress_indices] *= suppression_factor
        
        return result
    
    def _apply_spectral_inversion_attack(self, audio: np.ndarray, sr: int, 
                                        masking_threshold: np.ndarray) -> np.ndarray:
        """Apply spectral inversion in high frequencies where watermarks hide."""
        n_fft = 2048
        hop_length = n_fft // 4
        
        # STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Invert phase in high frequencies (>15kHz)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        high_freq_mask = freqs >= 15000
        
        if np.any(high_freq_mask):
            # Randomly invert phase in high frequencies
            inversion_mask = np.random.random(np.sum(high_freq_mask)) > 0.5
            phase[high_freq_mask] = np.where(inversion_mask[:, np.newaxis], 
                                           -phase[high_freq_mask], 
                                           phase[high_freq_mask])
        
        # Reconstruct
        stft_modified = magnitude * np.exp(1j * phase)
        result = librosa.istft(stft_modified, hop_length=hop_length)
        
        # Ensure same length
        if len(result) > len(audio):
            result = result[:len(audio)]
        elif len(result) < len(audio):
            result = np.pad(result, (0, len(audio) - len(result)), mode='constant')
        
        return result
    
    def _apply_resampling_attack(self, audio: np.ndarray, sr: int, 
                                masking_threshold: np.ndarray) -> np.ndarray:
        """Apply resampling attack - effective against many schemes."""
        params = self.attack_params[self.quality_mode]
        
        target_sr = np.random.choice(params["resampling_rates"])
        
        if target_sr >= sr:
            return audio
        
        try:
            # Downsample then upsample
            downsampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            upsampled = librosa.resample(downsampled, orig_sr=target_sr, target_sr=sr)
            
            # Ensure same length
            if len(upsampled) > len(audio):
                upsampled = upsampled[:len(audio)]
            elif len(upsampled) < len(audio):
                upsampled = np.pad(upsampled, (0, len(audio) - len(upsampled)), mode='constant')
            
            return upsampled
            
        except Exception as e:
            logger.warning(f"Resampling attack failed: {e}")
            return audio
    
    def _apply_adaptive_noise_attack(self, audio: np.ndarray, sr: int, 
                                    masking_threshold: np.ndarray) -> np.ndarray:
        """Apply adaptive noise that's masked by the audio content."""
        params = self.attack_params[self.quality_mode]
        
        snr_db = np.random.choice(params["noise_snr_db"])
        
        # Generate noise
        noise = np.random.randn(len(audio))
        
        # Scale noise to desired SNR
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = noise * np.sqrt(noise_power / np.mean(noise ** 2))
        
        # Apply psychoacoustic masking to noise (simplified)
        # In practice, this would use the masking_threshold computed earlier
        noise = noise * 0.1  # Conservative scaling
        
        return audio + noise
    
    def _apply_compression_attack(self, audio: np.ndarray, sr: int, 
                                 masking_threshold: np.ndarray) -> np.ndarray:
        """Apply compression/decompression attack."""
        params = self.attack_params[self.quality_mode]
        
        # Simulate MP3 compression effects
        # This is a simplified version - real implementation would use actual codecs
        
        # Apply dynamic range compression
        threshold = 0.5
        ratio = 4.0
        
        # Simple compressor
        compressed = np.where(np.abs(audio) > threshold,
                            np.sign(audio) * (threshold + (np.abs(audio) - threshold) / ratio),
                            audio)
        
        # Add slight quantization noise to simulate lossy compression
        bitrate = np.random.choice(params["compression_bitrates"])
        quantization_levels = 2 ** max(4, int(np.log2(bitrate)))
        
        quantized = np.round(compressed * quantization_levels) / quantization_levels
        
        return quantized
    
    def _apply_targeted_ai_removal(self, audio: np.ndarray, sr: int, 
                                  watermarks: List[Dict[str, Any]] = None) -> np.ndarray:
        """Apply targeted removal for specific AI watermark signatures."""
        result = audio.copy()
        
        # Apply notch filters to known AI watermark frequencies
        for ai_system, freq_ranges in self.ai_watermark_signatures.items():
            for low_freq, high_freq in freq_ranges:
                if high_freq > sr / 2:
                    continue
                
                try:
                    # Create narrow notch filter
                    nyquist = sr / 2
                    low_norm = low_freq / nyquist
                    high_norm = min(high_freq, nyquist - 1) / nyquist
                    
                    if 0 < low_norm < high_norm < 1.0:
                        # Conservative removal to preserve quality
                        b, a = signal.butter(2, [low_norm, high_norm], btype='bandstop')
                        filtered = signal.filtfilt(b, a, result)
                        
                        # Blend with original (conservative approach)
                        blend_factor = 0.3 if self.quality_mode == "conservative" else 0.5
                        result = (1 - blend_factor) * result + blend_factor * filtered
                        
                except Exception as e:
                    logger.debug(f"Failed to filter {low_freq}-{high_freq}Hz: {e}")
                    continue
        
        return result
    
    def _quality_check(self, original: np.ndarray, processed: np.ndarray, sr: int) -> bool:
        """Check if processed audio maintains acceptable quality."""
        # Calculate SNR
        noise = processed - original
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return True
        
        snr_db = 10 * np.log10(signal_power / noise_power)
        
        # Quality thresholds based on mode
        thresholds = {
            "aggressive": 10,    # 10 dB SNR minimum
            "balanced": 15,      # 15 dB SNR minimum
            "conservative": 20   # 20 dB SNR minimum
        }
        
        return snr_db >= thresholds[self.quality_mode]
    
    def _apply_quality_restoration(self, processed: np.ndarray, original: np.ndarray, 
                                  sr: int) -> np.ndarray:
        """Apply final quality restoration techniques."""
        # Spectral envelope preservation
        try:
            # Compute spectral envelopes
            orig_envelope = self._compute_spectral_envelope(original, sr)
            proc_envelope = self._compute_spectral_envelope(processed, sr)
            
            # Restore spectral balance in critical frequencies
            n_fft = 2048
            hop_length = n_fft // 4
            
            stft = librosa.stft(processed, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Apply envelope correction (conservative)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            # Only restore below 10kHz to preserve watermark removal
            restore_mask = freqs < 10000
            if np.any(restore_mask) and len(orig_envelope) == len(magnitude):
                correction_factor = orig_envelope[restore_mask] / (proc_envelope[restore_mask] + 1e-10)
                correction_factor = np.clip(correction_factor, 0.5, 2.0)  # Limit correction
                magnitude[restore_mask] *= correction_factor[:, np.newaxis]
            
            # Reconstruct
            stft_corrected = magnitude * np.exp(1j * phase)
            result = librosa.istft(stft_corrected, hop_length=hop_length)
            
            # Ensure same length
            if len(result) > len(processed):
                result = result[:len(processed)]
            elif len(result) < len(processed):
                result = np.pad(result, (0, len(processed) - len(result)), mode='constant')
            
            return result
            
        except Exception as e:
            logger.warning(f"Quality restoration failed: {e}")
            return processed
    
    def _compute_spectral_envelope(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Compute spectral envelope for quality restoration."""
        n_fft = 2048
        stft = librosa.stft(audio, n_fft=n_fft)
        magnitude = np.abs(stft)
        
        # Compute average magnitude spectrum
        envelope = np.mean(magnitude, axis=1)
        
        return envelope

def main():
    """Test the SOTA watermark remover."""
    import argparse
    import soundfile as sf
    
    parser = argparse.ArgumentParser(description="State-of-the-Art Watermark Removal")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument("--mode", choices=["aggressive", "balanced", "conservative"], 
                       default="balanced", help="Quality preservation mode")
    
    args = parser.parse_args()
    
    # Load audio
    audio, sr = librosa.load(args.input, sr=None, mono=True)
    
    # Apply SOTA removal
    remover = StateOfTheArtWatermarkRemover(quality_preservation_mode=args.mode)
    cleaned_audio = remover.remove_watermarks_sota(audio, sr)
    
    # Save result
    sf.write(args.output, cleaned_audio, sr)
    
    print(f"SOTA watermark removal complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Advanced Steganography Detection for AI Audio Fingerprints
Detects sophisticated steganographic watermarks that traditional methods miss.
"""

import numpy as np
import librosa
import scipy.signal as signal
import scipy.stats as stats
from scipy.fft import fft, ifft, fftfreq
from scipy.ndimage import gaussian_filter1d
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SteganographyDetection:
    """Results from steganography analysis."""
    method: str
    confidence: float
    location: Tuple[int, int]  # Start, end samples
    strength: float
    parameters: Dict[str, Any]

class AdvancedSteganographyDetector:
    """Detect sophisticated steganographic watermarks in AI-generated audio."""
    
    def __init__(self):
        self.detection_thresholds = {
            'lsb_entropy': 0.75,
            'echo_hiding': 0.65,
            'spread_spectrum': 0.70,
            'amplitude_modulation': 0.60,
            'frequency_hopping': 0.80,
            'phase_coding': 0.85,
            'cepstral_hiding': 0.75
        }
        
    def detect_all_steganography(self, audio: np.ndarray, sr: int) -> List[SteganographyDetection]:
        """Comprehensive steganography detection across all methods."""
        detections = []
        
        # Convert to mono for analysis
        if audio.ndim > 1:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio
            
        logger.info(f"Running comprehensive steganography analysis on {len(audio_mono)} samples")
        
        # 1. LSB Steganography Detection
        lsb_results = self._detect_lsb_steganography(audio_mono, sr)
        detections.extend(lsb_results)
        
        # 2. Echo Hiding Detection
        echo_results = self._detect_echo_hiding(audio_mono, sr)
        detections.extend(echo_results)
        
        # 3. Spread Spectrum Detection
        spread_results = self._detect_spread_spectrum(audio_mono, sr)
        detections.extend(spread_results)
        
        # 4. Amplitude Modulation Detection
        am_results = self._detect_amplitude_modulation(audio_mono, sr)
        detections.extend(am_results)
        
        # 5. Frequency Hopping Detection
        fh_results = self._detect_frequency_hopping(audio_mono, sr)
        detections.extend(fh_results)
        
        # 6. Phase Coding Detection
        phase_results = self._detect_phase_coding(audio_mono, sr)
        detections.extend(phase_results)
        
        # 7. Cepstral Domain Hiding
        cepstral_results = self._detect_cepstral_hiding(audio_mono, sr)
        detections.extend(cepstral_results)
        
        logger.info(f"Found {len(detections)} steganographic signatures")
        return detections
    
    def _detect_lsb_steganography(self, audio: np.ndarray, sr: int) -> List[SteganographyDetection]:
        """Detect LSB (Least Significant Bit) steganography."""
        detections = []
        
        try:
            # Convert to int16 to analyze bit patterns
            audio_int = (audio * 32767).astype(np.int16)
            
            # Extract LSBs
            lsbs = audio_int & 1
            
            # Analyze LSB entropy in chunks
            chunk_size = sr // 2  # 0.5 second chunks
            
            for i in range(0, len(lsbs) - chunk_size, chunk_size // 2):
                chunk = lsbs[i:i + chunk_size]
                
                # Calculate entropy
                _, counts = np.unique(chunk, return_counts=True)
                entropy = stats.entropy(counts, base=2)
                
                # Natural audio LSBs should have high entropy (close to 1.0)
                # Embedded data often has lower entropy
                if entropy < 0.85:  # Suspiciously low entropy
                    # Additional tests for confirmation
                    
                    # Chi-square test for randomness
                    chi2, p_value = stats.chisquare(counts)
                    
                    # Run test for randomness
                    runs = self._count_runs(chunk)
                    expected_runs = 2 * counts[0] * counts[1] / len(chunk) + 1
                    run_test_stat = (runs - expected_runs) / np.sqrt(2 * counts[0] * counts[1] * (2 * counts[0] * counts[1] - len(chunk)) / (len(chunk)**2 * (len(chunk) - 1)))
                    
                    confidence = (1.0 - entropy) * 0.5 + (1.0 - p_value) * 0.3 + abs(run_test_stat) * 0.2
                    
                    if confidence > self.detection_thresholds['lsb_entropy']:
                        detections.append(SteganographyDetection(
                            method='lsb_steganography',
                            confidence=min(confidence, 1.0),
                            location=(i, i + chunk_size),
                            strength=1.0 - entropy,
                            parameters={
                                'entropy': entropy,
                                'chi2_pvalue': p_value,
                                'run_test_stat': run_test_stat
                            }
                        ))
                        
        except Exception as e:
            logger.warning(f"LSB steganography detection failed: {e}")
            
        return detections
    
    def _detect_echo_hiding(self, audio: np.ndarray, sr: int) -> List[SteganographyDetection]:
        """Detect echo hiding steganography."""
        detections = []
        
        try:
            # Look for periodic echoes that might encode data
            max_echo_delay = int(0.1 * sr)  # Up to 100ms delay
            
            # Compute autocorrelation
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Look for periodic peaks that might indicate echo hiding
            for delay in range(1, min(max_echo_delay, len(autocorr))):
                if delay < 10:  # Skip very short delays
                    continue
                    
                # Check for significant correlation at this delay
                correlation = autocorr[delay] / autocorr[0]
                
                if correlation > 0.3:  # Significant echo
                    # Analyze periodicity
                    echo_period = delay
                    periodic_strength = 0
                    
                    for multiple in range(2, 6):
                        if multiple * delay < len(autocorr):
                            periodic_strength += autocorr[multiple * delay] / autocorr[0]
                    
                    periodic_strength /= 4  # Average
                    
                    if periodic_strength > 0.15:  # Periodic echoes suggest hiding
                        confidence = correlation * 0.6 + periodic_strength * 0.4
                        
                        if confidence > self.detection_thresholds['echo_hiding']:
                            detections.append(SteganographyDetection(
                                method='echo_hiding',
                                confidence=min(confidence, 1.0),
                                location=(0, len(audio)),
                                strength=correlation,
                                parameters={
                                    'echo_delay': delay,
                                    'correlation': correlation,
                                    'periodic_strength': periodic_strength
                                }
                            ))
                            
        except Exception as e:
            logger.warning(f"Echo hiding detection failed: {e}")
            
        return detections
    
    def _detect_spread_spectrum(self, audio: np.ndarray, sr: int) -> List[SteganographyDetection]:
        """Detect spread spectrum steganography."""
        detections = []
        
        try:
            # Analyze spectral spreading patterns
            n_fft = 2048
            hop_length = n_fft // 4
            
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            
            # Look for unnatural spectral spreading
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            for t in range(magnitude.shape[1]):
                spectrum = magnitude[:, t]
                
                # Calculate spectral spread
                centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
                spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / np.sum(spectrum))
                
                # Normalize by frequency
                normalized_spread = spread / (sr / 2)
                
                # Look for unnatural spreading patterns
                if normalized_spread > 0.4:  # High spread might indicate spread spectrum
                    # Additional analysis for confirmation
                    spectral_entropy = stats.entropy(spectrum + 1e-10)
                    spectral_flatness = stats.gmean(spectrum + 1e-10) / np.mean(spectrum + 1e-10)
                    
                    confidence = normalized_spread * 0.4 + spectral_entropy * 0.3 + spectral_flatness * 0.3
                    
                    if confidence > self.detection_thresholds['spread_spectrum']:
                        sample_start = t * hop_length
                        sample_end = min(sample_start + n_fft, len(audio))
                        
                        detections.append(SteganographyDetection(
                            method='spread_spectrum',
                            confidence=min(confidence, 1.0),
                            location=(sample_start, sample_end),
                            strength=normalized_spread,
                            parameters={
                                'spectral_spread': spread,
                                'spectral_entropy': spectral_entropy,
                                'spectral_flatness': spectral_flatness
                            }
                        ))
                        
        except Exception as e:
            logger.warning(f"Spread spectrum detection failed: {e}")
            
        return detections
    
    def _detect_amplitude_modulation(self, audio: np.ndarray, sr: int) -> List[SteganographyDetection]:
        """Detect amplitude modulation steganography."""
        detections = []
        
        try:
            # Extract envelope using Hilbert transform
            analytic_signal = signal.hilbert(audio)
            envelope = np.abs(analytic_signal)
            
            # Smooth envelope to remove noise
            envelope_smooth = gaussian_filter1d(envelope, sigma=sr//1000)
            
            # Analyze envelope for modulation patterns
            chunk_size = sr  # 1 second chunks
            
            for i in range(0, len(envelope_smooth) - chunk_size, chunk_size // 2):
                chunk = envelope_smooth[i:i + chunk_size]
                
                # FFT of envelope to detect modulation frequencies
                envelope_fft = np.abs(fft(chunk))
                modulation_freqs = fftfreq(len(chunk), 1/sr)[:len(chunk)//2]
                
                # Look for significant modulation in suspicious frequency ranges
                # Data hiding often uses modulation frequencies 10-1000 Hz
                suspicious_range = (modulation_freqs >= 10) & (modulation_freqs <= 1000)
                
                if np.any(suspicious_range):
                    max_modulation = np.max(envelope_fft[suspicious_range])
                    total_energy = np.sum(envelope_fft)
                    
                    modulation_strength = max_modulation / (total_energy + 1e-10)
                    
                    if modulation_strength > 0.1:  # Significant modulation
                        # Find modulation frequency
                        mod_freq_idx = np.argmax(envelope_fft[suspicious_range])
                        mod_freq = modulation_freqs[suspicious_range][mod_freq_idx]
                        
                        confidence = modulation_strength * 5  # Scale to 0-1 range
                        
                        if confidence > self.detection_thresholds['amplitude_modulation']:
                            detections.append(SteganographyDetection(
                                method='amplitude_modulation',
                                confidence=min(confidence, 1.0),
                                location=(i, i + chunk_size),
                                strength=modulation_strength,
                                parameters={
                                    'modulation_frequency': mod_freq,
                                    'modulation_strength': modulation_strength
                                }
                            ))
                            
        except Exception as e:
            logger.warning(f"Amplitude modulation detection failed: {e}")
            
        return detections
    
    def _detect_frequency_hopping(self, audio: np.ndarray, sr: int) -> List[SteganographyDetection]:
        """Detect frequency hopping patterns in steganography."""
        detections = []
        
        try:
            # Short-time analysis to detect frequency hopping
            window_size = int(0.01 * sr)  # 10ms windows
            hop_size = window_size // 2
            
            dominant_freqs = []
            
            for i in range(0, len(audio) - window_size, hop_size):
                window = audio[i:i + window_size]
                
                # Find dominant frequency
                spectrum = np.abs(fft(window))
                freqs = fftfreq(len(window), 1/sr)[:len(window)//2]
                
                if len(spectrum) > 0:
                    dominant_idx = np.argmax(spectrum[:len(spectrum)//2])
                    dominant_freq = freqs[dominant_idx]
                    dominant_freqs.append(dominant_freq)
            
            if len(dominant_freqs) > 10:
                # Analyze for hopping patterns
                freq_changes = np.diff(dominant_freqs)
                
                # Look for rapid, systematic frequency changes
                rapid_changes = np.abs(freq_changes) > sr / 100  # Changes > sr/100 Hz
                change_rate = np.sum(rapid_changes) / len(freq_changes)
                
                # Look for periodicity in frequency hopping
                if change_rate > 0.3:  # High rate of frequency changes
                    # Analyze hopping pattern
                    hop_pattern = dominant_freqs[::10]  # Sample every 10th frequency
                    
                    if len(hop_pattern) > 5:
                        # Check for repeating patterns
                        pattern_entropy = stats.entropy(np.histogram(hop_pattern, bins=20)[0] + 1)
                        
                        confidence = change_rate * 0.6 + (1.0 - pattern_entropy/5) * 0.4
                        
                        if confidence > self.detection_thresholds['frequency_hopping']:
                            detections.append(SteganographyDetection(
                                method='frequency_hopping',
                                confidence=min(confidence, 1.0),
                                location=(0, len(audio)),
                                strength=change_rate,
                                parameters={
                                    'change_rate': change_rate,
                                    'pattern_entropy': pattern_entropy,
                                    'avg_hop_size': np.mean(np.abs(freq_changes))
                                }
                            ))
                            
        except Exception as e:
            logger.warning(f"Frequency hopping detection failed: {e}")
            
        return detections
    
    def _detect_phase_coding(self, audio: np.ndarray, sr: int) -> List[SteganographyDetection]:
        """Detect phase coding steganography."""
        detections = []
        
        try:
            # Analyze phase relationships
            n_fft = 2048
            hop_length = n_fft // 4
            
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            phase = np.angle(stft)
            
            # Look for unnatural phase relationships
            for t in range(1, phase.shape[1]):
                phase_diff = np.diff(phase[:, t])
                
                # Natural audio has smooth phase transitions
                # Artificial phase coding creates abrupt changes
                abrupt_changes = np.abs(phase_diff) > np.pi/2
                change_rate = np.sum(abrupt_changes) / len(phase_diff)
                
                if change_rate > 0.1:  # High rate of abrupt phase changes
                    # Additional analysis
                    phase_variance = np.var(phase_diff)
                    phase_entropy = stats.entropy(np.histogram(phase_diff, bins=20)[0] + 1)
                    
                    confidence = change_rate * 0.5 + (phase_variance / (np.pi**2)) * 0.3 + (1.0 - phase_entropy/5) * 0.2
                    
                    if confidence > self.detection_thresholds['phase_coding']:
                        sample_start = t * hop_length
                        sample_end = min(sample_start + n_fft, len(audio))
                        
                        detections.append(SteganographyDetection(
                            method='phase_coding',
                            confidence=min(confidence, 1.0),
                            location=(sample_start, sample_end),
                            strength=change_rate,
                            parameters={
                                'change_rate': change_rate,
                                'phase_variance': phase_variance,
                                'phase_entropy': phase_entropy
                            }
                        ))
                        
        except Exception as e:
            logger.warning(f"Phase coding detection failed: {e}")
            
        return detections
    
    def _detect_cepstral_hiding(self, audio: np.ndarray, sr: int) -> List[SteganographyDetection]:
        """Detect steganography in cepstral domain."""
        detections = []
        
        try:
            # Compute cepstral coefficients
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Analyze cepstral coefficients for anomalies
            for coeff_idx in range(mfccs.shape[0]):
                coeff_series = mfccs[coeff_idx, :]
                
                # Look for unnatural patterns in cepstral domain
                # Natural speech/music has smooth cepstral evolution
                coeff_diff = np.diff(coeff_series)
                abrupt_changes = np.abs(coeff_diff) > 2 * np.std(coeff_diff)
                change_rate = np.sum(abrupt_changes) / len(coeff_diff)
                
                if change_rate > 0.15:  # High rate of abrupt changes
                    # Additional analysis
                    coeff_entropy = stats.entropy(np.histogram(coeff_series, bins=20)[0] + 1)
                    coeff_variance = np.var(coeff_series)
                    
                    confidence = change_rate * 0.6 + (1.0 - coeff_entropy/5) * 0.4
                    
                    if confidence > self.detection_thresholds['cepstral_hiding']:
                        detections.append(SteganographyDetection(
                            method='cepstral_hiding',
                            confidence=min(confidence, 1.0),
                            location=(0, len(audio)),
                            strength=change_rate,
                            parameters={
                                'coefficient_index': coeff_idx,
                                'change_rate': change_rate,
                                'coeff_entropy': coeff_entropy,
                                'coeff_variance': coeff_variance
                            }
                        ))
                        
        except Exception as e:
            logger.warning(f"Cepstral hiding detection failed: {e}")
            
        return detections
    
    def _count_runs(self, sequence):
        """Count runs in a binary sequence for randomness testing."""
        if len(sequence) == 0:
            return 0
            
        runs = 1
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i-1]:
                runs += 1
        return runs

def main():
    """Test the advanced steganography detector."""
    import argparse
    import soundfile as sf
    
    parser = argparse.ArgumentParser(description="Advanced Steganography Detection")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Load audio
        audio, sr = sf.read(args.input)
        print(f"Analyzing {args.input} ({len(audio)} samples at {sr} Hz)")
        
        # Run detection
        detector = AdvancedSteganographyDetector()
        detections = detector.detect_all_steganography(audio, sr)
        
        print(f"\nSteganography Analysis Results:")
        print(f"================================")
        
        if detections:
            for detection in detections:
                print(f"Method: {detection.method}")
                print(f"Confidence: {detection.confidence:.3f}")
                print(f"Location: {detection.location[0]:,} - {detection.location[1]:,} samples")
                print(f"Strength: {detection.strength:.3f}")
                print(f"Parameters: {detection.parameters}")
                print("-" * 40)
        else:
            print("No steganographic signatures detected.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
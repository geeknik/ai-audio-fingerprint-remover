#!/usr/bin/env python3
"""
Advanced Watermark Analysis Tool
Analyzes audio files for sophisticated watermarking patterns that might be missed by the main tool.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import stats
import argparse
import os
from typing import Dict, List, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedWatermarkAnalyzer:
    """Advanced analyzer for detecting sophisticated watermarking techniques."""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_spectral_anomalies(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Detect spectral anomalies that could indicate watermarks."""
        results = {}
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)  # Average across channels
        
        # Compute spectrogram with adaptive resolution
        nperseg = min(4096, len(audio) // 4)  # Ensure nperseg is reasonable
        nperseg = max(256, nperseg)  # Minimum window size
        noverlap = nperseg // 2
        
        freqs, times, spectrogram = signal.spectrogram(
            audio, fs=sr, window='hann', nperseg=nperseg, 
            noverlap=noverlap, scaling='spectrum'
        )
        
        # Convert to dB
        spec_db = 10 * np.log10(spectrogram + 1e-10)
        
        # Analyze frequency bands for anomalies
        results['frequency_analysis'] = self._analyze_frequency_bands(freqs, spec_db, sr)
        
        # Look for periodic patterns in time domain
        results['temporal_patterns'] = self._detect_temporal_patterns(spec_db, times)
        
        # Analyze spectral entropy variations
        results['entropy_analysis'] = self._analyze_spectral_entropy(spec_db)
        
        # Check for hidden carriers in high frequencies
        results['high_freq_carriers'] = self._detect_high_freq_carriers(freqs, spec_db, sr)
        
        return results
    
    def _analyze_frequency_bands(self, freqs: np.ndarray, spec_db: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze specific frequency bands for watermark signatures."""
        results = {}
        
        # Define suspicious frequency ranges (expanded from current implementation)
        suspicious_ranges = [
            (50, 200, "Low frequency steganography"),
            (8000, 8500, "Mid-range markers"),
            (12000, 12500, "Secondary watermark range"),
            (15000, 17000, "ElevenLabs/similar range"),
            (17000, 19000, "Extended high-freq range"),
            (19000, 20000, "Ultrasonic range"),
            (20000, 22000, "Extended ultrasonic"),
            (22000, 24000, "Maximum frequency range")
        ]
        
        for low_freq, high_freq, description in suspicious_ranges:
            if high_freq > sr / 2:
                continue
                
            # Find frequency indices
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if not np.any(freq_mask):
                continue
            
            band_energy = spec_db[freq_mask]
            
            # Calculate statistics
            mean_energy = np.mean(band_energy)
            std_energy = np.std(band_energy)
            
            # Look for anomalies
            anomaly_score = 0
            
            # Check for unusually constant energy
            if std_energy < 5:  # Very low variation in dB
                anomaly_score += 0.3
            
            # Check for periodic patterns
            if band_energy.shape[1] > 10:
                autocorr = np.correlate(np.mean(band_energy, axis=0), 
                                      np.mean(band_energy, axis=0), mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                peaks, _ = signal.find_peaks(autocorr, height=0.5 * np.max(autocorr))
                if len(peaks) > 3:
                    anomaly_score += 0.4
            
            # Check for energy spikes
            energy_threshold = mean_energy + 2 * std_energy
            spikes = np.sum(band_energy > energy_threshold) / band_energy.size
            if spikes > 0.1:  # More than 10% of points are spikes
                anomaly_score += 0.3
            
            results[f"{low_freq}-{high_freq}Hz"] = {
                'description': description,
                'mean_energy': float(mean_energy),
                'std_energy': float(std_energy),
                'anomaly_score': float(anomaly_score),
                'suspicious': anomaly_score > 0.5
            }
        
        return results
    
    def _detect_temporal_patterns(self, spec_db: np.ndarray, times: np.ndarray) -> Dict[str, Any]:
        """Detect temporal patterns that could indicate watermarks."""
        results = {}
        
        # Analyze energy variations over time
        total_energy = np.mean(spec_db, axis=0)
        
        # Look for periodic patterns in total energy
        if len(total_energy) > 20:
            autocorr = np.correlate(total_energy, total_energy, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks in autocorrelation
            peaks, properties = signal.find_peaks(autocorr, height=0.3 * np.max(autocorr), distance=5)
            
            if len(peaks) > 2:
                # Calculate period regularity
                periods = np.diff(peaks)
                period_regularity = np.std(periods) / np.mean(periods) if np.mean(periods) > 0 else 1
                
                results['periodic_energy'] = {
                    'detected': True,
                    'num_peaks': len(peaks),
                    'period_regularity': float(period_regularity),
                    'suspicious': period_regularity < 0.2  # Very regular periods
                }
            else:
                results['periodic_energy'] = {'detected': False}
        
        # Check for sudden energy changes (could indicate watermark insertion points)
        energy_diff = np.diff(total_energy)
        sudden_changes = np.where(np.abs(energy_diff) > 3 * np.std(energy_diff))[0]
        
        results['sudden_changes'] = {
            'count': len(sudden_changes),
            'suspicious': len(sudden_changes) > len(total_energy) * 0.05  # More than 5% sudden changes
        }
        
        return results
    
    def _analyze_spectral_entropy(self, spec_db: np.ndarray) -> Dict[str, Any]:
        """Analyze spectral entropy for watermark detection."""
        results = {}
        
        # Calculate entropy for each time frame
        entropies = []
        for t in range(spec_db.shape[1]):
            frame = spec_db[:, t]
            # Normalize to probability distribution
            frame_norm = frame - np.min(frame)
            if np.sum(frame_norm) > 0:
                prob_dist = frame_norm / np.sum(frame_norm)
                # Calculate entropy
                entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))
                entropies.append(entropy)
        
        if entropies:
            entropies = np.array(entropies)
            
            # Look for entropy anomalies
            mean_entropy = np.mean(entropies)
            std_entropy = np.std(entropies)
            
            # Low entropy regions might indicate watermarks
            low_entropy_threshold = mean_entropy - 2 * std_entropy
            low_entropy_frames = np.sum(entropies < low_entropy_threshold)
            
            # Very high entropy might indicate added noise
            high_entropy_threshold = mean_entropy + 2 * std_entropy
            high_entropy_frames = np.sum(entropies > high_entropy_threshold)
            
            results = {
                'mean_entropy': float(mean_entropy),
                'std_entropy': float(std_entropy),
                'low_entropy_frames': int(low_entropy_frames),
                'high_entropy_frames': int(high_entropy_frames),
                'low_entropy_ratio': float(low_entropy_frames / len(entropies)),
                'high_entropy_ratio': float(high_entropy_frames / len(entropies)),
                'suspicious_low': low_entropy_frames / len(entropies) > 0.1,
                'suspicious_high': high_entropy_frames / len(entropies) > 0.1
            }
        
        return results
    
    def _detect_high_freq_carriers(self, freqs: np.ndarray, spec_db: np.ndarray, sr: int) -> Dict[str, Any]:
        """Detect potential carrier signals in high frequencies."""
        results = {}
        
        # Focus on frequencies above 15kHz
        high_freq_mask = freqs >= 15000
        if not np.any(high_freq_mask):
            return {'detected': False, 'reason': 'No high frequency data available'}
        
        high_freq_spec = spec_db[high_freq_mask]
        high_freqs = freqs[high_freq_mask]
        
        # Look for narrow-band signals (potential carriers)
        carriers = []
        for i, freq in enumerate(high_freqs):
            freq_energy = high_freq_spec[i]
            
            # Check if this frequency has consistently high energy
            mean_energy = np.mean(freq_energy)
            std_energy = np.std(freq_energy)
            
            # A carrier would have high mean energy and low variation
            if mean_energy > -40 and std_energy < 5:  # Strong, stable signal
                carriers.append({
                    'frequency': float(freq),
                    'mean_energy': float(mean_energy),
                    'stability': float(1 / (std_energy + 0.1))  # Higher = more stable
                })
        
        results = {
            'detected': len(carriers) > 0,
            'carriers': carriers,
            'suspicious': len(carriers) > 2  # Multiple carriers suggest watermarking
        }
        
        return results
    
    def analyze_phase_patterns(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze phase patterns that could indicate watermarks."""
        results = {}
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)  # Average across channels
        
        # Compute STFT for phase analysis
        nperseg = min(2048, len(audio) // 4)
        nperseg = max(256, nperseg)
        freqs, times, stft = signal.stft(audio, fs=sr, nperseg=nperseg)
        
        # Extract phase information
        phases = np.angle(stft)
        
        # Look for phase anomalies in high frequencies
        high_freq_mask = freqs >= 15000
        if np.any(high_freq_mask):
            high_freq_phases = phases[high_freq_mask]
            
            # Check for phase coherence (watermarks might have structured phase)
            phase_coherence = []
            for i in range(high_freq_phases.shape[0]):
                phase_diff = np.diff(high_freq_phases[i])
                # Wrap phase differences to [-π, π]
                phase_diff = np.angle(np.exp(1j * phase_diff))
                coherence = 1 - np.std(phase_diff) / np.pi
                phase_coherence.append(coherence)
            
            results['high_freq_phase_coherence'] = {
                'mean_coherence': float(np.mean(phase_coherence)),
                'max_coherence': float(np.max(phase_coherence)),
                'suspicious': np.max(phase_coherence) > 0.8  # Very coherent phase
            }
        
        return results
    
    def generate_report(self, audio_path: str) -> str:
        """Generate a comprehensive watermark analysis report."""
        logger.info(f"Analyzing {audio_path}")
        
        # Load audio
        try:
            audio, sr = librosa.load(audio_path, sr=None, mono=False)
        except Exception as e:
            return f"Error loading audio: {e}"
        
        # Perform all analyses
        spectral_results = self.analyze_spectral_anomalies(audio, sr)
        phase_results = self.analyze_phase_patterns(audio, sr)
        
        # Generate report
        report = f"Advanced Watermark Analysis Report\n"
        report += f"=" * 50 + "\n"
        report += f"File: {os.path.basename(audio_path)}\n"
        report += f"Sample Rate: {sr} Hz\n"
        # Calculate duration properly for stereo/mono
        if len(audio.shape) > 1:
            duration = audio.shape[1] / sr
        else:
            duration = len(audio) / sr
        report += f"Duration: {duration:.2f} seconds\n\n"
        
        # Frequency band analysis
        report += "Frequency Band Analysis:\n"
        report += "-" * 25 + "\n"
        for band, data in spectral_results['frequency_analysis'].items():
            status = "SUSPICIOUS" if data['suspicious'] else "Normal"
            report += f"{band}: {data['description']} - {status}\n"
            report += f"  Anomaly Score: {data['anomaly_score']:.2f}\n"
            report += f"  Mean Energy: {data['mean_energy']:.1f} dB\n\n"
        
        # Temporal patterns
        report += "Temporal Pattern Analysis:\n"
        report += "-" * 26 + "\n"
        temporal = spectral_results['temporal_patterns']
        if temporal['periodic_energy']['detected']:
            status = "SUSPICIOUS" if temporal['periodic_energy']['suspicious'] else "Normal"
            report += f"Periodic Energy Patterns: {status}\n"
            report += f"  Number of peaks: {temporal['periodic_energy']['num_peaks']}\n"
            report += f"  Period regularity: {temporal['periodic_energy']['period_regularity']:.3f}\n\n"
        
        # High frequency carriers
        report += "High Frequency Carrier Analysis:\n"
        report += "-" * 32 + "\n"
        carriers = spectral_results['high_freq_carriers']
        if carriers['detected']:
            status = "SUSPICIOUS" if carriers['suspicious'] else "Normal"
            report += f"Carrier Detection: {status}\n"
            for carrier in carriers['carriers']:
                report += f"  {carrier['frequency']:.0f} Hz: {carrier['mean_energy']:.1f} dB (stability: {carrier['stability']:.2f})\n"
        else:
            report += "No high frequency carriers detected\n"
        
        report += "\n"
        
        # Phase analysis
        if 'high_freq_phase_coherence' in phase_results:
            report += "Phase Coherence Analysis:\n"
            report += "-" * 25 + "\n"
            phase_data = phase_results['high_freq_phase_coherence']
            status = "SUSPICIOUS" if phase_data['suspicious'] else "Normal"
            report += f"High Frequency Phase Coherence: {status}\n"
            report += f"  Max Coherence: {phase_data['max_coherence']:.3f}\n\n"
        
        # Overall assessment
        report += "Overall Assessment:\n"
        report += "-" * 18 + "\n"
        
        suspicious_indicators = 0
        total_indicators = 0
        
        # Count suspicious indicators
        for band_data in spectral_results['frequency_analysis'].values():
            total_indicators += 1
            if band_data['suspicious']:
                suspicious_indicators += 1
        
        if spectral_results['temporal_patterns']['periodic_energy'].get('suspicious', False):
            suspicious_indicators += 1
        total_indicators += 1
        
        if spectral_results['high_freq_carriers'].get('suspicious', False):
            suspicious_indicators += 1
        total_indicators += 1
        
        if phase_results.get('high_freq_phase_coherence', {}).get('suspicious', False):
            suspicious_indicators += 1
        total_indicators += 1
        
        suspicion_ratio = suspicious_indicators / total_indicators
        
        if suspicion_ratio > 0.5:
            assessment = "HIGH PROBABILITY of watermarking"
        elif suspicion_ratio > 0.3:
            assessment = "MODERATE PROBABILITY of watermarking"
        elif suspicion_ratio > 0.1:
            assessment = "LOW PROBABILITY of watermarking"
        else:
            assessment = "MINIMAL PROBABILITY of watermarking"
        
        report += f"{assessment}\n"
        report += f"Suspicious indicators: {suspicious_indicators}/{total_indicators}\n"
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Advanced Watermark Analysis Tool")
    parser.add_argument("input", help="Input audio file to analyze")
    parser.add_argument("--output", help="Output report file (optional)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return 1
    
    analyzer = AdvancedWatermarkAnalyzer()
    report = analyzer.generate_report(args.input)
    
    print(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport saved to {args.output}")

if __name__ == "__main__":
    main()
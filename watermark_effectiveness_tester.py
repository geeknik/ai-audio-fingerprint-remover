#!/usr/bin/env python3
"""
Watermark Effectiveness Tester
Comprehensive testing framework to validate watermark removal effectiveness.
"""

import numpy as np
import librosa
import scipy.signal as signal
from scipy import stats
import matplotlib.pyplot as plt
import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import argparse
from enhanced_suno_detector import SunoWatermarkDetector
from advanced_watermark_analysis import AdvancedWatermarkAnalyzer

logger = logging.getLogger(__name__)

class WatermarkEffectivenessTester:
    """Test and validate watermark removal effectiveness."""
    
    def __init__(self):
        self.suno_detector = SunoWatermarkDetector()
        self.analyzer = AdvancedWatermarkAnalyzer()
        self.test_results = {}
    
    def test_file_pair(self, original_path: str, processed_path: str) -> Dict[str, Any]:
        """Test effectiveness by comparing original and processed files."""
        logger.info(f"Testing effectiveness: {original_path} vs {processed_path}")
        
        # Load audio files
        try:
            original_audio, sr = librosa.load(original_path, sr=None, mono=False)
            processed_audio, _ = librosa.load(processed_path, sr=None, mono=False)
        except Exception as e:
            return {"error": f"Failed to load audio files: {e}"}
        
        # Convert to mono for analysis
        if len(original_audio.shape) > 1:
            original_mono = np.mean(original_audio, axis=0)
        else:
            original_mono = original_audio
            
        if len(processed_audio.shape) > 1:
            processed_mono = np.mean(processed_audio, axis=0)
        else:
            processed_mono = processed_audio
        
        # Ensure same length
        min_len = min(len(original_mono), len(processed_mono))
        original_mono = original_mono[:min_len]
        processed_mono = processed_mono[:min_len]
        
        results = {
            "file_info": {
                "original": original_path,
                "processed": processed_path,
                "sample_rate": sr,
                "duration": min_len / sr
            }
        }
        
        # 1. Watermark detection comparison
        results["watermark_detection"] = self._compare_watermark_detection(
            original_mono, processed_mono, sr
        )
        
        # 2. Spectral analysis
        results["spectral_analysis"] = self._compare_spectral_characteristics(
            original_mono, processed_mono, sr
        )
        
        # 3. Audio quality metrics
        results["quality_metrics"] = self._calculate_quality_metrics(
            original_mono, processed_mono, sr
        )
        
        # 4. Perceptual analysis
        results["perceptual_analysis"] = self._analyze_perceptual_changes(
            original_mono, processed_mono, sr
        )
        
        # 5. Overall effectiveness score
        results["effectiveness_score"] = self._calculate_effectiveness_score(results)
        
        return results
    
    def _compare_watermark_detection(self, original: np.ndarray, processed: np.ndarray, 
                                   sr: int) -> Dict[str, Any]:
        """Compare watermark detection between original and processed audio."""
        # Detect watermarks in both versions
        original_watermarks = self.suno_detector.detect_suno_watermarks(original, sr)
        processed_watermarks = self.suno_detector.detect_suno_watermarks(processed, sr)
        
        # Categorize watermarks by type
        def categorize_watermarks(watermarks):
            categories = {
                'neural': [],
                'frequency': [],
                'temporal': [],
                'phase': [],
                'statistical': []
            }
            
            for w in watermarks:
                w_type = w.get('type', '')
                if 'neural' in w_type:
                    categories['neural'].append(w)
                elif 'frequency' in w_type or 'carrier' in w_type:
                    categories['frequency'].append(w)
                elif 'temporal' in w_type:
                    categories['temporal'].append(w)
                elif 'phase' in w_type:
                    categories['phase'].append(w)
                elif 'statistical' in w_type or 'entropy' in w_type:
                    categories['statistical'].append(w)
            
            return categories
        
        original_categories = categorize_watermarks(original_watermarks)
        processed_categories = categorize_watermarks(processed_watermarks)
        
        # Calculate removal effectiveness by category
        removal_effectiveness = {}
        for category in original_categories:
            original_count = len(original_categories[category])
            processed_count = len(processed_categories[category])
            
            if original_count > 0:
                removal_rate = (original_count - processed_count) / original_count
                removal_effectiveness[category] = {
                    'original_count': original_count,
                    'processed_count': processed_count,
                    'removal_rate': removal_rate
                }
            else:
                removal_effectiveness[category] = {
                    'original_count': 0,
                    'processed_count': processed_count,
                    'removal_rate': 1.0 if processed_count == 0 else 0.0
                }
        
        return {
            'original_watermarks': len(original_watermarks),
            'processed_watermarks': len(processed_watermarks),
            'overall_removal_rate': (len(original_watermarks) - len(processed_watermarks)) / 
                                   max(len(original_watermarks), 1),
            'removal_by_category': removal_effectiveness,
            'remaining_high_confidence': len([w for w in processed_watermarks 
                                            if w.get('confidence', 0) > 0.7])
        }
    
    def _compare_spectral_characteristics(self, original: np.ndarray, processed: np.ndarray, 
                                        sr: int) -> Dict[str, Any]:
        """Compare spectral characteristics."""
        # Compute spectrograms
        n_fft = 2048
        hop_length = n_fft // 4
        
        original_stft = librosa.stft(original, n_fft=n_fft, hop_length=hop_length)
        processed_stft = librosa.stft(processed, n_fft=n_fft, hop_length=hop_length)
        
        original_mag = np.abs(original_stft)
        processed_mag = np.abs(processed_stft)
        
        # Frequency band analysis
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        frequency_bands = [
            (50, 200, "Low frequency"),
            (200, 2000, "Low-mid frequency"),
            (2000, 8000, "Mid frequency"),
            (8000, 15000, "High-mid frequency"),
            (15000, 20000, "High frequency"),
            (20000, sr//2, "Ultrasonic")
        ]
        
        band_analysis = {}
        for low_freq, high_freq, band_name in frequency_bands:
            if high_freq > sr // 2:
                continue
                
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if not np.any(freq_mask):
                continue
            
            original_band = original_mag[freq_mask]
            processed_band = processed_mag[freq_mask]
            
            # Calculate energy difference
            original_energy = np.mean(original_band ** 2)
            processed_energy = np.mean(processed_band ** 2)
            
            energy_change = (processed_energy - original_energy) / (original_energy + 1e-10)
            
            band_analysis[band_name] = {
                'frequency_range': [low_freq, high_freq],
                'original_energy': float(original_energy),
                'processed_energy': float(processed_energy),
                'energy_change_ratio': float(energy_change),
                'energy_reduction_db': float(10 * np.log10(
                    (original_energy + 1e-10) / (processed_energy + 1e-10)
                ))
            }
        
        return {
            'frequency_band_analysis': band_analysis,
            'overall_spectral_change': float(np.mean(np.abs(original_mag - processed_mag))),
            'spectral_correlation': float(np.corrcoef(
                original_mag.flatten(), processed_mag.flatten()
            )[0, 1])
        }
    
    def _calculate_quality_metrics(self, original: np.ndarray, processed: np.ndarray, 
                                  sr: int) -> Dict[str, Any]:
        """Calculate audio quality metrics."""
        # Ensure same length
        min_len = min(len(original), len(processed))
        original = original[:min_len]
        processed = processed[:min_len]
        
        # Signal-to-Noise Ratio
        noise = processed - original
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))
        
        # Total Harmonic Distortion
        def calculate_thd(signal, sr):
            # Simplified THD calculation
            f0 = librosa.yin(signal, fmin=80, fmax=400, sr=sr)
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) == 0:
                return 0.0
            
            median_f0 = np.median(f0_clean)
            if median_f0 <= 0:
                return 0.0
            
            # Extract fundamental and harmonics
            fundamental_power = 0
            harmonic_power = 0
            
            for harmonic in [1, 2, 3, 4, 5]:
                freq = median_f0 * harmonic
                if freq < sr / 2:
                    # Extract power at this frequency
                    stft = librosa.stft(signal, n_fft=2048)
                    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
                    freq_bin = np.argmin(np.abs(freqs - freq))
                    power = np.mean(np.abs(stft[freq_bin]) ** 2)
                    
                    if harmonic == 1:
                        fundamental_power = power
                    else:
                        harmonic_power += power
            
            if fundamental_power > 0:
                return harmonic_power / fundamental_power
            return 0.0
        
        original_thd = calculate_thd(original, sr)
        processed_thd = calculate_thd(processed, sr)
        
        # Dynamic range
        original_dynamic_range = 20 * np.log10(np.max(np.abs(original)) / 
                                              (np.sqrt(np.mean(original ** 2)) + 1e-10))
        processed_dynamic_range = 20 * np.log10(np.max(np.abs(processed)) / 
                                               (np.sqrt(np.mean(processed ** 2)) + 1e-10))
        
        # Spectral centroid change
        original_centroid = np.mean(librosa.feature.spectral_centroid(y=original, sr=sr))
        processed_centroid = np.mean(librosa.feature.spectral_centroid(y=processed, sr=sr))
        
        return {
            'snr_db': float(snr),
            'original_thd': float(original_thd),
            'processed_thd': float(processed_thd),
            'thd_change': float(processed_thd - original_thd),
            'original_dynamic_range_db': float(original_dynamic_range),
            'processed_dynamic_range_db': float(processed_dynamic_range),
            'dynamic_range_change_db': float(processed_dynamic_range - original_dynamic_range),
            'spectral_centroid_change_hz': float(processed_centroid - original_centroid),
            'rms_level_change_db': float(20 * np.log10(
                (np.sqrt(np.mean(processed ** 2)) + 1e-10) / 
                (np.sqrt(np.mean(original ** 2)) + 1e-10)
            ))
        }
    
    def _analyze_perceptual_changes(self, original: np.ndarray, processed: np.ndarray, 
                                   sr: int) -> Dict[str, Any]:
        """Analyze perceptual changes."""
        # Mel-frequency cepstral coefficients
        original_mfcc = librosa.feature.mfcc(y=original, sr=sr, n_mfcc=13)
        processed_mfcc = librosa.feature.mfcc(y=processed, sr=sr, n_mfcc=13)
        
        # Calculate MFCC distance
        mfcc_distance = np.mean(np.sqrt(np.sum((original_mfcc - processed_mfcc) ** 2, axis=0)))
        
        # Chroma features (harmonic content)
        original_chroma = librosa.feature.chroma_stft(y=original, sr=sr)
        processed_chroma = librosa.feature.chroma_stft(y=processed, sr=sr)
        
        chroma_correlation = np.corrcoef(
            original_chroma.flatten(), processed_chroma.flatten()
        )[0, 1]
        
        # Tempo analysis
        original_tempo = librosa.beat.tempo(y=original, sr=sr)[0]
        processed_tempo = librosa.beat.tempo(y=processed, sr=sr)[0]
        
        return {
            'mfcc_distance': float(mfcc_distance),
            'chroma_correlation': float(chroma_correlation),
            'original_tempo': float(original_tempo),
            'processed_tempo': float(processed_tempo),
            'tempo_change': float(processed_tempo - original_tempo)
        }
    
    def _calculate_effectiveness_score(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall effectiveness score."""
        scores = {}
        
        # Watermark removal effectiveness (0-100)
        watermark_results = results.get('watermark_detection', {})
        removal_rate = watermark_results.get('overall_removal_rate', 0)
        remaining_high_conf = watermark_results.get('remaining_high_confidence', 0)
        
        watermark_score = removal_rate * 100
        if remaining_high_conf > 0:
            watermark_score *= 0.8  # Penalty for remaining high-confidence watermarks
        
        scores['watermark_removal'] = max(0, min(100, watermark_score))
        
        # Audio quality preservation (0-100)
        quality_results = results.get('quality_metrics', {})
        snr = quality_results.get('snr_db', 0)
        
        # Good SNR is > 20dB, excellent is > 40dB
        quality_score = min(100, max(0, (snr - 10) * 2.5))  # Scale 10-50dB to 0-100
        
        scores['quality_preservation'] = quality_score
        
        # Perceptual preservation (0-100)
        perceptual_results = results.get('perceptual_analysis', {})
        chroma_corr = perceptual_results.get('chroma_correlation', 0)
        mfcc_distance = perceptual_results.get('mfcc_distance', 10)
        
        # High correlation is good, low MFCC distance is good
        perceptual_score = (chroma_corr * 50) + max(0, (10 - mfcc_distance) * 5)
        scores['perceptual_preservation'] = max(0, min(100, perceptual_score))
        
        # Overall score (weighted average)
        weights = {
            'watermark_removal': 0.5,      # 50% - most important
            'quality_preservation': 0.3,    # 30% - important
            'perceptual_preservation': 0.2   # 20% - nice to have
        }
        
        overall_score = sum(scores[key] * weights[key] for key in weights)
        scores['overall'] = overall_score
        
        return scores
    
    def generate_report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Generate a comprehensive effectiveness report."""
        report = "Watermark Removal Effectiveness Report\n"
        report += "=" * 50 + "\n\n"
        
        # File information
        file_info = results.get('file_info', {})
        report += f"Original File: {file_info.get('original', 'Unknown')}\n"
        report += f"Processed File: {file_info.get('processed', 'Unknown')}\n"
        report += f"Duration: {file_info.get('duration', 0):.2f} seconds\n"
        report += f"Sample Rate: {file_info.get('sample_rate', 0)} Hz\n\n"
        
        # Effectiveness scores
        scores = results.get('effectiveness_score', {})
        report += "Effectiveness Scores:\n"
        report += "-" * 20 + "\n"
        for score_type, score in scores.items():
            report += f"{score_type.replace('_', ' ').title()}: {score:.1f}/100\n"
        report += "\n"
        
        # Watermark detection results
        watermark_results = results.get('watermark_detection', {})
        report += "Watermark Detection Results:\n"
        report += "-" * 28 + "\n"
        report += f"Original watermarks detected: {watermark_results.get('original_watermarks', 0)}\n"
        report += f"Processed watermarks detected: {watermark_results.get('processed_watermarks', 0)}\n"
        report += f"Overall removal rate: {watermark_results.get('overall_removal_rate', 0):.1%}\n"
        report += f"Remaining high-confidence: {watermark_results.get('remaining_high_confidence', 0)}\n\n"
        
        # Category breakdown
        removal_by_category = watermark_results.get('removal_by_category', {})
        if removal_by_category:
            report += "Removal by Category:\n"
            for category, data in removal_by_category.items():
                report += f"  {category.title()}: {data['original_count']} → {data['processed_count']} "
                report += f"({data['removal_rate']:.1%} removed)\n"
            report += "\n"
        
        # Quality metrics
        quality_results = results.get('quality_metrics', {})
        report += "Audio Quality Metrics:\n"
        report += "-" * 21 + "\n"
        report += f"Signal-to-Noise Ratio: {quality_results.get('snr_db', 0):.1f} dB\n"
        report += f"THD Change: {quality_results.get('thd_change', 0):.3f}\n"
        report += f"Dynamic Range Change: {quality_results.get('dynamic_range_change_db', 0):.1f} dB\n"
        report += f"RMS Level Change: {quality_results.get('rms_level_change_db', 0):.1f} dB\n\n"
        
        # Assessment
        overall_score = scores.get('overall', 0)
        if overall_score >= 80:
            assessment = "EXCELLENT - Highly effective watermark removal with minimal quality impact"
        elif overall_score >= 60:
            assessment = "GOOD - Effective watermark removal with acceptable quality preservation"
        elif overall_score >= 40:
            assessment = "FAIR - Moderate watermark removal, some quality impact"
        else:
            assessment = "POOR - Limited effectiveness or significant quality degradation"
        
        report += f"Overall Assessment: {assessment}\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {output_path}")
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Test watermark removal effectiveness")
    parser.add_argument("original", help="Original audio file")
    parser.add_argument("processed", help="Processed audio file")
    parser.add_argument("--output", help="Output report file")
    parser.add_argument("--json", help="Output JSON results file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.original):
        print(f"Error: Original file '{args.original}' not found.")
        return 1
    
    if not os.path.exists(args.processed):
        print(f"Error: Processed file '{args.processed}' not found.")
        return 1
    
    # Run effectiveness test
    tester = WatermarkEffectivenessTester()
    results = tester.test_file_pair(args.original, args.processed)
    
    # Generate report
    report = tester.generate_report(results, args.output)
    print(report)
    
    # Save JSON results if requested
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"JSON results saved to {args.json}")

if __name__ == "__main__":
    main()
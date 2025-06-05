#!/usr/bin/env python3
"""
Watermark Removal Comparison Tool
Compares original and processed audio to verify watermark removal effectiveness.
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
from enhanced_suno_detector import SunoWatermarkDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WatermarkComparisonAnalyzer:
    """Analyzes the effectiveness of watermark removal."""
    
    def __init__(self):
        self.suno_detector = SunoWatermarkDetector()
    
    def compare_files(self, original_path: str, processed_path: str) -> Dict[str, Any]:
        """Compare original and processed audio files."""
        results = {}
        
        # Load both files
        try:
            original, sr_orig = librosa.load(original_path, sr=None, mono=False)
            processed, sr_proc = librosa.load(processed_path, sr=None, mono=False)
        except Exception as e:
            return {'error': f"Failed to load audio files: {e}"}
        
        if sr_orig != sr_proc:
            logger.warning(f"Sample rate mismatch: {sr_orig} vs {sr_proc}")
        
        # Convert to mono for analysis
        if len(original.shape) > 1:
            original_mono = np.mean(original, axis=0)
        else:
            original_mono = original
            
        if len(processed.shape) > 1:
            processed_mono = np.mean(processed, axis=0)
        else:
            processed_mono = processed
        
        # Ensure same length
        min_len = min(len(original_mono), len(processed_mono))
        original_mono = original_mono[:min_len]
        processed_mono = processed_mono[:min_len]
        
        # Basic comparison metrics
        results['basic_metrics'] = self._calculate_basic_metrics(original_mono, processed_mono)
        
        # Watermark detection comparison
        results['watermark_analysis'] = self._compare_watermark_detection(
            original_mono, processed_mono, sr_orig
        )
        
        # Spectral analysis
        results['spectral_analysis'] = self._compare_spectral_content(
            original_mono, processed_mono, sr_orig
        )
        
        # Audio quality metrics
        results['quality_metrics'] = self._calculate_quality_metrics(
            original_mono, processed_mono, sr_orig
        )
        
        return results
    
    def _calculate_basic_metrics(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """Calculate basic comparison metrics."""
        # Signal-to-Noise Ratio
        noise = processed - original
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((original - processed) ** 2))
        
        # Correlation coefficient
        correlation = np.corrcoef(original, processed)[0, 1]
        
        # Dynamic range comparison
        orig_dynamic_range = 20 * np.log10(np.max(np.abs(original)) / (np.std(original) + 1e-10))
        proc_dynamic_range = 20 * np.log10(np.max(np.abs(processed)) / (np.std(processed) + 1e-10))
        
        return {
            'snr_db': float(snr),
            'rmse': float(rmse),
            'correlation': float(correlation),
            'original_dynamic_range': float(orig_dynamic_range),
            'processed_dynamic_range': float(proc_dynamic_range),
            'dynamic_range_change': float(proc_dynamic_range - orig_dynamic_range)
        }
    
    def _compare_watermark_detection(self, original: np.ndarray, processed: np.ndarray, 
                                   sr: int) -> Dict[str, Any]:
        """Compare watermark detection before and after processing."""
        # Detect watermarks in original
        original_watermarks = self.suno_detector.detect_suno_watermarks(original, sr)
        
        # Detect watermarks in processed
        processed_watermarks = self.suno_detector.detect_suno_watermarks(processed, sr)
        
        # Categorize watermarks by type
        def categorize_watermarks(watermarks):
            categories = {}
            for w in watermarks:
                wtype = w.get('type', 'unknown')
                if wtype not in categories:
                    categories[wtype] = []
                categories[wtype].append(w)
            return categories
        
        orig_categories = categorize_watermarks(original_watermarks)
        proc_categories = categorize_watermarks(processed_watermarks)
        
        # Calculate removal effectiveness
        removal_stats = {}
        for wtype in orig_categories:
            orig_count = len(orig_categories[wtype])
            proc_count = len(proc_categories.get(wtype, []))
            removal_rate = (orig_count - proc_count) / orig_count if orig_count > 0 else 0
            
            removal_stats[wtype] = {
                'original_count': orig_count,
                'processed_count': proc_count,
                'removal_rate': float(removal_rate),
                'removed_count': orig_count - proc_count
            }
        
        return {
            'original_total': len(original_watermarks),
            'processed_total': len(processed_watermarks),
            'total_removal_rate': float((len(original_watermarks) - len(processed_watermarks)) / 
                                      len(original_watermarks)) if len(original_watermarks) > 0 else 0,
            'by_type': removal_stats
        }
    
    def _compare_spectral_content(self, original: np.ndarray, processed: np.ndarray, 
                                sr: int) -> Dict[str, Any]:
        """Compare spectral content between original and processed audio."""
        # Compute spectrograms
        nperseg = min(2048, len(original) // 4)
        nperseg = max(256, nperseg)
        
        freqs, times, spec_orig = signal.spectrogram(
            original, fs=sr, nperseg=nperseg, noverlap=nperseg//2
        )
        _, _, spec_proc = signal.spectrogram(
            processed, fs=sr, nperseg=nperseg, noverlap=nperseg//2
        )
        
        # Convert to dB
        spec_orig_db = 10 * np.log10(spec_orig + 1e-10)
        spec_proc_db = 10 * np.log10(spec_proc + 1e-10)
        
        # Calculate spectral differences
        spec_diff = spec_proc_db - spec_orig_db
        
        # Analyze frequency bands
        frequency_bands = [
            (0, 1000, "Low frequencies"),
            (1000, 5000, "Mid frequencies"),
            (5000, 10000, "High frequencies"),
            (10000, 15000, "Very high frequencies"),
            (15000, sr//2, "Ultrasonic frequencies")
        ]
        
        band_analysis = {}
        for low_freq, high_freq, description in frequency_bands:
            if high_freq > sr / 2:
                high_freq = sr // 2
            
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if not np.any(freq_mask):
                continue
            
            band_diff = spec_diff[freq_mask]
            
            band_analysis[f"{low_freq}-{high_freq}Hz"] = {
                'description': description,
                'mean_change_db': float(np.mean(band_diff)),
                'std_change_db': float(np.std(band_diff)),
                'max_reduction_db': float(np.min(band_diff)),
                'max_increase_db': float(np.max(band_diff))
            }
        
        return {
            'overall_spectral_change': float(np.mean(np.abs(spec_diff))),
            'frequency_bands': band_analysis
        }
    
    def _calculate_quality_metrics(self, original: np.ndarray, processed: np.ndarray, 
                                 sr: int) -> Dict[str, Any]:
        """Calculate audio quality metrics."""
        # Perceptual metrics
        
        # 1. Spectral centroid comparison
        orig_centroid = np.mean(librosa.feature.spectral_centroid(y=original, sr=sr))
        proc_centroid = np.mean(librosa.feature.spectral_centroid(y=processed, sr=sr))
        
        # 2. Spectral rolloff comparison
        orig_rolloff = np.mean(librosa.feature.spectral_rolloff(y=original, sr=sr))
        proc_rolloff = np.mean(librosa.feature.spectral_rolloff(y=processed, sr=sr))
        
        # 3. Zero crossing rate comparison
        orig_zcr = np.mean(librosa.feature.zero_crossing_rate(original))
        proc_zcr = np.mean(librosa.feature.zero_crossing_rate(processed))
        
        # 4. RMS energy comparison
        orig_rms = np.mean(librosa.feature.rms(y=original))
        proc_rms = np.mean(librosa.feature.rms(y=processed))
        
        # 5. Spectral bandwidth comparison
        orig_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=original, sr=sr))
        proc_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=processed, sr=sr))
        
        return {
            'spectral_centroid_change': float(proc_centroid - orig_centroid),
            'spectral_rolloff_change': float(proc_rolloff - orig_rolloff),
            'zero_crossing_rate_change': float(proc_zcr - orig_zcr),
            'rms_energy_change': float(proc_rms - orig_rms),
            'spectral_bandwidth_change': float(proc_bandwidth - orig_bandwidth),
            'relative_rms_change': float((proc_rms - orig_rms) / orig_rms) if orig_rms > 0 else 0
        }
    
    def generate_report(self, original_path: str, processed_path: str) -> str:
        """Generate a comprehensive comparison report."""
        results = self.compare_files(original_path, processed_path)
        
        if 'error' in results:
            return f"Error: {results['error']}"
        
        report = f"Watermark Removal Effectiveness Report\n"
        report += f"=" * 50 + "\n"
        report += f"Original: {os.path.basename(original_path)}\n"
        report += f"Processed: {os.path.basename(processed_path)}\n\n"
        
        # Basic metrics
        basic = results['basic_metrics']
        report += f"Basic Quality Metrics:\n"
        report += f"-" * 20 + "\n"
        report += f"Signal-to-Noise Ratio: {basic['snr_db']:.2f} dB\n"
        report += f"Correlation: {basic['correlation']:.4f}\n"
        report += f"RMSE: {basic['rmse']:.6f}\n"
        report += f"Dynamic Range Change: {basic['dynamic_range_change']:.2f} dB\n\n"
        
        # Watermark removal effectiveness
        watermark = results['watermark_analysis']
        report += f"Watermark Removal Effectiveness:\n"
        report += f"-" * 33 + "\n"
        report += f"Original watermarks detected: {watermark['original_total']}\n"
        report += f"Remaining watermarks: {watermark['processed_total']}\n"
        report += f"Overall removal rate: {watermark['total_removal_rate']:.1%}\n\n"
        
        if watermark['by_type']:
            report += f"Removal by watermark type:\n"
            for wtype, stats in watermark['by_type'].items():
                report += f"  {wtype}: {stats['removed_count']}/{stats['original_count']} "
                report += f"({stats['removal_rate']:.1%})\n"
            report += "\n"
        
        # Spectral analysis
        spectral = results['spectral_analysis']
        report += f"Spectral Content Changes:\n"
        report += f"-" * 24 + "\n"
        report += f"Overall spectral change: {spectral['overall_spectral_change']:.2f} dB\n\n"
        
        for band, data in spectral['frequency_bands'].items():
            report += f"{band} ({data['description']}):\n"
            report += f"  Mean change: {data['mean_change_db']:.2f} dB\n"
            report += f"  Max reduction: {data['max_reduction_db']:.2f} dB\n"
            report += f"  Max increase: {data['max_increase_db']:.2f} dB\n\n"
        
        # Quality assessment
        quality = results['quality_metrics']
        report += f"Perceptual Quality Changes:\n"
        report += f"-" * 26 + "\n"
        report += f"Spectral centroid change: {quality['spectral_centroid_change']:.1f} Hz\n"
        report += f"RMS energy change: {quality['relative_rms_change']:.1%}\n"
        report += f"Spectral bandwidth change: {quality['spectral_bandwidth_change']:.1f} Hz\n\n"
        
        # Overall assessment
        report += f"Overall Assessment:\n"
        report += f"-" * 18 + "\n"
        
        # Quality score (higher is better)
        quality_score = 0
        if basic['snr_db'] > 20:
            quality_score += 25
        elif basic['snr_db'] > 10:
            quality_score += 15
        elif basic['snr_db'] > 0:
            quality_score += 5
        
        if basic['correlation'] > 0.95:
            quality_score += 25
        elif basic['correlation'] > 0.9:
            quality_score += 15
        elif basic['correlation'] > 0.8:
            quality_score += 5
        
        # Watermark removal score
        removal_score = watermark['total_removal_rate'] * 50
        
        total_score = quality_score + removal_score
        
        if total_score >= 80:
            assessment = "EXCELLENT"
        elif total_score >= 60:
            assessment = "GOOD"
        elif total_score >= 40:
            assessment = "FAIR"
        else:
            assessment = "POOR"
        
        report += f"Quality preservation: {quality_score}/50\n"
        report += f"Watermark removal: {removal_score:.0f}/50\n"
        report += f"Total score: {total_score:.0f}/100\n"
        report += f"Assessment: {assessment}\n"
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Compare original and processed audio files")
    parser.add_argument("original", help="Original audio file")
    parser.add_argument("processed", help="Processed audio file")
    parser.add_argument("--output", help="Output report file (optional)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.original):
        print(f"Error: Original file '{args.original}' not found.")
        return 1
    
    if not os.path.exists(args.processed):
        print(f"Error: Processed file '{args.processed}' not found.")
        return 1
    
    analyzer = WatermarkComparisonAnalyzer()
    report = analyzer.generate_report(args.original, args.processed)
    
    print(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport saved to {args.output}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Comprehensive test script to verify audio processing fixes.
Tests for silent output, audio corruption, and quality preservation.
"""

import os
import sys
import numpy as np
import soundfile as sf
import logging
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_audio_fingerprint_remover import (
    apply_watermark_removal,
    process_audio,
    ProcessingConfig,
    validate_audio_content,
)
from aggressive_watermark_remover import AggressiveWatermarkRemover
from sota_watermark_remover import StateOfTheArtWatermarkRemover
from audio_processing_fixes import AudioProcessingFixes, WatermarkRemovalFixes

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_audio(duration=0.5, sr=16000):
    """Create a test audio signal with various frequency components."""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Mix of frequencies to test different processing scenarios
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +      # A4 note
        0.2 * np.sin(2 * np.pi * 880 * t) +      # A5 note
        0.1 * np.sin(2 * np.pi * 1760 * t) +     # A6 note
        0.05 * np.sin(2 * np.pi * 15000 * t) +   # High frequency
        0.02 * np.sin(2 * np.pi * 19000 * t) +   # Near ultrasonic
        0.1 * np.random.randn(len(t)) * 0.01     # Some noise
    )
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio, sr


def compute_audio_metrics(audio, name="Audio"):
    """Calculate and display audio metrics."""
    if audio is None or len(audio) == 0:
        return {
            'valid': False,
            'max_amplitude': 0.0,
            'rms': 0.0,
            'has_nan': True,
            'has_inf': True,
            'silent': True
        }
    
    max_amp = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    has_nan = np.any(np.isnan(audio))
    has_inf = np.any(np.isinf(audio))
    silent = max_amp < 1e-10
    
    logger.info(f"{name} metrics:")
    logger.info(f"  Max amplitude: {max_amp:.6f}")
    logger.info(f"  RMS: {rms:.6f}")
    logger.info(f"  Has NaN: {has_nan}")
    logger.info(f"  Has Inf: {has_inf}")
    logger.info(f"  Is silent: {silent}")
    
    return {
        'valid': not (has_nan or has_inf or silent),
        'max_amplitude': max_amp,
        'rms': rms,
        'has_nan': has_nan,
        'has_inf': has_inf,
        'silent': silent
    }


def test_processing_levels():
    """Test different processing levels to ensure they don't corrupt audio."""
    logger.info("\n=== Testing Processing Levels ===")
    
    # Create test audio
    test_audio, sr = create_test_audio()
    
    # Test each processing level using fast in-memory processing
    levels = ['moderate', 'aggressive']
    synthetic_watermarks = [{'freq_range': (6500, 9000)}]

    for level in levels:
        logger.info(f"\nTesting {level} processing...")
        config = ProcessingConfig.get_profile(level)
        processed_audio = apply_watermark_removal(test_audio, sr, synthetic_watermarks, config)

        metrics = compute_audio_metrics(processed_audio, f"{level.capitalize()} processed")
        assert metrics['valid'], f"{level} processing produced invalid audio"

        # Calculate quality preservation
        snr = calculate_snr(test_audio, processed_audio)
        logger.info(f"  SNR: {snr:.2f} dB")
        assert snr > 0, "SNR should remain positive after processing"


def test_short_audio():
    """Test processing of very short audio clips."""
    logger.info("\n=== Testing Short Audio Processing ===")

    duration = 0.1  # seconds
    sr = 22050

    logger.info(f"\nTesting {duration}s audio in-memory...")

    # Create short audio
    test_audio, sr = create_test_audio(duration=duration, sr=sr)

    config = ProcessingConfig.get_profile('moderate')
    synthetic_watermarks = [{'freq_range': (4000, 6000)}]
    processed_audio = apply_watermark_removal(test_audio, sr, synthetic_watermarks, config)

    metrics = compute_audio_metrics(processed_audio, f"{duration}s processed")
    assert metrics['valid'], f"Short audio ({duration}s) failed validation"


def test_edge_cases():
    """Test edge cases that might cause issues."""
    logger.info("\n=== Testing Edge Cases ===")

    sr = 16000

    # Test 1: Audio with some NaN values
    logger.info("\nTest 1: Audio with NaN values")
    audio_with_nan = create_test_audio(sr=sr)[0]
    audio_with_nan[100:110] = np.nan

    cleaned = AudioProcessingFixes.safe_nan_cleanup(audio_with_nan, fallback=audio_with_nan)
    metrics = compute_audio_metrics(cleaned, "NaN test processed")
    assert metrics['valid'], "Processing should recover from NaN regions"

    # Test 2: Very quiet audio
    logger.info("\nTest 2: Very quiet audio")
    quiet_audio = create_test_audio(sr=sr)[0] * 0.0001

    config = ProcessingConfig.get_profile('gentle')
    quiet_processed = apply_watermark_removal(
        quiet_audio, sr, [{'freq_range': (3000, 5000)}], config
    )
    metrics = compute_audio_metrics(quiet_processed, "Quiet audio processed")
    assert metrics['valid'], "Quiet audio should remain valid after processing"

    # Test 3: Mono vs Stereo
    logger.info("\nTest 3: Stereo audio")
    mono_audio, sr = create_test_audio(sr=sr)
    stereo_audio = np.stack([mono_audio, mono_audio * 0.8])  # Slightly different channels

    stereo_processed = apply_watermark_removal(
        stereo_audio[0], sr, [{'freq_range': (4000, 6000)}], ProcessingConfig.get_profile('moderate')
    )

    metrics_left = compute_audio_metrics(stereo_processed, "Left channel")
    metrics_right = compute_audio_metrics(stereo_audio[1], "Right channel baseline")
    assert metrics_left['valid'] and metrics_right['valid'], "Stereo channels should stay valid"


def test_validate_audio_content():
    """Test the validate_audio_content function."""
    logger.info("\n=== Testing validate_audio_content Function ===")
    
    # Test valid audio
    valid_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
    assert validate_audio_content(valid_audio), "Valid audio should pass validation"
    logger.info("✓ Valid audio passes")
    
    # Test silent audio
    silent_audio = np.zeros(16000)
    assert not validate_audio_content(silent_audio), "Silent audio should fail validation"
    logger.info("✓ Silent audio fails")
    
    # Test audio with NaN
    nan_audio = valid_audio.copy()
    nan_audio[100:200] = np.nan
    assert not validate_audio_content(nan_audio), "Audio with NaN should fail validation"
    logger.info("✓ Audio with NaN fails")
    
    # Test very quiet audio
    quiet_audio = valid_audio * 1e-11
    assert not validate_audio_content(quiet_audio), "Very quiet audio should fail validation"
    logger.info("✓ Very quiet audio fails")
    
    # Test empty audio
    assert not validate_audio_content(np.array([])), "Empty audio should fail validation"
    logger.info("✓ Empty audio fails")
    
    logger.info("\nAll validation tests passed!")


def test_repeating_peak_suppression_reduces_tonal_energy():
    """Ensure the repeating peak suppression attenuates tonal watermarks."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    watermark_freq = 6000
    base_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    watermark_tone = 0.25 * np.sin(2 * np.pi * watermark_freq * t)
    audio = base_audio + watermark_tone

    processed = WatermarkRemovalFixes.suppress_repeating_watermark_peaks(audio, sr)

    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    tone_bin = np.argmin(np.abs(freqs - watermark_freq))

    before = np.abs(np.fft.rfft(audio))[tone_bin]
    after = np.abs(np.fft.rfft(processed))[tone_bin]

    assert after < before * 0.5, "Watermark tone should be attenuated"


def calculate_snr(original, processed):
    """Calculate Signal-to-Noise Ratio in dB."""
    # Ensure same length
    min_len = min(len(original), len(processed))
    original = original[:min_len]
    processed = processed[:min_len]
    
    signal_power = np.mean(original ** 2)
    noise = processed - original
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def main():
    """Run all tests."""
    logger.info("Starting comprehensive audio processing tests...")
    
    # Test 1: Validation function
    test_validate_audio_content()
    
    # Test 2: Processing levels
    level_results = test_processing_levels()
    
    # Test 3: Short audio
    short_results = test_short_audio()
    
    # Test 4: Edge cases
    edge_results = test_edge_cases()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    # Check processing levels
    all_levels_ok = all(
        r.get('success', False) and 
        r.get('metrics', {}).get('valid', False) 
        for r in level_results.values()
    )
    
    if all_levels_ok:
        logger.info("✅ All processing levels produce valid audio")
        for level, result in level_results.items():
            if 'snr' in result:
                logger.info(f"  {level}: SNR = {result['snr']:.2f} dB")
    else:
        logger.error("❌ Some processing levels failed")
        for level, result in level_results.items():
            if not result.get('success', False):
                logger.error(f"  {level}: Failed - {result.get('error', 'Unknown error')}")
            elif not result.get('metrics', {}).get('valid', False):
                logger.error(f"  {level}: Invalid audio output")
    
    # Check short audio
    all_short_ok = all(
        r.get('success', False) and 
        r.get('metrics', {}).get('valid', False) 
        for r in short_results.values()
    )
    
    if all_short_ok:
        logger.info("✅ Short audio processing works correctly")
    else:
        logger.error("❌ Short audio processing has issues")
        for duration, result in short_results.items():
            if not result.get('success', False):
                logger.error(f"  {duration}s: Failed - {result.get('error', 'Unknown error')}")
    
    # Check edge cases
    edge_ok = sum(1 for r in edge_results.values() if r.get('success', False))
    logger.info(f"✅ Edge cases: {edge_ok}/{len(edge_results)} passed")
    
    # Overall result
    if all_levels_ok and all_short_ok and edge_ok >= 2:
        logger.info("\n🎉 All critical tests PASSED! Audio processing is working correctly.")
        return 0
    else:
        logger.error("\n❌ Some tests FAILED. Audio processing needs further fixes.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
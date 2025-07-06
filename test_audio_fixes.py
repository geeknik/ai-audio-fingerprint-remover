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

from ai_audio_fingerprint_remover import process_audio, ProcessingConfig, validate_audio_content
from aggressive_watermark_remover import AggressiveWatermarkRemover
from sota_watermark_remover import StateOfTheArtWatermarkRemover

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_audio(duration=5.0, sr=44100):
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


def test_audio_metrics(audio, name="Audio"):
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
    
    # Test each processing level
    levels = ['gentle', 'moderate', 'aggressive', 'extreme']
    results = {}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save test audio
        input_path = os.path.join(temp_dir, "test_input.wav")
        sf.write(input_path, test_audio, sr)
        
        for level in levels:
            logger.info(f"\nTesting {level} processing...")
            
            output_path = os.path.join(temp_dir, f"test_output_{level}.wav")
            
            try:
                # Process audio
                result_path, stats = process_audio(input_path, output_path, level=level)
                
                # Load and analyze result
                processed_audio, _ = sf.read(result_path)
                
                metrics = test_audio_metrics(processed_audio, f"{level.capitalize()} processed")
                results[level] = {
                    'success': True,
                    'metrics': metrics,
                    'stats': stats
                }
                
                # Calculate quality preservation
                if metrics['valid']:
                    snr = calculate_snr(test_audio, processed_audio)
                    logger.info(f"  SNR: {snr:.2f} dB")
                    results[level]['snr'] = snr
                
            except Exception as e:
                logger.error(f"Processing failed for {level}: {e}")
                results[level] = {
                    'success': False,
                    'error': str(e)
                }
    
    return results


def test_short_audio():
    """Test processing of very short audio clips."""
    logger.info("\n=== Testing Short Audio Processing ===")
    
    # Create very short audio clips
    durations = [0.1, 0.5, 1.0]  # seconds
    sr = 44100
    
    results = {}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for duration in durations:
            logger.info(f"\nTesting {duration}s audio...")
            
            # Create short audio
            test_audio, sr = create_test_audio(duration=duration, sr=sr)
            
            input_path = os.path.join(temp_dir, f"short_{duration}s.wav")
            output_path = os.path.join(temp_dir, f"short_{duration}s_processed.wav")
            
            sf.write(input_path, test_audio, sr)
            
            try:
                # Process with moderate settings
                result_path, stats = process_audio(input_path, output_path, level='moderate')
                
                # Load and analyze
                processed_audio, _ = sf.read(result_path)
                
                metrics = test_audio_metrics(processed_audio, f"{duration}s processed")
                results[duration] = {
                    'success': True,
                    'metrics': metrics,
                    'samples': len(test_audio)
                }
                
            except Exception as e:
                logger.error(f"Failed to process {duration}s audio: {e}")
                results[duration] = {
                    'success': False,
                    'error': str(e),
                    'samples': len(test_audio)
                }
    
    return results


def test_edge_cases():
    """Test edge cases that might cause issues."""
    logger.info("\n=== Testing Edge Cases ===")
    
    results = {}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test 1: Audio with some NaN values
        logger.info("\nTest 1: Audio with NaN values")
        audio_with_nan = create_test_audio()[0]
        audio_with_nan[1000:1010] = np.nan
        
        input_path = os.path.join(temp_dir, "nan_audio.wav")
        output_path = os.path.join(temp_dir, "nan_audio_processed.wav")
        
        # Save without NaN (soundfile can't write NaN)
        sf.write(input_path, np.nan_to_num(audio_with_nan), 44100)
        
        try:
            result_path, stats = process_audio(input_path, output_path, level='moderate')
            processed, _ = sf.read(result_path)
            metrics = test_audio_metrics(processed, "NaN test processed")
            results['nan_test'] = {'success': True, 'metrics': metrics}
        except Exception as e:
            results['nan_test'] = {'success': False, 'error': str(e)}
        
        # Test 2: Very quiet audio
        logger.info("\nTest 2: Very quiet audio")
        quiet_audio = create_test_audio()[0] * 0.0001
        
        input_path = os.path.join(temp_dir, "quiet_audio.wav")
        output_path = os.path.join(temp_dir, "quiet_audio_processed.wav")
        
        sf.write(input_path, quiet_audio, 44100)
        
        try:
            result_path, stats = process_audio(input_path, output_path, level='gentle')
            processed, _ = sf.read(result_path)
            metrics = test_audio_metrics(processed, "Quiet audio processed")
            results['quiet_test'] = {'success': True, 'metrics': metrics}
        except Exception as e:
            results['quiet_test'] = {'success': False, 'error': str(e)}
        
        # Test 3: Mono vs Stereo
        logger.info("\nTest 3: Stereo audio")
        mono_audio, sr = create_test_audio()
        stereo_audio = np.stack([mono_audio, mono_audio * 0.8])  # Slightly different channels
        
        input_path = os.path.join(temp_dir, "stereo_audio.wav")
        output_path = os.path.join(temp_dir, "stereo_audio_processed.wav")
        
        sf.write(input_path, stereo_audio.T, sr)
        
        try:
            result_path, stats = process_audio(input_path, output_path, level='moderate')
            processed, _ = sf.read(result_path)
            
            if len(processed.shape) > 1:
                logger.info("Stereo output preserved")
                metrics_left = test_audio_metrics(processed[:, 0], "Left channel")
                metrics_right = test_audio_metrics(processed[:, 1], "Right channel")
                results['stereo_test'] = {
                    'success': True,
                    'stereo': True,
                    'left': metrics_left,
                    'right': metrics_right
                }
            else:
                metrics = test_audio_metrics(processed, "Stereo->Mono processed")
                results['stereo_test'] = {
                    'success': True,
                    'stereo': False,
                    'metrics': metrics
                }
        except Exception as e:
            results['stereo_test'] = {'success': False, 'error': str(e)}
    
    return results


def test_validate_audio_content():
    """Test the validate_audio_content function."""
    logger.info("\n=== Testing validate_audio_content Function ===")
    
    # Test valid audio
    valid_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    assert validate_audio_content(valid_audio), "Valid audio should pass validation"
    logger.info("✓ Valid audio passes")
    
    # Test silent audio
    silent_audio = np.zeros(44100)
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
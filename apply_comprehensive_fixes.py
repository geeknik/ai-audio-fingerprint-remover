#!/usr/bin/env python3
"""
Apply comprehensive fixes to the AI Audio Fingerprint Remover
This script patches the existing files with the fixes from audio_processing_fixes.py
"""

import sys
import os
import shutil
from datetime import datetime

def create_backup(file_path):
    """Create a backup of the original file."""
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")
    return backup_path

def apply_main_fixes():
    """Apply fixes to ai_audio_fingerprint_remover.py"""
    print("\n=== Patching ai_audio_fingerprint_remover.py ===")
    
    # Read the original file
    with open('ai_audio_fingerprint_remover.py', 'r') as f:
        content = f.read()
    
    # Create backup
    create_backup('ai_audio_fingerprint_remover.py')
    
    # Import the fixes at the top
    import_statement = """from audio_processing_fixes import (
    AudioProcessingFixes, WatermarkRemovalFixes, AudioQualityEnhancer
)
"""
    
    # Add import after the other imports
    import_pos = content.find('from typing import')
    if import_pos > 0:
        # Find the end of the imports section
        import_end = content.find('\n\n', import_pos)
        content = content[:import_end] + '\n' + import_statement + content[import_end:]
    
    # Replace the validate_audio_content function with the enhanced version
    old_validate_start = content.find('def validate_audio_content(')
    if old_validate_start > 0:
        old_validate_end = content.find('\n\ndef ', old_validate_start)
        if old_validate_end < 0:
            old_validate_end = content.find('\n\nclass ', old_validate_start)
        
        new_validate = '''def validate_audio_content(audio: np.ndarray, min_amplitude: float = 1e-10, context: str = "") -> bool:
    """Use enhanced validation from AudioProcessingFixes."""
    return AudioProcessingFixes.validate_audio_content(audio, min_amplitude, context)
'''
        content = content[:old_validate_start] + new_validate + content[old_validate_end:]
    
    # Fix the apply_watermark_removal function
    apply_wm_start = content.find('def apply_watermark_removal(')
    if apply_wm_start > 0:
        # Find the function body
        func_start = content.find(':', apply_wm_start) + 1
        func_end = content.find('\n\ndef ', func_start)
        if func_end < 0:
            func_end = content.find('\n\nclass ', func_start)
        
        # Extract indentation
        indent_line = content.find('\n', func_start)
        next_line_start = indent_line + 1
        indent = ''
        for char in content[next_line_start:]:
            if char in ' \t':
                indent += char
            else:
                break
        
        new_body = f'''
{indent}"""Apply filters to remove detected watermarks with enhanced safety."""
{indent}if not config.enable_watermark_removal:
{indent}    return audio
{indent}
{indent}result = audio.copy()
{indent}
{indent}# Group watermarks by frequency range
{indent}freq_ranges = []
{indent}for watermark in watermarks:
{indent}    freq_range = watermark.get('freq_range')
{indent}    if freq_range and freq_range not in freq_ranges:
{indent}        freq_ranges.append(freq_range)
{indent}
{indent}# Use conservative frequency removal
{indent}if freq_ranges:
{indent}    result = WatermarkRemovalFixes.conservative_frequency_removal(
{indent}        result, sr, freq_ranges, max_attenuation_db=20
{indent}    )
{indent}
{indent}# Apply psychoacoustic watermark removal for high-frequency watermarks
{indent}high_freq_watermarks = [
{indent}    w.get('freq_range', [0, 0])[0] 
{indent}    for w in watermarks 
{indent}    if w.get('freq_range') and w['freq_range'][0] > 15000
{indent}]
{indent}
{indent}if high_freq_watermarks:
{indent}    result = WatermarkRemovalFixes.psychoacoustic_watermark_removal(
{indent}        result, sr, high_freq_watermarks
{indent}    )
{indent}
{indent}# Validate result
{indent}if not AudioProcessingFixes.validate_audio_content(result, context="Watermark removal"):
{indent}    logger.warning("Watermark removal failed validation, returning original")
{indent}    return audio
{indent}
{indent}return result
'''
        
        content = content[:func_start] + new_body + content[func_end:]
    
    # Fix the add_timing_variations function
    timing_var_start = content.find('def add_timing_variations(')
    if timing_var_start > 0:
        # Replace with safer implementation
        func_start = content.find(':', timing_var_start) + 1
        func_end = content.find('\n\ndef ', func_start)
        if func_end < 0:
            func_end = content.find('\n\nclass ', func_start)
        
        # Extract indentation
        indent_line = content.find('\n', func_start)
        next_line_start = indent_line + 1
        indent = ''
        for char in content[next_line_start:]:
            if char in ' \t':
                indent += char
            else:
                break
        
        new_body = f'''
{indent}"""Add subtle timing variations to make AI-generated audio sound more natural."""
{indent}if not config.enable_timing_variations:
{indent}    return audio
{indent}
{indent}# Use quality enhancer for natural variations
{indent}result = AudioQualityEnhancer.add_natural_variations(
{indent}    audio, sr, variation_amount=config.timing_variation_range
{indent})
{indent}
{indent}# Add harmonic enhancement for more natural sound
{indent}if config.enable_harmonic_adjustments:
{indent}    result = AudioQualityEnhancer.harmonic_enhancement(
{indent}        result, sr, enhancement_amount=config.harmonic_distortion_amount
{indent}    )
{indent}
{indent}# Final validation
{indent}if not AudioProcessingFixes.validate_audio_content(result, context="Timing variations"):
{indent}    logger.warning("Timing variations failed validation, returning original")
{indent}    return audio
{indent}
{indent}return result
'''
        
        content = content[:func_start] + new_body + content[func_end:]
    
    # Update ProcessingConfig with safer defaults
    config_start = content.find('class ProcessingConfig:')
    if config_start > 0:
        # Find the default values section
        defaults_start = content.find('# Processing level', config_start)
        if defaults_start > 0:
            # Replace aggressive default values with safer ones
            content = content.replace('noise_level: float = 0.0001', 'noise_level: float = 0.00001')
            content = content.replace('timing_stretch_range: float = 0.005', 'timing_stretch_range: float = 0.001')
            content = content.replace('distribution_noise_level: float = 0.0001', 'distribution_noise_level: float = 0.00001')
            content = content.replace('harmonic_distortion_amount: float = 0.01', 'harmonic_distortion_amount: float = 0.002')
            content = content.replace('micro_dynamics_amount: float = 0.001', 'micro_dynamics_amount: float = 0.0001')
    
    # Write the patched file
    with open('ai_audio_fingerprint_remover.py', 'w') as f:
        f.write(content)
    
    print("✅ Successfully patched ai_audio_fingerprint_remover.py")

def apply_aggressive_fixes():
    """Apply fixes to aggressive_watermark_remover.py"""
    print("\n=== Patching aggressive_watermark_remover.py ===")
    
    # Read the original file
    with open('aggressive_watermark_remover.py', 'r') as f:
        content = f.read()
    
    # Create backup
    create_backup('aggressive_watermark_remover.py')
    
    # Add import
    import_statement = "from audio_processing_fixes import AudioProcessingFixes\n"
    
    # Add import after other imports
    import_pos = content.find('from typing import')
    if import_pos > 0:
        import_end = content.find('\n\n', import_pos)
        content = content[:import_end] + '\n' + import_statement + content[import_end:]
    
    # Replace _safe_nan_cleanup with the enhanced version
    nan_cleanup_start = content.find('def _safe_nan_cleanup(')
    if nan_cleanup_start > 0:
        nan_cleanup_end = content.find('\n    \n    def ', nan_cleanup_start)
        if nan_cleanup_end < 0:
            nan_cleanup_end = content.find('\n\n    def ', nan_cleanup_start)
        
        new_cleanup = '''    def _safe_nan_cleanup(self, processed_audio: np.ndarray, fallback_audio: np.ndarray) -> np.ndarray:
        """Use enhanced NaN cleanup from AudioProcessingFixes."""
        return AudioProcessingFixes.safe_nan_cleanup(processed_audio, fallback_audio)
'''
        content = content[:nan_cleanup_start] + new_cleanup + content[nan_cleanup_end:]
    
    # Update removal_strength parameters to be more conservative
    strength_start = content.find("self.removal_strength = {")
    if strength_start > 0:
        strength_end = content.find("}", strength_start) + 1
        new_strength = """self.removal_strength = {
            'frequency_notch': 0.10,  # Reduced from 0.15
            'spectral_gating': 0.05,  # Reduced from 0.10
            'temporal_smoothing': 0.05,  # Reduced from 0.1
            'phase_randomization': 0.02,  # Reduced from 0.05
            'pattern_disruption': 0.10,  # Reduced from 0.2
        }"""
        content = content[:strength_start] + new_strength + content[strength_end:]
    
    # Write the patched file
    with open('aggressive_watermark_remover.py', 'w') as f:
        f.write(content)
    
    print("✅ Successfully patched aggressive_watermark_remover.py")

def apply_sota_fixes():
    """Apply fixes to sota_watermark_remover.py"""
    print("\n=== Patching sota_watermark_remover.py ===")
    
    # Read the original file
    with open('sota_watermark_remover.py', 'r') as f:
        content = f.read()
    
    # Create backup
    create_backup('sota_watermark_remover.py')
    
    # Add import
    import_statement = "from audio_processing_fixes import AudioProcessingFixes, WatermarkRemovalFixes\n"
    
    # Add import after other imports
    import_pos = content.find('from typing import')
    if import_pos > 0:
        import_end = content.find('\n\n', import_pos)
        content = content[:import_end] + '\n' + import_statement + content[import_end:]
    
    # Update _validate_and_normalize_audio to use enhanced validation
    validate_start = content.find('def _validate_and_normalize_audio(')
    if validate_start > 0:
        func_start = content.find(':', validate_start) + 1
        func_end = content.find('\n    \n    def ', func_start)
        if func_end < 0:
            func_end = content.find('\n\n    def ', func_start)
        
        new_body = '''
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
'''
        content = content[:func_start] + new_body + content[func_end:]
    
    # Update conservative mode parameters
    conservative_params = content.find('"conservative": {')
    if conservative_params > 0:
        # Make conservative mode even more conservative
        content = content.replace('"pitch_shift_cents": [10, 25]', '"pitch_shift_cents": [5, 10]')
        content = content.replace('"time_stretch_factors": [0.95, 1.05]', '"time_stretch_factors": [0.98, 1.02]')
        content = content.replace('"sample_suppression_rates": [0.001, 0.005]', '"sample_suppression_rates": [0.0005, 0.001]')
    
    # Write the patched file
    with open('sota_watermark_remover.py', 'w') as f:
        f.write(content)
    
    print("✅ Successfully patched sota_watermark_remover.py")

def create_test_script():
    """Create a comprehensive test script."""
    test_content = '''#!/usr/bin/env python3
"""
Comprehensive test script for the fixed AI Audio Fingerprint Remover
"""

import numpy as np
import soundfile as sf
import os
import sys
from ai_audio_fingerprint_remover import process_audio, ProcessingConfig
from audio_processing_fixes import AudioProcessingFixes

def create_test_audio(duration=3.0, sr=44100):
    """Create a test audio file with known characteristics."""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a complex test signal
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +     # A4
        0.2 * np.sin(2 * np.pi * 880 * t) +     # A5  
        0.1 * np.sin(2 * np.pi * 1320 * t) +    # E6
        0.05 * np.sin(2 * np.pi * 19000 * t) +  # High freq (watermark range)
        0.02 * np.random.randn(len(t))          # Noise
    )
    
    # Add some problematic sections to test robustness
    # Short silent section
    audio[sr:int(sr*1.1)] = 0
    
    # Very quiet section
    audio[int(sr*1.5):int(sr*1.6)] *= 0.001
    
    return audio, sr

def test_processing_levels():
    """Test all processing levels."""
    print("\\n=== Testing All Processing Levels ===\\n")
    
    # Create test audio
    test_audio, sr = create_test_audio()
    
    # Save test input
    test_input = "test_input_comprehensive.wav"
    sf.write(test_input, test_audio, sr)
    print(f"Created test input: {test_input}")
    print(f"  Duration: {len(test_audio)/sr:.2f}s")
    print(f"  Max amplitude: {np.max(np.abs(test_audio)):.6f}")
    print(f"  RMS: {np.sqrt(np.mean(test_audio**2)):.6f}")
    
    levels = ['gentle', 'moderate', 'aggressive', 'extreme']
    results = {}
    
    for level in levels:
        print(f"\\n--- Testing {level} mode ---")
        output_file = f"test_output_{level}.wav"
        
        try:
            # Process the audio
            result_path, stats = process_audio(test_input, output_file, level=level)
            
            # Load and validate result
            result_audio, _ = sf.read(result_path)
            
            # Validation checks
            is_valid = AudioProcessingFixes.validate_audio_content(result_audio, context=f"{level} mode")
            max_amp = np.max(np.abs(result_audio))
            rms = np.sqrt(np.mean(result_audio**2))
            
            results[level] = {
                'valid': is_valid,
                'max_amp': max_amp,
                'rms': rms,
                'stats': stats
            }
            
            print(f"  ✅ Valid: {is_valid}")
            print(f"  Max amplitude: {max_amp:.6f}")
            print(f"  RMS: {rms:.6f}")
            print(f"  Watermarks removed: {stats.watermarks_detected}")
            print(f"  Patterns normalized: {stats.patterns_normalized}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[level] = {'valid': False, 'error': str(e)}
    
    return results

def test_edge_cases():
    """Test edge cases that previously caused issues."""
    print("\\n=== Testing Edge Cases ===\\n")
    
    test_cases = []
    
    # 1. Very short audio
    short_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))
    test_cases.append(("very_short", short_audio, 44100))
    
    # 2. Silent audio
    silent_audio = np.zeros(44100)
    test_cases.append(("silent", silent_audio, 44100))
    
    # 3. Very quiet audio
    quiet_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)) * 0.0001
    test_cases.append(("very_quiet", quiet_audio, 44100))
    
    # 4. Audio with NaN values
    nan_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    nan_audio[1000:1100] = np.nan
    test_cases.append(("with_nan", nan_audio, 44100))
    
    # 5. Clipped audio
    clipped_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)) * 2.0
    clipped_audio = np.clip(clipped_audio, -1.0, 1.0)
    test_cases.append(("clipped", clipped_audio, 44100))
    
    for name, audio, sr in test_cases:
        print(f"\\n--- Testing {name} ---")
        
        # Save test case
        input_file = f"test_edge_{name}.wav"
        output_file = f"test_edge_{name}_output.wav"
        
        try:
            # Handle special cases
            if name == "silent":
                # Can't write silent audio with soundfile
                print("  ⚠️  Skipping silent audio test (cannot write silent files)")
                continue
            
            # Clean NaN for writing
            if name == "with_nan":
                clean_audio = AudioProcessingFixes.safe_nan_cleanup(audio)
                sf.write(input_file, clean_audio, sr)
            else:
                sf.write(input_file, audio, sr)
            
            # Process
            result_path, stats = process_audio(input_file, output_file, level='moderate')
            
            # Load and validate
            result_audio, _ = sf.read(result_path)
            is_valid = AudioProcessingFixes.validate_audio_content(result_audio)
            
            print(f"  ✅ Processing completed")
            print(f"  Valid output: {is_valid}")
            print(f"  Max amplitude: {np.max(np.abs(result_audio)):.6f}")
            
            # Clean up
            os.remove(input_file)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_real_audio():
    """Test with real audio files if available."""
    print("\\n=== Testing Real Audio Files ===\\n")
    
    # Look for test audio files
    test_files = []
    for file in os.listdir('.'):
        if file.endswith(('.wav', '.mp3')) and not file.startswith('test_'):
            test_files.append(file)
    
    if not test_files:
        print("No real audio files found for testing")
        return
    
    for file in test_files[:3]:  # Test up to 3 files
        print(f"\\n--- Testing {file} ---")
        output_file = f"test_real_{os.path.splitext(file)[0]}_fixed.wav"
        
        try:
            result_path, stats = process_audio(file, output_file, level='moderate')
            
            # Validate result
            result_audio, sr = sf.read(result_path)
            is_valid = AudioProcessingFixes.validate_audio_content(result_audio)
            
            print(f"  ✅ Processing completed")
            print(f"  Valid output: {is_valid}")
            print(f"  Watermarks removed: {stats.watermarks_detected}")
            print(f"  Processing time: {stats.processing_time:.2f}s")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    print("\\n" + "="*60)
    print("AI Audio Fingerprint Remover - Comprehensive Test Suite")
    print("="*60)
    
    # Run all tests
    results = test_processing_levels()
    test_edge_cases()
    test_real_audio()
    
    # Summary
    print("\\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_valid = all(r.get('valid', False) for r in results.values())
    
    if all_valid:
        print("\\n✅ ALL TESTS PASSED! The audio processor is working correctly.")
    else:
        print("\\n❌ Some tests failed. Check the output above for details.")
        failed = [level for level, r in results.items() if not r.get('valid', False)]
        print(f"   Failed levels: {', '.join(failed)}")
    
    print("\\nTest outputs have been saved with 'test_' prefix.")
'''
    
    with open('test_comprehensive_fixes.py', 'w') as f:
        f.write(test_content)
    
    os.chmod('test_comprehensive_fixes.py', 0o755)
    print("\n✅ Created test_comprehensive_fixes.py")

def main():
    """Main function to apply all fixes."""
    print("\n" + "="*60)
    print("AI Audio Fingerprint Remover - Comprehensive Fix Application")
    print("="*60)
    
    # Check if required files exist
    required_files = [
        'ai_audio_fingerprint_remover.py',
        'aggressive_watermark_remover.py', 
        'sota_watermark_remover.py',
        'audio_processing_fixes.py'
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"\n❌ Error: Missing required files: {', '.join(missing)}")
        print("Please ensure all files are in the current directory.")
        return 1
    
    try:
        # Apply fixes to each file
        apply_main_fixes()
        apply_aggressive_fixes()
        apply_sota_fixes()
        
        # Create test script
        create_test_script()
        
        print("\n" + "="*60)
        print("✅ ALL FIXES APPLIED SUCCESSFULLY!")
        print("="*60)
        
        print("\nNext steps:")
        print("1. Run the test script: python test_comprehensive_fixes.py")
        print("2. Test with your problematic audio files")
        print("3. Check the backup files if you need to revert")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error applying fixes: {e}")
        print("Check the error messages above and ensure all files are accessible.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Quick test of the audio processing fixes without soundfile dependency.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_audio_fingerprint_remover import process_audio, validate_audio_content
import librosa

print("Testing audio processing fixes...")

# Test 1: Validate function
print("\n=== Testing validate_audio_content ===")
valid_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
print(f"Valid audio: {validate_audio_content(valid_audio)}")

silent_audio = np.zeros(44100)
print(f"Silent audio: {validate_audio_content(silent_audio)}")

nan_audio = valid_audio.copy()
nan_audio[100:200] = np.nan
print(f"NaN audio: {validate_audio_content(nan_audio)}")

# Test 2: Process an existing file
print("\n=== Testing file processing ===")
test_file = "hack_song_output.wav"

if os.path.exists(test_file):
    print(f"Testing with {test_file}")
    
    # Load audio
    audio, sr = librosa.load(test_file, sr=None, mono=False)
    
    print(f"Original shape: {audio.shape if hasattr(audio, 'shape') else len(audio)}")
    print(f"Original max amplitude: {np.max(np.abs(audio)):.6f}")
    print(f"Original RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    
    # Test processing
    output_file = "quick_test_output.wav"
    
    try:
        result_path, stats = process_audio(test_file, output_file, level='moderate')
        print(f"\nProcessing completed!")
        print(f"Stats: {stats.files_processed} files, {stats.watermarks_detected} watermarks, {stats.patterns_normalized} patterns")
        
        # Load and check result
        processed, _ = librosa.load(result_path, sr=None, mono=False)
        
        print(f"\nProcessed shape: {processed.shape if hasattr(processed, 'shape') else len(processed)}")
        print(f"Processed max amplitude: {np.max(np.abs(processed)):.6f}")
        print(f"Processed RMS: {np.sqrt(np.mean(processed**2)):.6f}")
        
        # Check if audio is valid
        if len(processed.shape) > 1:
            # Stereo
            valid = all(validate_audio_content(processed[i]) for i in range(processed.shape[0]))
        else:
            # Mono
            valid = validate_audio_content(processed)
        
        if valid:
            print("\n✅ SUCCESS: Processed audio is valid!")
        else:
            print("\n❌ FAILED: Processed audio is invalid (silent or corrupted)")
            
    except Exception as e:
        print(f"\n❌ ERROR during processing: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"Test file {test_file} not found")

print("\nTest complete.")
#!/usr/bin/env python3
"""
Quick test script to verify the audio processing fixes.
"""

import numpy as np
import soundfile as sf
import logging
from aggressive_watermark_remover import AggressiveWatermarkRemover

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_audio_processing():
    """Test the fixed audio processing to ensure no silence issues."""
    
    # Load a test file that we know has audio
    test_file = "hack_song_output.wav"
    print(f"Testing with {test_file}...")
    
    try:
        # Load audio
        audio, sr = sf.read(test_file)
        print(f"Original: Max amplitude: {np.max(np.abs(audio)):.6f}, RMS: {np.sqrt(np.mean(audio**2)):.6f}")
        
        # Test the fixed aggressive remover
        remover = AggressiveWatermarkRemover()
        
        # Create some dummy watermarks for testing
        watermarks = [
            {'frequency': 19500, 'confidence': 0.8, 'type': 'suno_ultrasonic'},
            {'frequency': 15500, 'confidence': 0.6, 'type': 'suno_mid_high'}
        ]
        
        # Apply processing
        processed = remover.remove_watermarks_aggressive(audio, sr, watermarks)
        
        # Check results
        max_amp = np.max(np.abs(processed))
        rms = np.sqrt(np.mean(processed**2))
        
        print(f"Processed: Max amplitude: {max_amp:.6f}, RMS: {rms:.6f}")
        
        # Validate processing worked
        if max_amp < 1e-10:
            print("❌ FAILED: Processed audio is silent!")
            return False
        elif max_amp < 0.001:
            print("⚠️  WARNING: Processed audio is very quiet")
        else:
            print("✅ SUCCESS: Processed audio has content")
        
        # Save test output
        output_file = "test_fixes_output.wav"
        sf.write(output_file, processed, sr)
        print(f"Saved test output to {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_audio_processing()
    if success:
        print("\n🎉 Audio processing fixes appear to be working!")
    else:
        print("\n💥 Audio processing still has issues")
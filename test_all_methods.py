#!/usr/bin/env python3
"""
Comprehensive test script for all watermark removal methods.
Tests and compares different approaches on sample files.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
            return True
        else:
            print("❌ FAILED")
            print("Error:", result.stderr[-500:])  # Last 500 chars
            return False
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT (5 minutes)")
        return False
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")
        return False

def test_file_exists(filepath):
    """Check if a file exists and return its size."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ File exists: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"❌ File missing: {filepath}")
        return False

def main():
    """Run comprehensive tests on all watermark removal methods."""
    
    # Test files
    test_files = [
        "Like a Glitch in the Stack-190190-190190-2025-06-04.mp3",
        "_A Hack Song (Glitchy)_.wav"
    ]
    
    # Check if test files exist
    print("Checking test files...")
    available_files = []
    for file in test_files:
        if test_file_exists(file):
            available_files.append(file)
    
    if not available_files:
        print("❌ No test files available!")
        return 1
    
    print(f"\n📁 Found {len(available_files)} test files")
    
    # Test methods for each file
    methods = [
        {
            "name": "SOTA Balanced",
            "cmd": ["python", "sota_watermark_remover.py", "{input}", "{output}", "--mode", "balanced"],
            "suffix": "_sota_balanced"
        },
        {
            "name": "SOTA Aggressive", 
            "cmd": ["python", "sota_watermark_remover.py", "{input}", "{output}", "--mode", "aggressive"],
            "suffix": "_sota_aggressive"
        },
        {
            "name": "Integrated Aggressive",
            "cmd": ["python", "ai_audio_fingerprint_remover.py", "{input}", "{output}", "--level", "aggressive"],
            "suffix": "_integrated_aggressive"
        },
        {
            "name": "Integrated Extreme",
            "cmd": ["python", "ai_audio_fingerprint_remover.py", "{input}", "{output}", "--level", "extreme"],
            "suffix": "_integrated_extreme"
        }
    ]
    
    results = []
    
    for test_file in available_files:
        print(f"\n🎵 Testing file: {test_file}")
        file_results = {"file": test_file, "methods": {}}
        
        for method in methods:
            # Generate output filename
            base_name = Path(test_file).stem
            output_file = f"{base_name}{method['suffix']}.wav"
            
            # Prepare command
            cmd = [part.format(input=test_file, output=output_file) for part in method['cmd']]
            
            # Run test
            success = run_command(cmd, method['name'])
            
            # Check output
            if success:
                output_exists = test_file_exists(output_file)
                file_results["methods"][method['name']] = {
                    "success": True,
                    "output_exists": output_exists,
                    "output_file": output_file
                }
            else:
                file_results["methods"][method['name']] = {
                    "success": False,
                    "output_exists": False,
                    "output_file": output_file
                }
        
        results.append(file_results)
    
    # Run quality comparisons
    print(f"\n🔍 Running quality comparisons...")
    
    for file_result in results:
        test_file = file_result["file"]
        print(f"\n📊 Quality analysis for: {test_file}")
        
        for method_name, method_result in file_result["methods"].items():
            if method_result["success"] and method_result["output_exists"]:
                output_file = method_result["output_file"]
                
                print(f"\n--- {method_name} ---")
                cmd = ["python", "quick_comparison.py", test_file, output_file]
                run_command(cmd, f"Quality comparison: {method_name}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print(f"{'='*60}")
    
    for file_result in results:
        print(f"\n🎵 {file_result['file']}:")
        for method_name, method_result in file_result["methods"].items():
            status = "✅" if method_result["success"] and method_result["output_exists"] else "❌"
            print(f"  {status} {method_name}")
    
    print(f"\n🎉 Testing complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
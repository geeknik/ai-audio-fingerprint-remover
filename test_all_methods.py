#!/usr/bin/env python3
"""
Lightweight smoke tests for helper utilities used in the watermark removal scripts.
"""

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print("Output:", result.stdout[-200:])  # Last 200 chars
            return True
        print("❌ FAILED")
        print("Error:", result.stderr[-200:])  # Last 200 chars
        return False
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT (1 minute)")
        return False
    except Exception as exc:
        print(f"💥 EXCEPTION: {exc}")
        return False


def check_file_exists(filepath: Path) -> bool:
    """Check if a file exists and return its size."""
    if filepath.exists():
        size = filepath.stat().st_size
        print(f"✅ File exists: {filepath} ({size:,} bytes)")
        return True
    print(f"❌ File missing: {filepath}")
    return False


def test_run_command_success():
    assert run_command(["python", "-c", "print('ok')"], "python smoke")


def test_file_exists(tmp_path):
    sample_path = tmp_path / "tone.wav"
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    tone = 0.1 * np.sin(2 * np.pi * 440 * t)
    sf.write(sample_path, tone, 16000)

    assert check_file_exists(sample_path)
    assert not check_file_exists(sample_path.with_name("missing.wav"))

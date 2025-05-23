# AI Audio Fingerprint Remover

A comprehensive Python tool to remove AI-generated fingerprints, watermarks, and metadata from audio files.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

## Overview

AI Audio Fingerprint Remover is a powerful tool designed to address privacy concerns by removing all identifiable AI-generated traces from audio files. Modern AI systems like Suno, OpenAI, ElevenLabs, and others often embed various types of fingerprints in their generated audio—both visible metadata and hidden statistical patterns.

This tool implements multiple layers of protection to counter all known and theoretical fingerprinting techniques, ensuring your audio files maintain privacy while preserving quality.

## Features

### Complete Metadata Removal

- Strips all standard metadata (ID3, RIFF INFO, FLAC tags)
- Removes AI-specific tags and custom chunks
- Eliminates hidden identifiers in binary data
- Supports multiple formats: MP3, WAV, FLAC, AIFF

### Spectral Watermark Detection & Elimination

- Identifies and removes high-frequency watermarks
- Detects periodic patterns in specific frequency bands
- Applies targeted filters to neutralize watermarks
- Adds naturalistic noise to defeat absence-based fingerprinting

### Statistical Pattern Normalization

- Detects and corrects machine-like timing patterns
- Identifies unnatural amplitude distributions
- Normalizes frequency distributions
- Adds realistic micro-variations in timing

### Human-Like Imperfections

- Introduces subtle non-linearities in harmonics
- Adds realistic micro-dynamics
- Creates natural stereo imaging variations
- Applies minor phase adjustments

### Robust Verification

- Provides detailed reports of modifications
- Offers before/after metadata comparison
- Verifies effectiveness through hash comparison
- Handles batch processing for multiple files

## Installation

### Requirements

- Python 3.7+
- Required libraries: numpy, scipy, librosa, soundfile, mutagen

### Setup

1. Clone the repository:

```
git clone https://github.com/geeknik/ai-audio-fingerprint-remover.git
cd ai-audio-fingerprint-remover
```

2. Install dependencies:

```
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

## Usage

### Basic Usage

**Process a single file:**

```
python ai_audio_fingerprint_remover.py input.mp3 output.mp3
```

**Process a file in-place:**

```
python ai_audio_fingerprint_remover.py input.wav
```

**Process all audio files in a directory:**

```
python ai_audio_fingerprint_remover.py --directory input_folder output_folder
```

### Advanced Options

**Processing Intensity Levels** (adjusts the balance between effectiveness and audio quality):

```bash
# Gentle - minimal processing, preserves quality but may leave some fingerprints
python ai_audio_fingerprint_remover.py input.mp3 output.mp3 --level gentle

# Moderate - balanced approach (default)
python ai_audio_fingerprint_remover.py input.mp3 output.mp3 --level moderate

# Aggressive - thorough fingerprint removal with minimal quality impact
python ai_audio_fingerprint_remover.py input.mp3 output.mp3 --level aggressive

# Extreme - maximum fingerprint removal, may introduce subtle artifacts
python ai_audio_fingerprint_remover.py input.mp3 output.mp3 --level extreme
```

**Legacy Aggressive Mode** (equivalent to --level aggressive):

```
python ai_audio_fingerprint_remover.py input.mp3 output.mp3 --aggressive
```

**View Metadata Before Removal:**

```
python ai_audio_fingerprint_remover.py --show input.wav output.wav
```

**Generate Detailed Report:**

```
python ai_audio_fingerprint_remover.py input.mp3 output.mp3 --report
```

**Verify Results:**

```
python ai_audio_fingerprint_remover.py input.mp3 output.mp3 --verify
```

## Example Output

```
AI Audio Fingerprint Remover
========================================
Processing level: extreme - Maximum processing - removes all detectable fingerprints but may affect quality


Processing _A Hack Song (Glitchy)_.wav...

Results:
  Processing level: extreme
  Files processed: 1
  Watermarks detected and removed: 5
  Statistical patterns normalized: 1
  Timing adjustments applied: 1

Metadata removed:
  wav_rewrite: 1 items

Verification:
  Original file hash: 511194868d6287c54123b0ace467b321f504f99088d02499915a1fbbf9c63930
  Processed file hash: 400cf80b63d1238b373b5a35db2fc6ffd4c34740adaac901989ae7a7b311b149
  Files are different

Processing complete.
```

## Under the Hood

The tool implements a multi-layered approach to address all known and theoretical AI fingerprinting techniques:

1. **First Pass**: Complete metadata stripping, removing all standard and custom tags.
2. **Second Pass**: Spectral analysis to detect watermarks, applying targeted band-reject filters.
3. **Third Pass**: Statistical pattern analysis to detect machine-like distributions, normalizing them to human-like patterns.
4. **Final Pass**: Addition of subtle human-like imperfections to counter AI detection models that look for "too perfect" audio.

## Privacy and Security

- Processing is done entirely on your local machine - no data is sent to external servers
- No logs or processed audio data are stored unless you explicitly save them
- The tool is designed for legitimate privacy protection, not for removing copyright protections

## Limitations

- Extremely aggressive watermarks might require quality trade-offs to fully remove
- Some countermeasures may minimally impact audio quality (typically inaudible)
- Cannot remove content-based fingerprinting (where the actual content itself is the fingerprint)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is intended for legitimate privacy protection purposes. Users are responsible for ensuring they comply with all applicable laws and terms of service when using this software. The authors do not condone or support any illegal activities.

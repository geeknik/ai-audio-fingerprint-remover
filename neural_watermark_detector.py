#!/usr/bin/env python3
"""
Neural Network-Based Watermark Detection
Advanced AI-powered detection of modern watermarking techniques.
"""

import numpy as np
import librosa
import scipy.signal as signal
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import joblib
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logger = logging.getLogger(__name__)

@dataclass
class NeuralDetection:
    """Results from neural network watermark detection."""
    method: str
    confidence: float
    location: Tuple[int, int]  # Start, end samples
    watermark_type: str
    ai_platform: str
    features: Dict[str, float]

class NeuralWatermarkDetector:
    """Neural network-based watermark detection for modern AI systems."""
    
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.pca_reducer = PCA(n_components=50)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.clustering_model = DBSCAN(eps=0.5, min_samples=5)
        
        # Pre-trained feature extractors (simulated - in practice would load real models)
        self.is_trained = False
        
        # AI platform signatures
        self.platform_signatures = {
            'suno': {
                'spectral_peaks': [19500, 15500, 8150, 12100],
                'temporal_patterns': ['regular_onset', 'quantized_timing'],
                'harmonic_ratios': [0.618, 1.414, 2.718],  # Golden ratio, sqrt(2), e
                'phase_signatures': ['coherent_high_freq', 'structured_low_freq']
            },
            'openai_jukebox': {
                'spectral_peaks': [11025, 5512, 22050],  # Nyquist-related
                'temporal_patterns': ['hierarchical_structure', 'vq_quantization'],
                'harmonic_ratios': [2.0, 4.0, 8.0],  # Powers of 2
                'phase_signatures': ['vq_artifacts', 'reconstruction_patterns']
            },
            'elevenlabs': {
                'spectral_peaks': [16000, 8000, 4000],  # Speech-focused
                'temporal_patterns': ['prosody_artifacts', 'voice_synthesis'],
                'harmonic_ratios': [1.5, 3.0, 6.0],  # Speech harmonics
                'phase_signatures': ['vocoder_artifacts', 'synthesis_patterns']
            },
            'google_musiclm': {
                'spectral_peaks': [12288, 6144, 3072],  # Powers of 3*2^n
                'temporal_patterns': ['semantic_tokens', 'acoustic_tokens'],
                'harmonic_ratios': [1.732, 2.449, 3.464],  # sqrt(3), sqrt(6), sqrt(12)
                'phase_signatures': ['transformer_artifacts', 'attention_patterns']
            },
            'meta_audiocraft': {
                'spectral_peaks': [20000, 10000, 5000, 2500],
                'temporal_patterns': ['conditioned_generation', 'compression_artifacts'],
                'harmonic_ratios': [1.618, 2.618, 4.236],  # Fibonacci ratios
                'phase_signatures': ['codec_artifacts', 'generation_patterns']
            }
        }
    
    def detect_neural_watermarks(self, audio: np.ndarray, sr: int) -> List[NeuralDetection]:
        """Comprehensive neural watermark detection."""
        detections = []
        
        # Convert to mono
        if audio.ndim > 1:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio
            
        logger.info(f"Running neural watermark analysis on {len(audio_mono)} samples")
        
        # Extract comprehensive features
        features = self._extract_comprehensive_features(audio_mono, sr)
        
        # Platform-specific detection
        for platform, signature in self.platform_signatures.items():
            platform_detections = self._detect_platform_specific(
                audio_mono, sr, platform, signature, features
            )
            detections.extend(platform_detections)
        
        # Anomaly-based detection
        anomaly_detections = self._detect_anomalies(audio_mono, sr, features)
        detections.extend(anomaly_detections)
        
        # Semantic pattern detection
        semantic_detections = self._detect_semantic_patterns(audio_mono, sr, features)
        detections.extend(semantic_detections)
        
        logger.info(f"Neural detection found {len(detections)} potential watermarks")
        return detections
    
    def _extract_comprehensive_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract comprehensive features for neural analysis."""
        features = {}
        
        try:
            # 1. Spectral features
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Mel-frequency features
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            features['mel_spectrogram'] = mel_spec
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features['mfccs'] = mfccs
            
            # Chroma features
            chroma = librosa.feature.chroma(y=audio, sr=sr)
            features['chroma'] = chroma
            
            # Spectral features
            features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)[0]
            
            # 2. Temporal features
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
            features['onset_times'] = librosa.frames_to_time(onset_frames, sr=sr)
            
            # Tempo and beat
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = tempo
            features['beat_times'] = librosa.frames_to_time(beats, sr=sr)
            
            # 3. Harmonic features
            harmonic, percussive = librosa.effects.hpss(audio)
            features['harmonic_ratio'] = np.mean(np.abs(harmonic)) / (np.mean(np.abs(audio)) + 1e-10)
            
            # 4. Phase features
            features['phase_coherence'] = self._calculate_phase_coherence(phase)
            features['phase_entropy'] = self._calculate_phase_entropy(phase)
            
            # 5. Statistical features
            features['amplitude_distribution'] = self._analyze_amplitude_distribution(audio)
            features['spectral_statistics'] = self._analyze_spectral_statistics(magnitude)
            
            # 6. Advanced features
            features['spectral_contrast'] = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features['tonnetz'] = librosa.feature.tonnetz(y=audio, sr=sr)
            
        except Exception as e:
            logger.warning(f"Feature extraction error: {e}")
            
        return features
    
    def _detect_platform_specific(self, audio: np.ndarray, sr: int, platform: str, 
                                 signature: Dict, features: Dict) -> List[NeuralDetection]:
        """Detect platform-specific watermarks."""
        detections = []
        
        try:
            # Analyze spectral peaks
            spectral_score = self._analyze_spectral_peaks(audio, sr, signature['spectral_peaks'])
            
            # Analyze temporal patterns
            temporal_score = self._analyze_temporal_patterns(features, signature['temporal_patterns'])
            
            # Analyze harmonic ratios
            harmonic_score = self._analyze_harmonic_ratios(features, signature['harmonic_ratios'])
            
            # Analyze phase signatures
            phase_score = self._analyze_phase_signatures(features, signature['phase_signatures'])
            
            # Combined confidence
            confidence = (spectral_score * 0.3 + temporal_score * 0.25 + 
                         harmonic_score * 0.25 + phase_score * 0.2)
            
            if confidence > 0.7:  # High confidence threshold
                detections.append(NeuralDetection(
                    method='platform_specific',
                    confidence=confidence,
                    location=(0, len(audio)),
                    watermark_type='ai_signature',
                    ai_platform=platform,
                    features={
                        'spectral_score': spectral_score,
                        'temporal_score': temporal_score,
                        'harmonic_score': harmonic_score,
                        'phase_score': phase_score
                    }
                ))
                
        except Exception as e:
            logger.warning(f"Platform-specific detection failed for {platform}: {e}")
            
        return detections
    
    def _detect_anomalies(self, audio: np.ndarray, sr: int, features: Dict) -> List[NeuralDetection]:
        """Detect anomalies using machine learning."""
        detections = []
        
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            
            if len(feature_vector) == 0:
                return detections
            
            # Reshape for sklearn
            X = feature_vector.reshape(1, -1)
            
            # Simple anomaly detection (in practice, would use pre-trained models)
            # For now, use statistical analysis
            
            # Z-score based anomaly detection
            z_scores = np.abs(stats.zscore(feature_vector))
            anomaly_score = np.mean(z_scores > 3)  # Features > 3 standard deviations
            
            if anomaly_score > 0.1:  # 10% of features are anomalous
                confidence = min(anomaly_score * 5, 1.0)  # Scale to 0-1
                
                detections.append(NeuralDetection(
                    method='anomaly_detection',
                    confidence=confidence,
                    location=(0, len(audio)),
                    watermark_type='statistical_anomaly',
                    ai_platform='unknown',
                    features={
                        'anomaly_score': anomaly_score,
                        'mean_z_score': np.mean(z_scores),
                        'max_z_score': np.max(z_scores)
                    }
                ))
                
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            
        return detections
    
    def _detect_semantic_patterns(self, audio: np.ndarray, sr: int, features: Dict) -> List[NeuralDetection]:
        """Detect semantic patterns that indicate AI generation."""
        detections = []
        
        try:
            # Analyze musical/semantic structure
            semantic_scores = {}
            
            # 1. Repetition analysis
            if 'chroma' in features:
                chroma = features['chroma']
                # Calculate self-similarity matrix
                similarity_matrix = np.dot(chroma.T, chroma)
                # Look for unnatural repetition patterns
                repetition_score = self._analyze_repetition_patterns(similarity_matrix)
                semantic_scores['repetition'] = repetition_score
            
            # 2. Harmonic progression analysis
            if 'chroma' in features:
                harmonic_score = self._analyze_harmonic_progressions(features['chroma'])
                semantic_scores['harmonic_progression'] = harmonic_score
            
            # 3. Rhythmic pattern analysis
            if 'onset_times' in features and 'beat_times' in features:
                rhythmic_score = self._analyze_rhythmic_patterns(
                    features['onset_times'], features['beat_times']
                )
                semantic_scores['rhythmic_pattern'] = rhythmic_score
            
            # 4. Spectral evolution analysis
            if 'mel_spectrogram' in features:
                evolution_score = self._analyze_spectral_evolution(features['mel_spectrogram'])
                semantic_scores['spectral_evolution'] = evolution_score
            
            # Combine semantic scores
            if semantic_scores:
                avg_score = np.mean(list(semantic_scores.values()))
                
                if avg_score > 0.6:  # Semantic anomaly threshold
                    detections.append(NeuralDetection(
                        method='semantic_analysis',
                        confidence=avg_score,
                        location=(0, len(audio)),
                        watermark_type='semantic_pattern',
                        ai_platform='ai_generated',
                        features=semantic_scores
                    ))
                    
        except Exception as e:
            logger.warning(f"Semantic pattern detection failed: {e}")
            
        return detections
    
    def _analyze_spectral_peaks(self, audio: np.ndarray, sr: int, target_peaks: List[float]) -> float:
        """Analyze spectral peaks for platform signatures."""
        try:
            # Compute spectrum
            spectrum = np.abs(np.fft.fft(audio))
            freqs = np.fft.fftfreq(len(audio), 1/sr)[:len(spectrum)//2]
            spectrum = spectrum[:len(spectrum)//2]
            
            score = 0.0
            for target_freq in target_peaks:
                if target_freq < sr/2:
                    # Find nearest frequency bin
                    freq_idx = np.argmin(np.abs(freqs - target_freq))
                    
                    # Analyze peak strength
                    window = 5  # ±5 bins
                    start_idx = max(0, freq_idx - window)
                    end_idx = min(len(spectrum), freq_idx + window)
                    
                    local_spectrum = spectrum[start_idx:end_idx]
                    peak_strength = np.max(local_spectrum)
                    local_mean = np.mean(spectrum)
                    
                    if local_mean > 0:
                        normalized_strength = peak_strength / local_mean
                        score += min(normalized_strength / 10, 1.0)  # Normalize
            
            return score / len(target_peaks)
            
        except Exception as e:
            logger.warning(f"Spectral peak analysis failed: {e}")
            return 0.0
    
    def _analyze_temporal_patterns(self, features: Dict, patterns: List[str]) -> float:
        """Analyze temporal patterns for AI signatures."""
        score = 0.0
        
        try:
            for pattern in patterns:
                if pattern == 'regular_onset' and 'onset_times' in features:
                    onsets = features['onset_times']
                    if len(onsets) > 1:
                        intervals = np.diff(onsets)
                        regularity = 1.0 - np.std(intervals) / (np.mean(intervals) + 1e-10)
                        score += regularity
                
                elif pattern == 'quantized_timing' and 'beat_times' in features:
                    beats = features['beat_times']
                    if len(beats) > 1:
                        intervals = np.diff(beats)
                        # Check for overly precise timing
                        quantization = np.mean(intervals % 0.1 < 0.01)  # 100ms grid
                        score += quantization
            
            return score / len(patterns) if patterns else 0.0
            
        except Exception as e:
            logger.warning(f"Temporal pattern analysis failed: {e}")
            return 0.0
    
    def _analyze_harmonic_ratios(self, features: Dict, target_ratios: List[float]) -> float:
        """Analyze harmonic ratios for AI signatures."""
        try:
            if 'harmonic_ratio' not in features:
                return 0.0
            
            harmonic_ratio = features['harmonic_ratio']
            
            score = 0.0
            for target_ratio in target_ratios:
                # Check if harmonic ratio matches target
                difference = abs(harmonic_ratio - target_ratio)
                if difference < 0.1:  # Close match
                    score += 1.0 - difference * 10
            
            return score / len(target_ratios)
            
        except Exception as e:
            logger.warning(f"Harmonic ratio analysis failed: {e}")
            return 0.0
    
    def _analyze_phase_signatures(self, features: Dict, signatures: List[str]) -> float:
        """Analyze phase signatures for AI artifacts."""
        score = 0.0
        
        try:
            if 'phase_coherence' in features and 'phase_entropy' in features:
                coherence = features['phase_coherence']
                entropy = features['phase_entropy']
                
                for signature in signatures:
                    if signature == 'coherent_high_freq':
                        # High frequency phase coherence suggests artificial generation
                        score += coherence
                    elif signature == 'structured_low_freq':
                        # Low entropy suggests structured phase patterns
                        score += 1.0 - entropy
            
            return score / len(signatures) if signatures else 0.0
            
        except Exception as e:
            logger.warning(f"Phase signature analysis failed: {e}")
            return 0.0
    
    def _calculate_phase_coherence(self, phase: np.ndarray) -> float:
        """Calculate phase coherence across frequencies."""
        try:
            # Calculate coherence in high frequency range
            high_freq_phase = phase[phase.shape[0]//2:, :]
            coherence = np.mean(np.abs(np.cos(high_freq_phase)))
            return coherence
        except:
            return 0.0
    
    def _calculate_phase_entropy(self, phase: np.ndarray) -> float:
        """Calculate phase entropy."""
        try:
            # Flatten and calculate entropy
            phase_flat = phase.flatten()
            hist, _ = np.histogram(phase_flat, bins=50)
            entropy = stats.entropy(hist + 1e-10)
            return entropy / np.log(50)  # Normalize
        except:
            return 0.0
    
    def _analyze_amplitude_distribution(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze amplitude distribution statistics."""
        try:
            return {
                'mean': np.mean(audio),
                'std': np.std(audio),
                'skewness': stats.skew(audio),
                'kurtosis': stats.kurtosis(audio),
                'entropy': stats.entropy(np.histogram(audio, bins=50)[0] + 1e-10)
            }
        except:
            return {}
    
    def _analyze_spectral_statistics(self, magnitude: np.ndarray) -> Dict[str, float]:
        """Analyze spectral statistics."""
        try:
            mean_spectrum = np.mean(magnitude, axis=1)
            return {
                'spectral_mean': np.mean(mean_spectrum),
                'spectral_std': np.std(mean_spectrum),
                'spectral_skew': stats.skew(mean_spectrum),
                'spectral_kurtosis': stats.kurtosis(mean_spectrum),
                'spectral_entropy': stats.entropy(mean_spectrum + 1e-10)
            }
        except:
            return {}
    
    def _prepare_feature_vector(self, features: Dict) -> np.ndarray:
        """Prepare feature vector for machine learning."""
        try:
            feature_list = []
            
            # Statistical features
            if 'amplitude_distribution' in features:
                amp_dist = features['amplitude_distribution']
                feature_list.extend([
                    amp_dist.get('mean', 0),
                    amp_dist.get('std', 0),
                    amp_dist.get('skewness', 0),
                    amp_dist.get('kurtosis', 0),
                    amp_dist.get('entropy', 0)
                ])
            
            if 'spectral_statistics' in features:
                spec_stats = features['spectral_statistics']
                feature_list.extend([
                    spec_stats.get('spectral_mean', 0),
                    spec_stats.get('spectral_std', 0),
                    spec_stats.get('spectral_skew', 0),
                    spec_stats.get('spectral_kurtosis', 0),
                    spec_stats.get('spectral_entropy', 0)
                ])
            
            # Aggregate features from multi-dimensional arrays
            for key in ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'zero_crossing_rate']:
                if key in features:
                    arr = features[key]
                    feature_list.extend([
                        np.mean(arr),
                        np.std(arr),
                        np.min(arr),
                        np.max(arr)
                    ])
            
            return np.array(feature_list)
            
        except Exception as e:
            logger.warning(f"Feature vector preparation failed: {e}")
            return np.array([])
    
    def _analyze_repetition_patterns(self, similarity_matrix: np.ndarray) -> float:
        """Analyze repetition patterns in self-similarity matrix."""
        try:
            # Look for diagonal structures (repetitions)
            diagonal_strength = 0.0
            for offset in range(1, min(20, similarity_matrix.shape[0])):
                diagonal = np.diag(similarity_matrix, offset)
                diagonal_strength += np.mean(diagonal)
            
            # Normalize
            return diagonal_strength / 20
        except:
            return 0.0
    
    def _analyze_harmonic_progressions(self, chroma: np.ndarray) -> float:
        """Analyze harmonic progressions for AI signatures."""
        try:
            # Simple analysis of chord transitions
            chord_changes = np.diff(chroma, axis=1)
            change_magnitude = np.linalg.norm(chord_changes, axis=0)
            
            # AI-generated music might have unnatural progression patterns
            change_regularity = 1.0 - np.std(change_magnitude) / (np.mean(change_magnitude) + 1e-10)
            return change_regularity
        except:
            return 0.0
    
    def _analyze_rhythmic_patterns(self, onsets: np.ndarray, beats: np.ndarray) -> float:
        """Analyze rhythmic patterns for AI signatures."""
        try:
            if len(onsets) > 1 and len(beats) > 1:
                onset_intervals = np.diff(onsets)
                beat_intervals = np.diff(beats)
                
                # Check for overly regular patterns
                onset_regularity = 1.0 - np.std(onset_intervals) / (np.mean(onset_intervals) + 1e-10)
                beat_regularity = 1.0 - np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-10)
                
                return (onset_regularity + beat_regularity) / 2
            return 0.0
        except:
            return 0.0
    
    def _analyze_spectral_evolution(self, mel_spec: np.ndarray) -> float:
        """Analyze spectral evolution for AI signatures."""
        try:
            # Analyze how spectrum evolves over time
            temporal_diff = np.diff(mel_spec, axis=1)
            evolution_variance = np.var(temporal_diff)
            
            # AI-generated audio might have unnatural spectral evolution
            # Too much variance suggests chaotic generation
            # Too little variance suggests overly smooth generation
            
            # Normalize variance (target natural range)
            natural_variance_range = [0.1, 2.0]
            if evolution_variance < natural_variance_range[0]:
                return 1.0 - evolution_variance / natural_variance_range[0]
            elif evolution_variance > natural_variance_range[1]:
                return min(evolution_variance / natural_variance_range[1] - 1.0, 1.0)
            else:
                return 0.0
        except:
            return 0.0

def main():
    """Test the neural watermark detector."""
    import argparse
    import soundfile as sf
    
    parser = argparse.ArgumentParser(description="Neural Watermark Detection")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Load audio
        audio, sr = sf.read(args.input)
        print(f"Analyzing {args.input} ({len(audio)} samples at {sr} Hz)")
        
        # Run detection
        detector = NeuralWatermarkDetector()
        detections = detector.detect_neural_watermarks(audio, sr)
        
        print(f"\nNeural Watermark Analysis Results:")
        print(f"==================================")
        
        if detections:
            for detection in detections:
                print(f"Method: {detection.method}")
                print(f"AI Platform: {detection.ai_platform}")
                print(f"Watermark Type: {detection.watermark_type}")
                print(f"Confidence: {detection.confidence:.3f}")
                print(f"Location: {detection.location[0]:,} - {detection.location[1]:,} samples")
                print(f"Features: {detection.features}")
                print("-" * 50)
        else:
            print("No neural watermarks detected.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
"""
Audio processing module for speech-to-text conversion and audio preprocessing.

This module handles various audio formats and provides high-quality transcription
using multiple speech recognition engines with fallback options.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json

import librosa
import soundfile as sf
import speech_recognition as sr
import whisper
from pydub import AudioSegment
import numpy as np


class AudioProcessor:
    """
    Advanced audio processor with multiple transcription engines and preprocessing.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the audio processor with configuration.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        # Load Whisper model for high-quality transcription
        self.whisper_model = self._load_whisper_model()
        
        # Audio processing parameters
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.chunk_duration = self.config.get('chunk_duration', 30)  # seconds
        self.overlap_duration = self.config.get('overlap_duration', 2)  # seconds
    
    def _load_whisper_model(self) -> whisper.Whisper:
        """Load the Whisper model for transcription."""
        model_size = self.config.get('whisper_model', 'base')
        try:
            model = whisper.load_model(model_size)
            self.logger.info(f"Loaded Whisper model: {model_size}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    async def process_audio_file(self, file_path: str) -> Dict:
        """
        Process an audio file and return transcription with metadata.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary containing transcript, confidence scores, and metadata
        """
        self.logger.info(f"Processing audio file: {file_path}")
        
        # Validate and load audio
        audio_data, metadata = await self._load_and_validate_audio(file_path)
        
        # Preprocess audio
        processed_audio = self._preprocess_audio(audio_data, metadata['original_sample_rate'])
        
        # Chunk audio for processing
        chunks = self._chunk_audio(processed_audio, metadata)
        
        # Transcribe chunks
        transcription_results = await self._transcribe_chunks(chunks)
        
        # Combine and post-process results
        final_transcript = self._combine_transcripts(transcription_results)
        
        return {
            'transcript': final_transcript['text'],
            'confidence': final_transcript.get('confidence', 0.0),
            'metadata': {
                **metadata,
                'processing_time': final_transcript.get('processing_time', 0),
                'chunks_processed': len(chunks),
                'transcription_engine': 'whisper_primary'
            },
            'segments': final_transcript.get('segments', [])
        }
    
    async def _load_and_validate_audio(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        """Load and validate audio file."""
        try:
            # Convert file path to Path object
            audio_path = Path(file_path)
            
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            # Load audio using librosa for consistency
            audio_data, sample_rate = librosa.load(
                str(audio_path),
                sr=None,  # Keep original sample rate initially
                mono=True
            )
            
            # Get file metadata
            metadata = {
                'original_sample_rate': sample_rate,
                'duration': len(audio_data) / sample_rate,
                'channels': 1,  # We convert to mono
                'file_size': audio_path.stat().st_size,
                'file_format': audio_path.suffix.lower()
            }
            
            self.logger.info(f"Loaded audio: {metadata['duration']:.2f}s, {sample_rate}Hz")
            
            return audio_data, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load audio file {file_path}: {e}")
            raise
    
    def _preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Preprocess audio for better transcription quality."""
        # Resample to target sample rate if needed
        if sample_rate != self.sample_rate:
            audio_data = librosa.resample(
                audio_data,
                orig_sr=sample_rate,
                target_sr=self.sample_rate
            )
        
        # Normalize audio
        audio_data = librosa.util.normalize(audio_data)
        
        # Remove silence
        audio_data = self._remove_silence(audio_data)
        
        # Apply noise reduction (basic)
        audio_data = self._reduce_noise(audio_data)
        
        return audio_data
    
    def _remove_silence(self, audio_data: np.ndarray) -> np.ndarray:
        """Remove silent segments from audio."""
        # Use librosa to detect non-silent intervals
        intervals = librosa.effects.split(
            audio_data,
            top_db=20,  # Threshold for silence detection
            frame_length=2048,
            hop_length=512
        )
        
        # Concatenate non-silent segments
        non_silent_audio = []
        for start, end in intervals:
            non_silent_audio.append(audio_data[start:end])
        
        if non_silent_audio:
            return np.concatenate(non_silent_audio)
        else:
            return audio_data  # Return original if no non-silent segments found
    
    def _reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply basic noise reduction to audio."""
        # Simple spectral subtraction-based noise reduction
        # This is a basic implementation; more sophisticated methods can be added
        
        # Compute spectral statistics
        spectral = librosa.stft(audio_data)
        magnitude = np.abs(spectral)
        
        # Estimate noise floor from first 0.5 seconds
        noise_samples = int(0.5 * self.sample_rate)
        noise_floor = np.mean(magnitude[:, :noise_samples // 512], axis=1, keepdims=True)
        
        # Apply spectral subtraction
        clean_magnitude = np.maximum(magnitude - 0.5 * noise_floor, 0.1 * magnitude)
        
        # Reconstruct audio
        clean_spectral = clean_magnitude * np.exp(1j * np.angle(spectral))
        clean_audio = librosa.istft(clean_spectral)
        
        return clean_audio
    
    def _chunk_audio(self, audio_data: np.ndarray, metadata: Dict) -> List[Dict]:
        """Split audio into overlapping chunks for processing."""
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        overlap_samples = int(self.overlap_duration * self.sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(audio_data):
            end_idx = min(start_idx + chunk_samples, len(audio_data))
            
            chunk_data = audio_data[start_idx:end_idx]
            
            # Skip very short chunks
            if len(chunk_data) < self.sample_rate:  # Less than 1 second
                break
            
            chunks.append({
                'audio': chunk_data,
                'start_time': start_idx / self.sample_rate,
                'end_time': end_idx / self.sample_rate,
                'duration': len(chunk_data) / self.sample_rate,
                'chunk_id': len(chunks)
            })
            
            start_idx += step_samples
        
        self.logger.info(f"Created {len(chunks)} audio chunks")
        return chunks
    
    async def _transcribe_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Transcribe audio chunks using Whisper."""
        results = []
        
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Transcribing chunk {i+1}/{len(chunks)}")
            
            try:
                # Transcribe with Whisper
                result = self.whisper_model.transcribe(
                    chunk['audio'],
                    language='en',  # Can be made configurable
                    task='transcribe',
                    verbose=False
                )
                
                chunk_result = {
                    'chunk_id': chunk['chunk_id'],
                    'start_time': chunk['start_time'],
                    'end_time': chunk['end_time'],
                    'text': result['text'].strip(),
                    'confidence': self._calculate_confidence(result),
                    'segments': result.get('segments', [])
                }
                
                results.append(chunk_result)
                
            except Exception as e:
                self.logger.error(f"Failed to transcribe chunk {i}: {e}")
                # Add empty result to maintain chunk order
                results.append({
                    'chunk_id': chunk['chunk_id'],
                    'start_time': chunk['start_time'],
                    'end_time': chunk['end_time'],
                    'text': '',
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        return results
    
    def _calculate_confidence(self, whisper_result: Dict) -> float:
        """Calculate overall confidence score from Whisper result."""
        if 'segments' not in whisper_result:
            return 0.5  # Default confidence if no segments
        
        segments = whisper_result['segments']
        if not segments:
            return 0.5
        
        # Calculate average confidence from segments
        total_confidence = 0
        total_duration = 0
        
        for segment in segments:
            duration = segment.get('end', 0) - segment.get('start', 0)
            confidence = segment.get('avg_logprob', -1.0)
            
            # Convert log probability to confidence score (0-1)
            confidence_score = np.exp(confidence) if confidence > -5 else 0.1
            
            total_confidence += confidence_score * duration
            total_duration += duration
        
        if total_duration > 0:
            return min(total_confidence / total_duration, 1.0)
        else:
            return 0.5
    
    def _combine_transcripts(self, chunk_results: List[Dict]) -> Dict:
        """Combine chunk transcripts into final transcript."""
        # Sort chunks by start time
        chunk_results.sort(key=lambda x: x['start_time'])
        
        # Combine text
        combined_text = []
        combined_segments = []
        total_confidence = 0
        valid_chunks = 0
        
        for chunk in chunk_results:
            if chunk.get('text') and not chunk.get('error'):
                combined_text.append(chunk['text'])
                combined_segments.extend(chunk.get('segments', []))
                total_confidence += chunk.get('confidence', 0)
                valid_chunks += 1
        
        # Calculate overall confidence
        overall_confidence = total_confidence / valid_chunks if valid_chunks > 0 else 0.0
        
        # Clean up text
        final_text = ' '.join(combined_text)
        final_text = self._clean_transcript(final_text)
        
        return {
            'text': final_text,
            'confidence': overall_confidence,
            'segments': combined_segments,
            'valid_chunks': valid_chunks,
            'total_chunks': len(chunk_results)
        }
    
    def _clean_transcript(self, text: str) -> str:
        """Clean and normalize the transcript text."""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common transcription artifacts
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        text = text.replace(' ?', '?')
        text = text.replace(' !', '!')
        
        # Capitalize first letter of sentences
        sentences = text.split('. ')
        sentences = [sent.capitalize() if sent else sent for sent in sentences]
        text = '. '.join(sentences)
        
        return text.strip()


# Utility functions for audio processing
def supported_formats() -> List[str]:
    """Return list of supported audio formats."""
    return ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma']


def convert_audio_format(input_path: str, output_path: str, target_format: str = 'wav') -> str:
    """
    Convert audio file to target format using pydub.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for converted output file
        target_format: Target audio format
        
    Returns:
        Path to converted file
    """
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format=target_format)
        return output_path
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to convert {input_path}: {e}")
        raise

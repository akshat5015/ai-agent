"""
Video processing module to extract audio transcript and on-screen text (OCR).

This module supports:
- Audio extraction from video and transcription via existing AudioProcessor
- Frame sampling and OCR using EasyOCR (fallback-friendly)
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np

try:
    import easyocr  # type: ignore
    _EASYOCR_AVAILABLE = True
except Exception:
    _EASYOCR_AVAILABLE = False

from .audio_processor import AudioProcessor


@dataclass
class VideoOCRConfig:
    frame_sample_rate: float = 1.0  # frames per second to OCR
    min_text_length: int = 3
    languages: Tuple[str, ...] = ("en",)
    detect_captions_only: bool = False  # if True, focus on bottom area heuristics


class VideoProcessor:
    """Process videos to obtain transcript (audio) + OCR text."""

    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.audio_processor = AudioProcessor(self.config.get("audio_processing", {}))

        ocr_cfg = self.config.get("video_processing", {}).get("ocr", {})
        self.ocr_config = VideoOCRConfig(
            frame_sample_rate=ocr_cfg.get("frame_sample_rate", 1.0),
            min_text_length=ocr_cfg.get("min_text_length", 3),
            languages=tuple(ocr_cfg.get("languages", ["en"])),
            detect_captions_only=ocr_cfg.get("detect_captions_only", False),
        )

        self.reader: Optional[Any] = None
        if _EASYOCR_AVAILABLE:
            try:
                self.reader = easyocr.Reader(list(self.ocr_config.languages), gpu=self.config.get("advanced", {}).get("gpu_acceleration", True))
            except Exception as e:
                self.logger.warning(f"Failed to initialize EasyOCR: {e}. OCR will be disabled.")
                self.reader = None
        else:
            self.logger.warning("EasyOCR is not available. Install 'easyocr' to enable on-screen text extraction.")

    async def process_video_file(self, file_path: str) -> Dict[str, Any]:
        """Process a video file: transcribe audio + OCR frames; return combined results."""
        video_path = Path(file_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")

        self.logger.info(f"Processing video file: {file_path}")

        # 1) Transcribe audio via AudioProcessor (MoviePy extracts audio when needed by librosa)
        audio_result = await self._transcribe_video_audio(file_path)

        # 2) OCR sampled frames
        ocr_text, ocr_meta = await self._extract_ocr_text(file_path)

        # 3) Combine transcript and OCR into a single text for downstream summarization
        combined_text = self._combine_transcript_and_ocr(audio_result.get("transcript", ""), ocr_text)

        duration = audio_result.get("metadata", {}).get("duration")
        metadata: Dict[str, Any] = {
            "duration": duration,
            "file_format": video_path.suffix.lower(),
            "ocr": ocr_meta,
            "audio": audio_result.get("metadata", {}),
        }

        return {
            "transcript": combined_text,
            "confidence": audio_result.get("confidence", 0.0),
            "metadata": metadata,
            "segments": audio_result.get("segments", []),
        }

    async def _transcribe_video_audio(self, file_path: str) -> Dict[str, Any]:
        """Delegate to AudioProcessor by passing the video path; librosa can read audio streams for many containers."""
        try:
            return await self.audio_processor.process_audio_file(file_path)
        except Exception as e:
            self.logger.warning(f"Direct audio read failed for video. Error: {e}. Attempting MoviePy extraction.")
            # Fallback: extract audio to temp WAV via MoviePy
            import tempfile
            from moviepy.editor import VideoFileClip  # type: ignore

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_wav = tmp.name
            try:
                clip = VideoFileClip(file_path)
                if clip.audio is None:
                    raise RuntimeError("No audio track found in video.")
                clip.audio.write_audiofile(tmp_wav, logger=None)
                return await self.audio_processor.process_audio_file(tmp_wav)
            finally:
                try:
                    Path(tmp_wav).unlink(missing_ok=True)
                except Exception:
                    pass

    async def _extract_ocr_text(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Sample frames and run OCR to extract on-screen text."""
        if self.reader is None:
            return "", {"enabled": False, "reason": "no_ocr_reader"}

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return "", {"enabled": False, "reason": "cv2_open_failed"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        step = max(int(fps // self.ocr_config.frame_sample_rate), 1)

        texts: List[str] = []
        frames_scanned = 0
        ocr_hits = 0

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            roi = frame
            if self.ocr_config.detect_captions_only:
                # Heuristic: bottom 25% often contains captions
                y0 = int(height * 0.75)
                roi = frame[y0:height, 0:width]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            # Adaptive threshold helps subtitle-like text
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 10)

            result = self.reader.readtext(th)
            extracted = [r[1] for r in result if isinstance(r, (list, tuple)) and len(r) >= 2]
            extracted = [t.strip() for t in extracted if len(t.strip()) >= self.ocr_config.min_text_length]
            if extracted:
                texts.extend(extracted)
                ocr_hits += 1

            frames_scanned += 1
            frame_idx += 1

        cap.release()

        # Lightweight deduplication
        deduped = []
        seen = set()
        for t in texts:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(t)

        joined_text = " \n".join(deduped)
        meta = {
            "enabled": True,
            "frames_scanned": frames_scanned,
            "ocr_hits": ocr_hits,
            "fps": fps,
            "frame_sample_rate": self.ocr_config.frame_sample_rate,
            "resolution": {"width": width, "height": height},
            "unique_lines": len(deduped),
        }
        return joined_text, meta

    def _combine_transcript_and_ocr(self, transcript: str, ocr_text: str) -> str:
        if not ocr_text:
            return transcript
        if not transcript:
            return f"[ON-SCREEN TEXT]\n{ocr_text}"
        return transcript + "\n\n[ON-SCREEN TEXT]\n" + ocr_text



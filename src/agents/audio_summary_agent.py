"""
Main AI Agent for Audio-to-Summary processing with reasoning and planning capabilities.

This agent coordinates the entire pipeline from audio input to summary and task generation,
using fine-tuned models and implementing reasoning strategies.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from src.processors.audio_processor import AudioProcessor
from src.processors.video_processor import VideoProcessor
from src.models.fine_tuned_summarizer import FineTunedSummarizer
from src.extractors.task_extractor import TaskExtractor
from src.evaluators.quality_assessor import QualityAssessor


class AudioSummaryAgent:
    """
    Intelligent AI agent that processes audio recordings into summaries and actionable tasks.
 
    This agent implements reasoning and planning capabilities to optimize the processing
    workflow based on content type, length, and user requirements.
    """

    def __init__(self, config: Dict):
        """
        Initialize the AI agent with configuration.

        Args:
            config: Configuration settings for the agent
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize component processors
        self.audio_processor = AudioProcessor(config.get('audio_processing', {}))
        self.video_processor = VideoProcessor(config) # Assuming VideoProcessor also uses the root config
        # --- FIX #1: Correct config key for summarizer (Already done) ---
        self.summarizer = FineTunedSummarizer(config.get('model', {}))
        # -----------------------------------------------
        self.task_extractor = TaskExtractor(config.get('task_extraction', {}))
        self.quality_assessor = QualityAssessor(config.get('evaluation', {}))

        # Agent reasoning state
        self.processing_history = []
        self.current_context = {} # Store current processing context if needed

        # Performance tracking
        self.metrics = {
            'total_processed': 0,
            'successful_runs': 0, # Added for success rate calculation
            'failed_runs': 0,     # Added for success rate calculation
            'total_processing_time': 0, # Sum of processing times
            'average_processing_time': 0.0,
            'success_rate': 1.0 # Start with 100% success rate
        }
    
    # This function is kept for compatibility but `process_media` is the new primary entry point
    async def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Main processing method that orchestrates the complete pipeline.
        This now routes to the new process_media function.
        """
        self.logger.info(f"process_audio called, routing to process_media for: {audio_path}")
        return await self.process_media(audio_path)


    async def process_media(self, media_path: str) -> Dict[str, Any]:
        """Process audio or video media; routes to appropriate pipeline and merges OCR for video."""
        start_time = time.time()
        success = False
        processing_time = 0.0 # Initialize processing_time
        
        try:
            suffix = Path(media_path).suffix.lower().lstrip('.')
            # Broader video formats list
            is_video = suffix in { 'mp4', 'mkv', 'mov', 'avi', 'webm', 'm4v', 'flv', 'wmv' } 

            # Plan
            processing_plan = await self._analyze_and_plan(media_path)
            self.current_context = {'plan': processing_plan, 'file': media_path}

            # Process
            if is_video:
                self.logger.info(f"Detected video file, routing to VideoProcessor: {media_path}")
                media_results = await self.video_processor.process_video_file(media_path)
            else:
                self.logger.info(f"Detected audio file, routing to AudioProcessor: {media_path}")
                media_results = await self._process_audio_phase(media_path, processing_plan)

            if not media_results.get('transcript'):
                raise ValueError("Processing failed to produce a transcript.")

            # Summarize
            summary_results = await self._process_summary_phase(
                media_results['transcript'], processing_plan
            )

            # Tasks
            task_results = await self._process_task_phase(
                media_results['transcript'], summary_results.get('summary', ''), processing_plan
            )

            # Quality
            quality_results = await self._assess_quality_phase(
                media_results, summary_results, task_results
            )

            final_results = await self._finalize_results(
                media_results, summary_results, task_results, quality_results, processing_plan
            )
            
            processing_time = time.time() - start_time
            self.logger.info(f"Media processing completed successfully in {processing_time:.2f} seconds")
            success = True
            return final_results
        
        except Exception as e:
            processing_time = time.time() - start_time # Record time even on failure
            self.logger.error(f"Media processing failed for {media_path}: {e}", exc_info=True)
            return {"error": str(e), "status": "failed", "processing_time": processing_time}
        
        finally:
            # Update metrics regardless of success or failure
            await self._update_processing_metrics(processing_time, success)
            self.current_context = {} # Clear context after processing

    async def _analyze_and_plan(self, media_path: str) -> Dict[str, Any]:
        """
        Analyze the input and create an optimal processing plan.

        This implements the reasoning capability of the AI agent.
        """
        self.logger.info(f"Analyzing input and creating processing plan for: {media_path}")

        # Get basic file information
        file_path = Path(media_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input media file not found: {media_path}")

        file_info = {
            'size_mb': file_path.stat().st_size / (1024 * 1024) if file_path.stat().st_size > 0 else 0,
            'format': file_path.suffix.lower(),
            'name': file_path.stem
        }

        # Estimate duration (basic heuristic)
        estimated_duration_minutes = self._estimate_audio_duration(file_info)

        # Determine content type based on filename patterns
        content_type = self._classify_content_type(file_info['name'])

        # Create processing plan
        plan = {
            'content_type': content_type,
            'estimated_duration_minutes': estimated_duration_minutes,
            'file_info': file_info,
            'processing_strategy': self._select_processing_strategy(content_type, estimated_duration_minutes),
            'summary_style': self._select_summary_style(content_type),
            'task_extraction_focus': self._select_task_focus(content_type),
            'quality_thresholds': self._get_quality_thresholds(content_type)
        }

        self.logger.info(f"Processing plan created: Strategy='{plan['processing_strategy']}', Type='{content_type}', Est. Duration='{estimated_duration_minutes:.1f} mins'")
        return plan

    def _estimate_audio_duration(self, file_info: Dict) -> float:
        """Estimate audio/media duration based on file size and format (in minutes)."""
        size_mb = file_info.get('size_mb', 0)
        format_type = file_info.get('format', '.tmp')

        # Rough estimates (minutes per MB) - adjust based on real-world testing if needed
        # Added common video formats
        compression_rates = {
            # Audio
            '.wav': 0.1,   # ~10MB per minute for standard WAV
            '.mp3': 1.0,   # ~1MB per minute for 128kbps MP3
            '.m4a': 1.2,   # Slightly better compression than MP3 typically
            '.flac': 0.25, # Lossless, larger than MP3 but smaller than WAV
            '.ogg': 1.5,   # High compression
            '.aac': 1.2,
            '.wma': 1.0,
            # Video (very rough estimates, highly dependent on resolution/codec)
            '.mp4': 0.1,   # ~10MB/min for 720p h.264
            '.mov': 0.1,
            '.avi': 0.08,
            '.mkv': 0.1,
            '.webm': 0.15,
            '.flv': 0.2,
            '.wmv': 0.12,
        }

        # Use a default rate if format unknown or size is zero
        # Defaulting to 1.0 (like MP3) is a reasonable guess
        rate_mb_per_min = compression_rates.get(format_type, 1.0) 
        if rate_mb_per_min == 0: rate_mb_per_min = 1.0 # Avoid division by zero

        # Calculate duration in minutes: duration = size / rate
        estimated_duration = size_mb / rate_mb_per_min if size_mb > 0 else 0.0

        return estimated_duration

    def _classify_content_type(self, filename: str) -> str:
        """Classify content type based on filename patterns."""
        filename_lower = filename.lower()

        # Define keywords for different types
        academic_keywords = ['lecture', 'class', 'seminar', 'tutorial', 'course', 'lesson', 'prof', 'university', 'college']
        meeting_keywords = ['meeting', 'conference', 'call', 'discussion', 'standup', 'sync', 'huddle', 'briefing', 'minutes', 'agenda']
        study_keywords = ['study', 'review', 'notes', 'exam', 'homework', 'prep', 'session']
        podcast_keywords = ['podcast', 'episode', 'interview', 'show', 'ep']

        if any(keyword in filename_lower for keyword in academic_keywords):
            return 'academic'
        elif any(keyword in filename_lower for keyword in meeting_keywords):
            return 'meeting'
        elif any(keyword in filename_lower for keyword in study_keywords):
            return 'study_session'
        elif any(keyword in filename_lower for keyword in podcast_keywords):
            return 'podcast' # Added podcast type
        else:
            return 'general'

    def _select_processing_strategy(self, content_type: str, duration_minutes: float) -> str:
        """Select optimal processing strategy based on content analysis."""
        # Note: 'chunked_parallel' isn't implemented, defaulting to sequential
        # AudioProcessor *always* chunks, so this is more for logging/future use
        if duration_minutes > 60:  # More than 1 hour
            self.logger.info(f"Media duration > 60 mins ({duration_minutes:.1f}), flagging for 'chunked_sequential' strategy.")
            return 'chunked_sequential'
        else: # Less than 60 minutes
            self.logger.info(f"Media duration <= 60 mins ({duration_minutes:.1f}), flagging for 'chunked_sequential' (default).")
            return 'chunked_sequential' # Defaulting to chunked as it's robust

    def _select_summary_style(self, content_type: str) -> str:
        """Select summary style based on content type."""
        style_mapping = {
            'academic': 'structured_academic',
            'meeting': 'action_oriented',
            'study_session': 'key_points',
            'podcast': 'key_points', # Added style for podcast
            'general': 'comprehensive'
        }
        return style_mapping.get(content_type, 'comprehensive')

    def _select_task_focus(self, content_type: str) -> str:
        """Select task extraction focus based on content type."""
        focus_mapping = {
            'academic': 'assignments_deadlines',
            'meeting': 'action_items',
            'study_session': 'study_tasks',
            'podcast': 'general_todos', # Podcasts usually don't have specific tasks like meetings
            'general': 'general_todos'
        }
        return focus_mapping.get(content_type, 'general_todos')

    def _get_quality_thresholds(self, content_type: str) -> Dict[str, float]:
        """Get quality thresholds based on content type."""
        # Use defaults from config first
        base_thresholds = self.config.get('evaluation', {}).get('thresholds', {
             'transcript_confidence': 0.8,
             'summary_quality': 0.7,
             'task_relevance': 0.75,
             'overall_quality': 0.7 # Added overall threshold
        })

        # Override based on content type if needed (example adjustments)
        if content_type == 'academic':
            base_thresholds['transcript_confidence'] = max(base_thresholds.get('transcript_confidence', 0.8), 0.85) # Ensure higher accuracy
        elif content_type == 'meeting':
            base_thresholds['task_relevance'] = max(base_thresholds.get('task_relevance', 0.75), 0.8) # Ensure higher relevance

        self.logger.info(f"Using quality thresholds for {content_type}: {base_thresholds}")
        return base_thresholds

    async def _process_audio_phase(self, audio_path: str, plan: Dict) -> Dict[str, Any]:
        """Process audio to transcript using the planned strategy."""
        self.logger.info("Starting audio processing phase...")
        
        # Pass the audio_processing config from the main config
        audio_config = self.config.get('audio_processing', {})
        # Re-initialize processor with the specific config for this run if needed,
        # or ensure config passed in __init__ is used correctly.
        # Assuming __init__ config is used by AudioProcessor.
        
        audio_results = await self.audio_processor.process_audio_file(audio_path)

        # --- FIX #2: Robust Confidence Check (Already done) ---
        # Validate transcript quality using the robust block
        transcript_confidence_value = audio_results.get('confidence', 0.0) # Ensure we get the float value, default to 0.0
        threshold = plan.get('quality_thresholds', {}).get('transcript_confidence', 0.8) # Safely get the threshold float

        # Ensure transcript_confidence_value is treated as a float
        try:
            confidence_float = float(transcript_confidence_value)
            if confidence_float < threshold:
                self.logger.warning(f"Low transcript confidence: {confidence_float:.2f} (Threshold: {threshold})")
                # Optional: Could add logic here to maybe try a different Whisper model size if configured
                # Or raise a specific exception if confidence is too low to proceed
                # raise ValueError(f"Transcript confidence {confidence_float:.2f} is below threshold {threshold}")
        except (TypeError, ValueError):
            self.logger.error(f"Could not compare confidence. Confidence value was not a number: {transcript_confidence_value}")
            # Decide how to handle this - proceed with caution or raise error?
            # For now, log the error and proceed.
        # ----------------------------------------

        return audio_results


    async def _process_summary_phase(self, transcript: str, plan: Dict) -> Dict[str, Any]:
        """Generate summary using fine-tuned model."""
        self.logger.info("Starting summarization phase...")
        if not transcript:
             self.logger.warning("Transcript is empty, skipping summarization.")
             return {'summary': '', 'metadata': {'style': plan['summary_style'], 'status': 'skipped'}}

        # Configure summarizer based on plan
        # Get max_length from UI/plan if available, otherwise calculate dynamically
        ui_max_length = self.config.get('summarization', {}).get('max_length') # Check if UI set a specific length
        calculated_max_length = self._calculate_summary_length(len(transcript))
        final_max_length = ui_max_length if ui_max_length else calculated_max_length

        # --- THIS IS THE FIX ---
        # Get min_length from the model's config, defaulting to 150 for longer summaries
        final_min_length = self.config.get('model', {}).get('min_summary_length', 150)

        summary_config = {
            'style': plan['summary_style'],
            'content_type': plan['content_type'],
            'max_length': final_max_length, # Use the determined max length
            'min_length': final_min_length # Pass the minimum length
        }
        # ----------------------
        
        self.logger.info(f"Generating summary with style '{summary_config['style']}', min_length {summary_config['min_length']}, max_length {summary_config['max_length']}")

        # Generate summary
        summary_results = await self.summarizer.generate_summary(
            transcript, summary_config
        )

        return summary_results

    async def _process_task_phase(
        self,
        transcript: str,
        summary: str,
        plan: Dict
    ) -> Dict[str, Any]:
        """Extract actionable tasks from transcript and summary."""
        self.logger.info("Starting task extraction phase...")
        if not transcript and not summary:
            self.logger.warning("Both transcript and summary are empty, skipping task extraction.")
            return {'tasks': [], 'metadata': {'focus': plan['task_extraction_focus'], 'status': 'skipped'}}


        # Configure task extractor based on plan and potentially UI overrides
        task_config = {
            'focus': plan['task_extraction_focus'],
            'content_type': plan['content_type'],
            # Allow override from main config if provided (e.g., from UI)
            'confidence_threshold': self.config.get('task_extraction', {}).get('confidence_threshold'),
        }
        # Remove None values so TaskExtractor uses its defaults if not overridden
        task_config = {k: v for k, v in task_config.items() if v is not None}


        self.logger.info(f"Extracting tasks with focus '{task_config.get('focus', 'default')}'")

        # Extract tasks
        task_results = await self.task_extractor.extract_tasks(
            transcript or "", summary or "", task_config # Pass empty strings if None
        )

        return task_results

    async def _assess_quality_phase(
        self,
        media_results: Dict, # Changed from audio_results
        summary_results: Dict,
        task_results: Dict
    ) -> Dict[str, Any]:
        """Assess quality of all generated outputs."""
        self.logger.info("Starting quality assessment phase...")

        # Ensure results are not None before passing
        media_results = media_results or {} # Changed from audio_results
        summary_results = summary_results or {}
        task_results = task_results or {}


        quality_results = await self.quality_assessor.assess_outputs(
            media_results, summary_results, task_results # Changed from audio_results
        )
        self.logger.info(f"Quality assessment completed. Overall score: {quality_results.get('overall_quality', {}).get('score', 'N/A'):.2f}")

        return quality_results

    async def _finalize_results(
        self,
        media_results: Dict, # Changed from audio_results
        summary_results: Dict,
        task_results: Dict,
        quality_results: Dict,
        plan: Dict # Pass plan for context
    ) -> Dict[str, Any]:
        """Combine and finalize all results."""
        self.logger.info("Finalizing results...")

        # Ensure safe access to potentially missing keys
        media_results = media_results or {} # Changed from audio_results
        summary_results = summary_results or {}
        task_results = task_results or {}
        quality_results = quality_results or {}


        final_results = {
            'transcript': media_results.get('transcript', ''),
            'transcript_metadata': media_results.get('metadata', {}),
            # Include OCR text if present (from VideoProcessor)
            'ocr_text': media_results.get('ocr_text', None), 
            'summary': summary_results.get('summary', ''),
            'summary_metadata': summary_results.get('metadata', {}),
            'tasks': task_results.get('tasks', []),
            'task_metadata': task_results.get('metadata', {}),
            'quality_scores': quality_results,
            'processing_info': {
                'agent_version': '1.1.0 (Video Enabled)', # Updated version
                'processing_timestamp': time.time(),
                'content_type_detected': plan.get('content_type', 'unknown'),
                'strategy_used': plan.get('processing_strategy', 'unknown'),
                'file_info': plan.get('file_info', {})
            }
        }
        
        # Clean up final_results: remove None values like ocr_text if it's empty
        if not final_results['ocr_text']:
            del final_results['ocr_text']

        # Add structured insights based on the *final* results
        final_results['insights'] = self._generate_insights(final_results) # Generate insights at the very end

        self.logger.info("Results finalized.")
        return final_results

    def _calculate_summary_length(self, transcript_length: int) -> int:
        """Calculate optimal summary length based on transcript length."""
        if transcript_length <= 0: return self.config.get('model', {}).get('min_summary_length', 50) # Return min if no transcript

        # Aim for a target ratio, but bounded by min/max from config
        base_ratio = 0.25 # Make summaries longer by default
        min_length = self.config.get('model', {}).get('min_summary_length', 150) # Increased default min
        max_length = self.config.get('model', {}).get('max_summary_length', 1024) # Allow longer summaries

        target_length = int(transcript_length * base_ratio)

        # Clamp the target length between min and max
        final_length = max(min_length, min(target_length, max_length))
        self.logger.debug(f"Calculated summary length: {final_length} (Transcript: {transcript_length}, Ratio: {base_ratio}, Min: {min_length}, Max: {max_length})")
        return final_length

    def _generate_insights(self, results: Dict) -> Dict[str, Any]:
        """Generate high-level insights from processing results."""
        insights = {
            'content_analysis': {},
            'quality_assessment_summary': {},
            'recommendations': []
        }

        # Safe extraction of values
        transcript_len = len(results.get('transcript', ''))
        ocr_len = len(results.get('ocr_text', ''))
        summary_len = len(results.get('summary', ''))
        tasks_count = len(results.get('tasks', []))
        overall_quality_dict = results.get('quality_scores', {}).get('overall_quality', {})
        overall_score = overall_quality_dict.get('score', 0.0)
        meets_threshold = overall_quality_dict.get('meets_threshold', False)

        insights['content_analysis'] = {
            'transcript_length': transcript_len,
            'ocr_text_length': ocr_len,
            'summary_length': summary_len,
            'summary_compression_ratio': summary_len / (transcript_len + ocr_len) if (transcript_len + ocr_len) > 0 else 0,
            'tasks_identified': tasks_count,
        }
        # Remove ocr_text_length if it's 0
        if ocr_len == 0:
             del insights['content_analysis']['ocr_text_length']


        insights['quality_assessment_summary'] = {
            'overall_score': round(overall_score, 2),
            'meets_overall_threshold': meets_threshold,
            'transcript_score': round(results.get('quality_scores', {}).get('transcript_quality', {}).get('quality_score', 0.0), 2),
            'summary_score': round(results.get('quality_scores', {}).get('summary_quality', {}).get('quality_score', 0.0), 2),
            'task_score': round(results.get('quality_scores', {}).get('task_quality', {}).get('quality_score', 0.0), 2),
        }

        # Add recommendations based on quality insights if available
        quality_insights = results.get('quality_scores', {}).get('insights', {})
        insights['recommendations'].extend(quality_insights.get('recommendations', []))

        # Add simple recommendations based on score
        if not meets_threshold and overall_score > 0: # Check > 0 to avoid recommending on failed runs
            insights['recommendations'].append(
                f"Overall quality score ({overall_score:.2f}) is below the target threshold. Review component scores for areas needing improvement."
            )
        elif tasks_count == 0 and results.get('processing_info', {}).get('content_type_detected') == 'meeting':
             insights['recommendations'].append(
                 "No action items identified for a meeting. Verify if this is expected or if task extraction needs adjustment."
             )


        # Remove duplicates from recommendations
        insights['recommendations'] = sorted(list(set(insights['recommendations'])))

        return insights

    async def _update_processing_metrics(self, processing_time: float, success: bool):
        """Update agent performance metrics thread-safely (important for async)."""
        # In a real async scenario, use locks if metrics are shared across tasks
        # For this single-task agent, direct update is okay
        self.metrics['total_processed'] += 1
        self.metrics['total_processing_time'] += processing_time

        if success:
            self.metrics['successful_runs'] += 1
        else:
            self.metrics['failed_runs'] += 1

        total = self.metrics['total_processed']
        successful = self.metrics['successful_runs']

        # Update averages safely
        self.metrics['average_processing_time'] = self.metrics['total_processing_time'] / total if total > 0 else 0.0
        self.metrics['success_rate'] = successful / total if total > 0 else 1.0


    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        # Return a copy to prevent external modification
        status = {
            'metrics': self.metrics.copy(),
            # 'current_context': self.current_context.copy(), # Maybe too verbose, depends on need
            'processing_history_length': len(self.processing_history), # Assuming history is appended elsewhere
            'agent_ready': True # Simple status flag
        }
        # Format metrics for display
        status['metrics']['average_processing_time'] = round(status['metrics']['average_processing_time'], 2)
        status['metrics']['success_rate'] = round(status['metrics']['success_rate'] * 100, 1) # As percentage
        return status
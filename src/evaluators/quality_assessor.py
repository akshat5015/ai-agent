"""
Quality assessment module for evaluating the outputs of the AI agent.

This module implements various evaluation metrics to assess the quality of
transcripts, summaries, and extracted tasks, providing comprehensive quality scores.
"""

import logging
import asyncio
from typing import Dict, List, Any, Tuple
import time
import json
import re

import numpy as np
from rouge_score import rouge_scorer
from textstat import flesch_reading_ease, flesch_kincaid_grade
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


class QualityAssessor:
    """
    Comprehensive quality assessment system for AI agent outputs.
    
    Evaluates transcript accuracy, summary quality, task relevance,
    and overall system performance using multiple metrics.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the quality assessor.
        
        Args:
            config: Configuration dictionary with evaluation parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize evaluation components
        self._initialize_evaluators()
        
        # Quality thresholds
        self.thresholds = {
            'transcript_confidence': config.get('transcript_threshold', 0.8),
            'summary_coherence': config.get('summary_threshold', 0.7),
            'task_relevance': config.get('task_threshold', 0.75),
            'overall_quality': config.get('overall_threshold', 0.7)
        }
        
        # Evaluation weights
        self.weights = {
            'transcript': config.get('transcript_weight', 0.3),
            'summary': config.get('summary_weight', 0.4),
            'tasks': config.get('task_weight', 0.3)
        }
    
    def _initialize_evaluators(self):
        """Initialize evaluation tools and models."""
        try:
            # Initialize ROUGE scorer for summary evaluation
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
            
            # Download NLTK data if needed
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            self.stopwords = set(stopwords.words('english'))
            
            self.logger.info("Quality evaluators initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize evaluators: {e}")
            raise
    
    async def assess_outputs(
        self,
        audio_results: Dict,
        summary_results: Dict,
        task_results: Dict
    ) -> Dict[str, Any]:
        """
        Assess the quality of all AI agent outputs.
        
        Args:
            audio_results: Results from audio processing
            summary_results: Results from summarization
            task_results: Results from task extraction
            
        Returns:
            Dictionary containing quality scores and analysis
        """
        self.logger.info("Starting comprehensive quality assessment...")
        start_time = time.time()
        
        # Assess individual components
        transcript_quality = await self._assess_transcript_quality(audio_results)
        summary_quality = await self._assess_summary_quality(
            summary_results, audio_results.get('transcript', '')
        )
        task_quality = await self._assess_task_quality(
            task_results, audio_results.get('transcript', ''), summary_results.get('summary', '')
        )
        
        # Calculate overall quality
        overall_quality = self._calculate_overall_quality(
            transcript_quality, summary_quality, task_quality
        )
        
        # Generate insights and recommendations
        insights = self._generate_quality_insights(
            transcript_quality, summary_quality, task_quality, overall_quality
        )
        
        processing_time = time.time() - start_time
        
        return {
            'transcript_quality': transcript_quality,
            'summary_quality': summary_quality,
            'task_quality': task_quality,
            'overall_quality': overall_quality,
            'insights': insights,
            'metadata': {
                'assessment_time': processing_time,
                'thresholds': self.thresholds,
                'weights': self.weights,
                'evaluation_methods': ['confidence_analysis', 'rouge_scores', 'readability', 'coherence']
            }
        }
    
    async def _assess_transcript_quality(self, audio_results: Dict) -> Dict[str, Any]:
        """Assess the quality of audio transcription."""
        transcript = audio_results.get('transcript', '')
        confidence = audio_results.get('confidence', 0.0)
        metadata = audio_results.get('metadata', {})
        
        # Basic metrics
        word_count = len(transcript.split())
        sentence_count = len(sent_tokenize(transcript))
        
        # Language quality metrics
        readability = self._assess_readability(transcript)
        coherence = self._assess_coherence(transcript)
        
        # Technical quality metrics
        processing_quality = self._assess_processing_quality(metadata)
        
        # Aggregate score
        quality_score = self._calculate_transcript_score(
            confidence, readability, coherence, processing_quality
        )
        
        return {
            'confidence_score': confidence,
            'readability_metrics': readability,
            'coherence_score': coherence,
            'processing_metrics': processing_quality,
            'quality_score': quality_score,
            'meets_threshold': quality_score >= self.thresholds['transcript_confidence'],
            'statistics': {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_words_per_sentence': word_count / max(sentence_count, 1)
            }
        }
    
    async def _assess_summary_quality(self, summary_results: Dict, original_transcript: str) -> Dict[str, Any]:
        """Assess the quality of generated summary."""
        summary = summary_results.get('summary', '')
        metadata = summary_results.get('metadata', {})
        
        if not summary or not original_transcript:
            return self._empty_quality_result()
        
        # Content quality metrics
        rouge_scores = self._calculate_rouge_scores(summary, original_transcript)
        
        # Structural quality metrics
        readability = self._assess_readability(summary)
        coherence = self._assess_coherence(summary)
        compression_ratio = len(summary) / len(original_transcript)
        
        # Information preservation
        key_info_preservation = self._assess_information_preservation(summary, original_transcript)
        
        # Factual consistency (basic check)
        consistency_score = self._assess_factual_consistency(summary, original_transcript)
        
        # Aggregate quality score
        quality_score = self._calculate_summary_score(
            rouge_scores, readability, coherence, key_info_preservation, consistency_score
        )
        
        return {
            'rouge_scores': rouge_scores,
            'readability_metrics': readability,
            'coherence_score': coherence,
            'compression_ratio': compression_ratio,
            'information_preservation': key_info_preservation,
            'factual_consistency': consistency_score,
            'quality_score': quality_score,
            'meets_threshold': quality_score >= self.thresholds['summary_coherence'],
            'generation_metadata': metadata
        }
    
    async def _assess_task_quality(
        self, 
        task_results: Dict, 
        transcript: str, 
        summary: str
    ) -> Dict[str, Any]:
        """Assess the quality of extracted tasks."""
        tasks = task_results.get('tasks', [])
        metadata = task_results.get('metadata', {})
        
        if not tasks:
            return self._empty_task_quality_result()
        
        # Individual task quality
        task_scores = []
        for task in tasks:
            task_score = self._assess_individual_task(task, transcript, summary)
            task_scores.append(task_score)
        
        # Aggregate metrics
        avg_relevance = np.mean([score['relevance'] for score in task_scores])
        avg_clarity = np.mean([score['clarity'] for score in task_scores])
        avg_actionability = np.mean([score['actionability'] for score in task_scores])
        
        # Coverage metrics
        coverage_score = self._assess_task_coverage(tasks, transcript, summary)
        
        # Diversity metrics
        diversity_score = self._assess_task_diversity(tasks)
        
        # Overall task quality
        quality_score = self._calculate_task_quality_score(
            avg_relevance, avg_clarity, avg_actionability, coverage_score, diversity_score
        )
        
        return {
            'task_count': len(tasks),
            'individual_scores': task_scores,
            'average_relevance': avg_relevance,
            'average_clarity': avg_clarity,
            'average_actionability': avg_actionability,
            'coverage_score': coverage_score,
            'diversity_score': diversity_score,
            'quality_score': quality_score,
            'meets_threshold': quality_score >= self.thresholds['task_relevance'],
            'extraction_metadata': metadata
        }
    
    def _assess_readability(self, text: str) -> Dict[str, float]:
        """Assess text readability using multiple metrics."""
        if not text:
            return {'flesch_ease': 0, 'flesch_kincaid': 0, 'normalized_score': 0}
        
        try:
            flesch_ease = flesch_reading_ease(text)
            flesch_kincaid = flesch_kincaid_grade(text)
            
            # Normalize scores to 0-1 range
            # Flesch ease: 90-100 is very easy, 0-30 is very difficult
            normalized_ease = max(0, min(100, flesch_ease)) / 100
            
            # Flesch-Kincaid: Lower grades are easier
            normalized_kincaid = max(0, 1 - (min(20, flesch_kincaid) / 20))
            
            normalized_score = (normalized_ease + normalized_kincaid) / 2
            
            return {
                'flesch_ease': flesch_ease,
                'flesch_kincaid': flesch_kincaid,
                'normalized_score': normalized_score
            }
            
        except Exception as e:
            self.logger.warning(f"Readability assessment failed: {e}")
            return {'flesch_ease': 50, 'flesch_kincaid': 10, 'normalized_score': 0.5}
    
    def _assess_coherence(self, text: str) -> float:
        """Assess text coherence using linguistic features."""
        if not text:
            return 0.0
        
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0.8  # Single sentence is coherent by definition
        
        coherence_score = 0.0
        
        # Check for transition words and phrases
        transition_words = [
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'meanwhile', 'nevertheless', 'thus', 'hence',
            'first', 'second', 'finally', 'next', 'then', 'also'
        ]
        
        transition_count = sum(1 for word in transition_words if word in text.lower())
        transition_ratio = min(1.0, transition_count / len(sentences))
        coherence_score += transition_ratio * 0.3
        
        # Check for consistent topic (simple keyword overlap)
        sentence_words = [set(word_tokenize(sent.lower())) - self.stopwords 
                         for sent in sentences]
        
        overlaps = []
        for i in range(len(sentence_words) - 1):
            overlap = len(sentence_words[i].intersection(sentence_words[i + 1]))
            total_words = len(sentence_words[i].union(sentence_words[i + 1]))
            overlap_ratio = overlap / max(total_words, 1)
            overlaps.append(overlap_ratio)
        
        avg_overlap = np.mean(overlaps) if overlaps else 0
        coherence_score += min(1.0, avg_overlap * 2) * 0.4
        
        # Check for repetition (negative indicator)
        word_counts = {}
        words = word_tokenize(text.lower())
        for word in words:
            if word not in self.stopwords and word.isalpha():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        repetition_penalty = 0
        for count in word_counts.values():
            if count > 3:  # Word appears more than 3 times
                repetition_penalty += 0.05
        
        coherence_score = max(0.0, coherence_score + 0.3 - repetition_penalty)
        
        return min(1.0, coherence_score)
    
    def _assess_processing_quality(self, metadata: Dict) -> Dict[str, float]:
        """Assess technical processing quality."""
        processing_time = metadata.get('processing_time', 0)
        chunks_processed = metadata.get('chunks_processed', 0)
        
        # Time efficiency score (faster is better, but not too fast)
        time_score = 1.0 if processing_time == 0 else min(1.0, 60 / max(processing_time, 1))
        
        # Coverage score (more chunks suggests thorough processing)
        coverage_score = min(1.0, chunks_processed / 10) if chunks_processed > 0 else 0.5
        
        return {
            'time_efficiency': time_score,
            'processing_coverage': coverage_score,
            'overall_processing': (time_score + coverage_score) / 2
        }
    
    def _calculate_rouge_scores(self, summary: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores for summary evaluation."""
        try:
            scores = self.rouge_scorer.score(reference, summary)
            
            return {
                'rouge1_f': scores['rouge1'].fmeasure,
                'rouge2_f': scores['rouge2'].fmeasure,
                'rougeL_f': scores['rougeL'].fmeasure,
                'average_rouge': (scores['rouge1'].fmeasure + 
                                scores['rouge2'].fmeasure + 
                                scores['rougeL'].fmeasure) / 3
            }
            
        except Exception as e:
            self.logger.warning(f"ROUGE calculation failed: {e}")
            return {'rouge1_f': 0, 'rouge2_f': 0, 'rougeL_f': 0, 'average_rouge': 0}
    
    def _assess_information_preservation(self, summary: str, original: str) -> float:
        """Assess how well the summary preserves key information."""
        if not summary or not original:
            return 0.0
        
        # Extract key entities and concepts
        original_entities = self._extract_key_entities(original)
        summary_entities = self._extract_key_entities(summary)
        
        if not original_entities:
            return 0.5  # No entities to preserve
        
        # Calculate preservation ratio
        preserved_entities = summary_entities.intersection(original_entities)
        preservation_ratio = len(preserved_entities) / len(original_entities)
        
        return min(1.0, preservation_ratio)
    
    def _extract_key_entities(self, text: str) -> set:
        """Extract key entities from text (simple approach)."""
        # Extract capitalized words and numbers (potential entities)
        words = word_tokenize(text)
        entities = set()
        
        for word in words:
            # Capitalized words (potential proper nouns)
            if word[0].isupper() and len(word) > 2 and word not in self.stopwords:
                entities.add(word.lower())
            
            # Numbers and dates
            if re.match(r'\d+', word):
                entities.add(word)
        
        return entities
    
    def _assess_factual_consistency(self, summary: str, original: str) -> float:
        """Basic factual consistency check between summary and original."""
        if not summary or not original:
            return 0.0
        
        # Simple approach: check if summary statements can be supported by original
        summary_sentences = sent_tokenize(summary)
        original_words = set(word_tokenize(original.lower()))
        
        consistent_sentences = 0
        
        for sentence in summary_sentences:
            sentence_words = set(word_tokenize(sentence.lower()))
            # Remove stop words for better matching
            content_words = sentence_words - self.stopwords
            
            # Check if most content words appear in original
            if content_words:
                overlap_ratio = len(content_words.intersection(original_words)) / len(content_words)
                if overlap_ratio > 0.6:  # At least 60% of words found in original
                    consistent_sentences += 1
        
        return consistent_sentences / max(len(summary_sentences), 1)
    
    def _assess_individual_task(self, task: Dict, transcript: str, summary: str) -> Dict[str, float]:
        """Assess the quality of an individual task."""
        description = task.get('description', '')
        
        # Relevance: how relevant is the task to the content
        relevance = self._calculate_task_relevance(task, transcript, summary)
        
        # Clarity: how clear and understandable is the task
        clarity = self._calculate_task_clarity(description)
        
        # Actionability: how actionable is the task
        actionability = self._calculate_task_actionability(description)
        
        return {
            'relevance': relevance,
            'clarity': clarity,
            'actionability': actionability,
            'overall': (relevance + clarity + actionability) / 3
        }
    
    def _calculate_task_relevance(self, task: Dict, transcript: str, summary: str) -> float:
        """Calculate task relevance to the source content."""
        description = task.get('description', '').lower()
        context = task.get('context', '').lower()
        
        # Check if task keywords appear in source content
        task_words = set(word_tokenize(description)) - self.stopwords
        
        transcript_words = set(word_tokenize(transcript.lower())) - self.stopwords
        summary_words = set(word_tokenize(summary.lower())) - self.stopwords
        
        # Calculate overlap with source content
        transcript_overlap = len(task_words.intersection(transcript_words))
        summary_overlap = len(task_words.intersection(summary_words))
        
        if not task_words:
            return 0.0
        
        relevance_score = (transcript_overlap + summary_overlap * 2) / (len(task_words) * 2)
        
        # Boost score if task has high confidence
        confidence_boost = task.get('confidence', 0.5) * 0.3
        
        return min(1.0, relevance_score + confidence_boost)
    
    def _calculate_task_clarity(self, description: str) -> float:
        """Calculate how clear and well-formed a task description is."""
        if not description:
            return 0.0
        
        clarity_score = 0.5  # Base score
        
        # Check length (not too short, not too long)
        word_count = len(description.split())
        if 3 <= word_count <= 15:
            clarity_score += 0.2
        elif word_count < 3:
            clarity_score -= 0.3
        elif word_count > 20:
            clarity_score -= 0.1
        
        # Check for action verbs
        action_verbs = ['do', 'make', 'create', 'write', 'read', 'study', 'complete', 'finish', 'review']
        if any(verb in description.lower() for verb in action_verbs):
            clarity_score += 0.2
        
        # Check for specific details
        if any(char.isdigit() for char in description):  # Contains numbers
            clarity_score += 0.1
        
        return min(1.0, max(0.0, clarity_score))
    
    def _calculate_task_actionability(self, description: str) -> float:
        """Calculate how actionable a task description is."""
        if not description:
            return 0.0
        
        actionability_score = 0.3  # Base score
        
        description_lower = description.lower()
        
        # Check for imperative mood (starts with verb)
        words = description.split()
        if words and words[0].lower() in ['do', 'make', 'create', 'write', 'read', 'study', 'complete']:
            actionability_score += 0.3
        
        # Check for specific objects or targets
        specific_indicators = ['the', 'this', 'that', 'chapter', 'page', 'section', 'assignment']
        if any(indicator in description_lower for indicator in specific_indicators):
            actionability_score += 0.2
        
        # Check for time indicators
        time_indicators = ['by', 'before', 'until', 'deadline', 'tomorrow', 'next']
        if any(indicator in description_lower for indicator in time_indicators):
            actionability_score += 0.2
        
        return min(1.0, actionability_score)
    
    def _assess_task_coverage(self, tasks: List[Dict], transcript: str, summary: str) -> float:
        """Assess how well tasks cover the actionable content."""
        if not tasks:
            return 0.0
        
        # Simple heuristic: more tasks from longer content suggests better coverage
        task_count = len(tasks)
        content_length = len(transcript) + len(summary)
        
        # Expected tasks per 1000 characters of content
        expected_ratio = 0.5  # 0.5 tasks per 1000 characters
        expected_tasks = (content_length / 1000) * expected_ratio
        
        coverage_ratio = min(1.0, task_count / max(expected_tasks, 1))
        
        return coverage_ratio
    
    def _assess_task_diversity(self, tasks: List[Dict]) -> float:
        """Assess the diversity of extracted tasks."""
        if not tasks:
            return 0.0
        
        # Check category diversity
        categories = set(task.get('category', 'general') for task in tasks)
        category_diversity = len(categories) / len(tasks) if tasks else 0
        
        # Check description diversity (avoid too similar tasks)
        descriptions = [task.get('description', '') for task in tasks]
        unique_starts = set(desc[:10].lower() for desc in descriptions if len(desc) >= 10)
        start_diversity = len(unique_starts) / len(descriptions) if descriptions else 0
        
        return (category_diversity + start_diversity) / 2
    
    def _calculate_transcript_score(
        self, 
        confidence: float, 
        readability: Dict, 
        coherence: float, 
        processing: Dict
    ) -> float:
        """Calculate overall transcript quality score."""
        scores = [
            confidence * 0.4,
            readability['normalized_score'] * 0.3,
            coherence * 0.2,
            processing['overall_processing'] * 0.1
        ]
        
        return sum(scores)
    
    def _calculate_summary_score(
        self, 
        rouge_scores: Dict, 
        readability: Dict, 
        coherence: float, 
        preservation: float, 
        consistency: float
    ) -> float:
        """Calculate overall summary quality score."""
        scores = [
            rouge_scores['average_rouge'] * 0.3,
            readability['normalized_score'] * 0.2,
            coherence * 0.2,
            preservation * 0.15,
            consistency * 0.15
        ]
        
        return sum(scores)
    
    def _calculate_task_quality_score(
        self, 
        relevance: float, 
        clarity: float, 
        actionability: float, 
        coverage: float, 
        diversity: float
    ) -> float:
        """Calculate overall task quality score."""
        scores = [
            relevance * 0.3,
            clarity * 0.25,
            actionability * 0.25,
            coverage * 0.1,
            diversity * 0.1
        ]
        
        return sum(scores)
    
    def _calculate_overall_quality(
        self, 
        transcript_quality: Dict, 
        summary_quality: Dict, 
        task_quality: Dict
    ) -> Dict[str, Any]:
        """Calculate overall system quality score."""
        # Extract component scores
        transcript_score = transcript_quality.get('quality_score', 0)
        summary_score = summary_quality.get('quality_score', 0)
        task_score = task_quality.get('quality_score', 0)
        
        # Apply weights
        weighted_score = (
            transcript_score * self.weights['transcript'] +
            summary_score * self.weights['summary'] +
            task_score * self.weights['tasks']
        )
        
        # Check if all components meet thresholds
        all_thresholds_met = (
            transcript_quality.get('meets_threshold', False) and
            summary_quality.get('meets_threshold', False) and
            task_quality.get('meets_threshold', False)
        )
        
        return {
            'score': weighted_score,
            'meets_threshold': weighted_score >= self.thresholds['overall_quality'],
            'all_components_pass': all_thresholds_met,
            'component_scores': {
                'transcript': transcript_score,
                'summary': summary_score,
                'tasks': task_score
            },
            'weighted_components': {
                'transcript': transcript_score * self.weights['transcript'],
                'summary': summary_score * self.weights['summary'],
                'tasks': task_score * self.weights['tasks']
            }
        }
    
    def _generate_quality_insights(
        self, 
        transcript_quality: Dict,
        summary_quality: Dict, 
        task_quality: Dict, 
        overall_quality: Dict
    ) -> Dict[str, Any]:
        """Generate insights and recommendations based on quality assessment."""
        insights = {
            'strengths': [],
            'weaknesses': [],
            'recommendations': [],
            'quality_summary': {}
        }
        
        # Analyze transcript quality
        if transcript_quality['quality_score'] > 0.8:
            insights['strengths'].append("High-quality audio transcription with good confidence")
        elif transcript_quality['quality_score'] < 0.6:
            insights['weaknesses'].append("Low transcription quality may affect downstream processing")
            insights['recommendations'].append("Consider using higher quality audio or different transcription settings")
        
        # Analyze summary quality
        if summary_quality.get('rouge_scores', {}).get('average_rouge', 0) > 0.4:
            insights['strengths'].append("Summary effectively captures key content from original")
        elif summary_quality.get('rouge_scores', {}).get('average_rouge', 0) < 0.2:
            insights['weaknesses'].append("Summary may miss important information from original content")
            insights['recommendations'].append("Consider adjusting summary length or generation parameters")
        
        # Analyze task quality
        task_count = task_quality.get('task_count', 0)
        if task_count > 0:
            if task_quality['average_relevance'] > 0.7:
                insights['strengths'].append(f"Successfully identified {task_count} relevant tasks")
            else:
                insights['weaknesses'].append("Some extracted tasks may not be highly relevant")
                insights['recommendations'].append("Review task extraction patterns for better relevance")
        else:
            insights['weaknesses'].append("No actionable tasks identified")
            insights['recommendations'].append("Content may be informational only, or extraction rules need adjustment")
        
        # Overall assessment
        overall_score = overall_quality['score']
        if overall_score > 0.8:
            insights['quality_summary']['level'] = 'excellent'
            insights['quality_summary']['description'] = 'High-quality processing across all components'
        elif overall_score > 0.7:
            insights['quality_summary']['level'] = 'good'
            insights['quality_summary']['description'] = 'Good quality with minor areas for improvement'
        elif overall_score > 0.5:
            insights['quality_summary']['level'] = 'acceptable'
            insights['quality_summary']['description'] = 'Acceptable quality but significant room for improvement'
        else:
            insights['quality_summary']['level'] = 'poor'
            insights['quality_summary']['description'] = 'Poor quality results, review inputs and settings'
        
        return insights
    
    def _empty_quality_result(self) -> Dict[str, Any]:
        """Return empty quality result for error cases."""
        return {
            'quality_score': 0.0,
            'meets_threshold': False,
            'error': 'No content to evaluate'
        }
    
    def _empty_task_quality_result(self) -> Dict[str, Any]:
        """Return empty task quality result."""
        return {
            'task_count': 0,
            'quality_score': 0.0,
            'meets_threshold': False,
            'message': 'No tasks extracted'
        }

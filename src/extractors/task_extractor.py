"""
Task extraction module for identifying actionable items from transcripts and summaries.

This module uses NLP techniques and pattern matching to identify and categorize
actionable tasks, deadlines, and important follow-up items from processed content.
"""

import logging
import re
import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import asyncio

import spacy
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk


class TaskExtractor:
    """
    Intelligent task extraction system that identifies actionable items
    from audio transcripts and summaries using multiple NLP approaches.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the task extractor.
        
        Args:
            config: Configuration dictionary with extraction parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP models
        self._initialize_nlp_models()
        
        # Task patterns for different content types
        self.task_patterns = self._initialize_task_patterns()
        
        # Priority and deadline detection patterns
        self.priority_patterns = self._initialize_priority_patterns()
        self.deadline_patterns = self._initialize_deadline_patterns()
        
        # Content type specific extraction rules
        self.content_type_rules = {
            'academic': self._get_academic_rules(),
            'meeting': self._get_meeting_rules(),
            'study_session': self._get_study_rules(),
            'general': self._get_general_rules()
        }
    
    def _initialize_nlp_models(self):
        """Initialize required NLP models and tools."""
        try:
            # Load spaCy model for advanced NLP
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize NLTK components
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download('averaged_perceptron_tagger', quiet=True)
            
            try:
                nltk.data.find('chunkers/maxent_ne_chunker')
            except LookupError:
                nltk.download('maxent_ne_chunker', quiet=True)
                nltk.download('words', quiet=True)
            
            # Initialize transformer pipeline for classification (optional)
            # Wrap in try-except to gracefully handle initialization failures
            try:
                # --- THIS IS THE FIX ---
                # Replaced generative model with a proper classification model
                self.classifier = pipeline(
                    "text-classification",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    return_all_scores=True,
                    device=-1  # Force CPU
                )
                # ------------------------
                self.logger.info("Transformer classifier initialized successfully")
            except Exception as classifier_error:
                self.logger.warning(f"Failed to initialize transformer classifier: {classifier_error}. Continuing without it.")
                self.classifier = None
            
            self.logger.info("NLP models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP models: {e}")
            # Use fallback methods if models fail to load
            self.nlp = None
            self.classifier = None
    
    def _initialize_task_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for task detection."""
        return {
            'assignment_verbs': [
                r'\b(?:need|needs|have|has|should|must|will|going)\s+to\s+(\w+(?:\s+\w+){0,5})',
                r'\b(?:assigned|tasked|responsible)\s+(?:to|with|for)\s+(\w+(?:\s+\w+){0,5})',
                r'\b(?:action|todo|task)(?:\s*:|\s+item)?\s*[-:]?\s*(\w+(?:\s+\w+){0,5})',
            ],
            'deadline_indicators': [
                r'\b(?:by|before|until|deadline|due)\s+(\w+(?:\s+\w+){0,3})',
                r'\b(?:next|this)\s+(\w+(?:\s+\w+){0,2})',
                r'\b(\w+day)\s*,?\s*(\w+(?:\s+\w+){0,2})',
            ],
            'responsibility_indicators': [
                r'\b(\w+)\s+(?:will|should|needs to|has to)\s+(\w+(?:\s+\w+){0,5})',
                r'\b(\w+)(?:\s+is)?\s+responsible\s+for\s+(\w+(?:\s+\w+){0,5})',
                r'\b(\w+)\s+takes?\s+(?:care\s+of|charge\s+of)\s+(\w+(?:\s+\w+){0,5})',
            ],
            'study_tasks': [
                r'\b(?:study|review|read|practice|solve|complete)\s+(\w+(?:\s+\w+){0,5})',
                r'\b(?:homework|assignment|problem|exercise)\s*:?\s*(\w+(?:\s+\w+){0,5})',
                r'\b(?:prepare|research)\s+(?:for|about)\s+(\w+(?:\s+\w+){0,5})',
            ]
        }
    
    def _initialize_priority_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for priority detection."""
        return {
            'high': [
                r'\b(?:urgent|critical|important|asap|immediately|priority|crucial)\b',
                r'\b(?:high\s+priority|top\s+priority|must\s+do)\b',
                r'\b(?:emergency|deadline\s+tomorrow|due\s+today)\b'
            ],
            'medium': [
                r'\b(?:should|need\s+to|ought\s+to|important\s+to)\b',
                r'\b(?:next\s+week|soon|moderate\s+priority)\b'
            ],
            'low': [
                r'\b(?:when\s+possible|if\s+time|eventually|sometime)\b',
                r'\b(?:low\s+priority|nice\s+to\s+have|optional)\b'
            ]
        }
    
    def _initialize_deadline_patterns(self) -> List[str]:
        """Initialize patterns for deadline extraction."""
        return [
            r'\b(?:by|before|until|deadline)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(?:by|before|until|deadline)\s+(\d{1,2}[\/\-]\d{1,2}(?:[\/\-]\d{2,4})?)\b',
            r'\b(?:by|before|until|deadline)\s+(tomorrow|today|next\s+week|this\s+week|next\s+month|end\s+of\s+week|eow|end\s+of\s+day|eod|close\s+of\s+business|cob)\b',
            r'\b(?:by|before|until)\s+(?:\d{1,2})(?::\d{2})?\s*(am|pm)?\b',
            r'\b(?:by|before|until)\s+(?:\d{1,2})(?::\d{2})?\s*(am|pm)?\s+on\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{2,4}))?\b',
            r'\b(?:due|deadline)(?:\s+is)?\s+(.*?)(?:\.|,|\n|$)'
        ]
    
    def _get_academic_rules(self) -> Dict:
        """Get extraction rules specific to academic content."""
        return {
            'task_keywords': ['assignment', 'homework', 'project', 'essay', 'exam', 'study', 'read', 'research'],
            'deadline_keywords': ['due', 'deadline', 'test', 'exam', 'presentation'],
            'priority_boost': ['exam', 'test', 'final', 'midterm', 'grade'],
            'context_window': 2  # sentences around task keywords
        }
    
    def _get_meeting_rules(self) -> Dict:
        """Get extraction rules specific to meeting content."""
        return {
            'task_keywords': ['action', 'follow up', 'next steps', 'assigned', 'responsible', 'deliver'],
            'deadline_keywords': ['by', 'before', 'next meeting', 'end of week'],
            'priority_boost': ['urgent', 'asap', 'immediately', 'critical'],
            'context_window': 1
        }
    
    def _get_study_rules(self) -> Dict:
        """Get extraction rules specific to study sessions."""
        return {
            'task_keywords': ['review', 'practice', 'memorize', 'solve', 'understand', 'focus on'],
            'deadline_keywords': ['test', 'quiz', 'exam', 'next class'],
            'priority_boost': ['weak area', 'confused about', 'don\'t understand'],
            'context_window': 1
        }
    
    def _get_general_rules(self) -> Dict:
        """Get general extraction rules."""
        return {
            'task_keywords': ['do', 'complete', 'finish', 'work on', 'handle', 'take care of'],
            'deadline_keywords': ['by', 'before', 'deadline', 'due'],
            'priority_boost': ['important', 'urgent', 'priority'],
            'context_window': 1
        }
    
    async def extract_tasks(
        self, 
        transcript: str, 
        summary: str, 
        config: Dict
    ) -> Dict[str, Any]:
        """
        Extract actionable tasks from transcript and summary.
        
        Args:
            transcript: Full transcript text
            summary: Generated summary
            config: Extraction configuration
            
        Returns:
            Dictionary containing extracted tasks and metadata
        """
        self.logger.info("Starting task extraction...")
        start_time = datetime.now()
        
        # Get content type and focus
        content_type = config.get('content_type', 'general')
        focus = config.get('focus', 'general_todos')
        
        # Extract tasks from both transcript and summary
        transcript_tasks = await self._extract_from_text(transcript, content_type, 'transcript')
        summary_tasks = await self._extract_from_text(summary, content_type, 'summary')
        
        # Combine and deduplicate tasks
        all_tasks = transcript_tasks + summary_tasks
        deduplicated_tasks = self._deduplicate_tasks(all_tasks)
        
        # Enhance tasks with additional information
        enhanced_tasks = await self._enhance_tasks(deduplicated_tasks, transcript, summary)
        
        # Filter and prioritize based on focus
        filtered_tasks = self._filter_by_focus(enhanced_tasks, focus)
        
        # Sort by priority and relevance
        sorted_tasks = self._sort_tasks(filtered_tasks)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'tasks': sorted_tasks,
            'metadata': {
                'total_tasks_found': len(all_tasks),
                'after_deduplication': len(deduplicated_tasks),
                'final_tasks': len(sorted_tasks),
                'content_type': content_type,
                'focus': focus,
                'processing_time': processing_time,
                'extraction_methods': ['pattern_matching', 'nlp_analysis', 'semantic_analysis']
            }
        }
        
        self.logger.info(f"Extracted {len(sorted_tasks)} tasks in {processing_time:.2f}s")
        return result
    
    async def _extract_from_text(
        self, 
        text: str, 
        content_type: str, 
        source: str
    ) -> List[Dict]:
        """Extract tasks from a single text using multiple methods."""
        tasks = []
        
        # Method 1: Pattern-based extraction
        pattern_tasks = self._extract_with_patterns(text, content_type)
        tasks.extend(pattern_tasks)
        
        # Method 2: NLP-based extraction using spaCy
        if self.nlp:
            nlp_tasks = self._extract_with_nlp(text, content_type)
            tasks.extend(nlp_tasks)
        
        # Method 3: Sentence-level analysis
        sentence_tasks = self._extract_with_sentence_analysis(text, content_type)
        tasks.extend(sentence_tasks)
        
        # Add source information
        for task in tasks:
            task['source'] = source
            task['extraction_method'] = task.get('method', 'unknown')
        
        return tasks
    
    def _extract_with_patterns(self, text: str, content_type: str) -> List[Dict]:
        """Extract tasks using regex patterns."""
        tasks = []
        rules = self.content_type_rules.get(content_type, self.content_type_rules['general'])
        
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Check for task keywords
            for keyword in rules['task_keywords']:
                if keyword in sentence_lower:
                    # Extract context around the keyword
                    context_start = max(0, i - rules['context_window'])
                    context_end = min(len(sentences), i + rules['context_window'] + 1)
                    context = ' '.join(sentences[context_start:context_end])
                    
                    # Extract task details
                    task = self._parse_task_from_sentence(sentence, keyword, context)
                    if task:
                        task['method'] = 'pattern_matching'
                        task['confidence'] = self._calculate_pattern_confidence(sentence, keyword)
                        tasks.append(task)
        
        return tasks
    
    def _extract_with_nlp(self, text: str, content_type: str) -> List[Dict]:
        """Extract tasks using spaCy NLP analysis."""
        tasks = []
        
        try:
            doc = self.nlp(text)
            
            # Analyze each sentence
            for sent in doc.sents:
                # Look for verb phrases that indicate actions
                action_phrases = self._find_action_phrases(sent)
                
                for phrase in action_phrases:
                    task = self._create_task_from_phrase(phrase, sent.text)
                    if task:
                        task['method'] = 'nlp_analysis'
                        task['confidence'] = self._calculate_nlp_confidence(phrase, sent)
                        tasks.append(task)
                        
        except Exception as e:
            self.logger.warning(f"NLP extraction failed: {e}")
        
        return tasks
    
    def _extract_with_sentence_analysis(self, text: str, content_type: str) -> List[Dict]:
        """Extract tasks using sentence-level semantic analysis."""
        tasks = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            # Check for imperative mood or future tense
            if self._is_action_sentence(sentence):
                task = self._extract_task_from_action_sentence(sentence)
                if task:
                    task['method'] = 'semantic_analysis'
                    task['confidence'] = self._calculate_semantic_confidence(sentence)
                    tasks.append(task)
        
        return tasks
    
    def _parse_task_from_sentence(
        self, 
        sentence: str, 
        keyword: str, 
        context: str
    ) -> Optional[Dict]:
        """Parse a task from a sentence containing a task keyword."""
        # Find the action after the keyword
        keyword_index = sentence.lower().find(keyword)
        if keyword_index == -1:
            return None
        
        # Extract the action part
        after_keyword = sentence[keyword_index + len(keyword):].strip()
        
        # Clean and extract the main action
        action = self._clean_task_description(after_keyword)
        
        if len(action) < 3 or len(action) > 200:  # Filter out too short or too long actions
            return None
        
        # Extract additional information
        deadline = self._extract_deadline(context)
        priority = self._extract_priority(context)
        assignee = self._extract_assignee(context)
        
        return {
            'description': action,
            'deadline': deadline,
            'priority': priority,
            'assignee': assignee,
            'context': sentence,
            'full_context': context
        }
    
    def _find_action_phrases(self, sent) -> List:
        """Find action phrases using spaCy dependency parsing."""
        action_phrases = []
        
        for token in sent:
            # Look for verbs that indicate actions
            if (token.pos_ == 'VERB' and 
                token.dep_ in ['ROOT', 'xcomp', 'ccomp'] and
                not token.lemma_ in ['be', 'have', 'do']):
                
                # Collect the verb phrase
                phrase_tokens = [token]
                
                # Add direct objects and modifiers
                for child in token.children:
                    if child.dep_ in ['dobj', 'prep', 'advmod', 'amod']:
                        phrase_tokens.extend([child] + list(child.subtree))
                
                phrase_text = ' '.join([t.text for t in sorted(phrase_tokens, key=lambda x: x.i)])
                action_phrases.append({
                    'text': phrase_text,
                    'verb': token.lemma_,
                    'tokens': phrase_tokens
                })
        
        return action_phrases
    
    def _create_task_from_phrase(self, phrase: Dict, sentence: str) -> Optional[Dict]:
        """Create a task dictionary from an action phrase."""
        description = phrase['text'].strip()
        
        if len(description) < 5 or len(description) > 150:
            return None
        
        # Extract additional information from sentence
        deadline = self._extract_deadline(sentence)
        priority = self._extract_priority(sentence)
        assignee = self._extract_assignee(sentence)
        
        return {
            'description': description,
            'deadline': deadline,
            'priority': priority,
            'assignee': assignee,
            'context': sentence,
            'verb': phrase['verb']
        }
    
    def _is_action_sentence(self, sentence: str) -> bool:
        """Check if a sentence contains actionable content."""
        # Check for imperative mood indicators
        imperative_patterns = [
            r'^\s*[A-Z][a-z]+',  # Starts with capitalized verb
            r'\b(?:need|must|should|have to|will|going to)\b',
            r'\b(?:please|let\'s|we should)\b'
        ]
        
        for pattern in imperative_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        
        # Check for future tense
        future_indicators = ['will', 'shall', 'going to', 'plan to', 'intend to']
        sentence_lower = sentence.lower()
        
        return any(indicator in sentence_lower for indicator in future_indicators)
    
    def _extract_task_from_action_sentence(self, sentence: str) -> Optional[Dict]:
        """Extract task information from an action-oriented sentence."""
        # Clean the sentence
        cleaned = self._clean_task_description(sentence)
        
        if len(cleaned) < 10 or len(cleaned) > 200:
            return None
        
        # Extract components
        deadline = self._extract_deadline(sentence)
        priority = self._extract_priority(sentence)
        assignee = self._extract_assignee(sentence)
        
        return {
            'description': cleaned,
            'deadline': deadline,
            'priority': priority,
            'assignee': assignee,
            'context': sentence
        }
    
    def _clean_task_description(self, text: str) -> str:
        """Clean and normalize task description."""
        # Remove common prefixes
        prefixes_to_remove = [
            r'^\s*(?:we need to|need to|should|must|have to|will|going to)\s*',
            r'^\s*(?:please|let\'s)\s*',
            r'^\s*(?:action item|todo|task):\s*',
            r'^\s*[-•]\s*'
        ]
        
        cleaned = text
        for prefix in prefixes_to_remove:
            cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE).strip()
        
        # Remove trailing punctuation and clean up
        cleaned = re.sub(r'[.!?]+$', '', cleaned).strip()
        
        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned
    
    def _extract_deadline(self, text: str) -> Optional[str]:
        """Extract deadline information from text."""
        text_lower = text.lower().strip()
        
        for pattern in self.deadline_patterns:
            match = re.search(pattern, text_lower)
            if match:
                # Return the most specific group, or the full match
                return next((g for g in match.groups() if g), match.group(0))
        
        return None
    
    def _extract_priority(self, text: str) -> str:
        """Extract priority level from text."""
        text_lower = text.lower()
        
        # Check for high priority indicators
        for pattern in self.priority_patterns['high']:
            if re.search(pattern, text_lower):
                return 'high'
        
        # Check for medium priority indicators
        for pattern in self.priority_patterns['medium']:
            if re.search(pattern, text_lower):
                return 'medium'
        
        # Check for low priority indicators
        for pattern in self.priority_patterns['low']:
            if re.search(pattern, text_lower):
                return 'low'
        
        return 'medium'  # Default priority
    
    def _extract_assignee(self, text: str) -> Optional[str]:
        """Extract assignee information from text."""
        # Look for name patterns after responsibility indicators
        responsibility_pattern = r'\b(\w+)\s+(?:will|should|needs? to|(?:is )?responsible for|assigned to)\b'
        match = re.search(responsibility_pattern, text, re.IGNORECASE)
        
        if match:
            assignee = match.group(1)
            # Filter out common non-names
            if assignee.lower() not in ['i', 'you', 'we', 'he', 'she', 'they', 'team', 'everyone']:
                 return assignee
        
        return None
    
    def _calculate_pattern_confidence(self, sentence: str, keyword: str) -> float:
        """Calculate confidence score for pattern-based extraction."""
        base_confidence = 0.6
        
        # Boost confidence based on context
        if any(word in sentence.lower() for word in ['must', 'need', 'should', 'deadline']):
            base_confidence += 0.2
        
        # Reduce confidence for questions
        if '?' in sentence:
            base_confidence -= 0.3
        
        return max(0.1, min(1.0, base_confidence))
    
    def _calculate_nlp_confidence(self, phrase: Dict, sent) -> float:
        """Calculate confidence score for NLP-based extraction."""
        base_confidence = 0.7
        
        # Boost confidence for strong action verbs
        strong_verbs = ['complete', 'finish', 'deliver', 'submit', 'create', 'develop']
        if phrase['verb'] in strong_verbs:
            base_confidence += 0.2
        
        # Consider sentence structure
        if sent.root.pos_ == 'VERB':
            base_confidence += 0.1
        
        return max(0.1, min(1.0, base_confidence))
    
    def _calculate_semantic_confidence(self, sentence: str) -> float:
        """Calculate confidence score for semantic analysis."""
        base_confidence = 0.5
        
        # Check for strong action indicators
        action_indicators = ['will', 'must', 'need to', 'should', 'have to']
        for indicator in action_indicators:
            if indicator in sentence.lower():
                base_confidence += 0.1
        
        return max(0.1, min(1.0, base_confidence))
    
    def _deduplicate_tasks(self, tasks: List[Dict]) -> List[Dict]:
        """Remove duplicate tasks based on similarity."""
        if not tasks:
            return []
        
        unique_tasks = []
        
        for task in tasks:
            is_duplicate = False
            
            for existing_task in unique_tasks:
                # Check similarity of descriptions
                similarity = self._calculate_similarity(
                    task['description'], 
                    existing_task['description']
                )
                
                if similarity > 0.8:  # High similarity threshold
                    # Merge tasks, keeping the one with higher confidence
                    if task.get('confidence', 0) > existing_task.get('confidence', 0):
                        # Replace existing with current
                        unique_tasks[unique_tasks.index(existing_task)] = task
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_tasks.append(task)
        
        return unique_tasks
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def _enhance_tasks(
        self, 
        tasks: List[Dict], 
        transcript: str, 
        summary: str
    ) -> List[Dict]:
        """Enhance tasks with additional context and information."""
        enhanced_tasks = []
        
        for task in tasks:
            enhanced_task = task.copy()
            
            # Add relevance score
            enhanced_task['relevance_score'] = self._calculate_relevance(task, transcript, summary)
            
            # Standardize deadline format
            if task.get('deadline'):
                parsed_deadline = self._parse_deadline(task['deadline'])
                if parsed_deadline:
                     # Store as standard string, JSON doesn't like datetime objects
                     enhanced_task['deadline_parsed'] = parsed_deadline.isoformat()

            
            # Add task category
            enhanced_task['category'] = self._categorize_task(task['description'])
            
            # Add estimated effort
            enhanced_task['estimated_effort'] = self._estimate_effort(task['description'])
            
            enhanced_tasks.append(enhanced_task)
        
        return enhanced_tasks
    
    def _calculate_relevance(self, task: Dict, transcript: str, summary: str) -> float:
        """Calculate task relevance score."""
        description = task['description'].lower()
        
        # Count occurrences in transcript and summary
        transcript_count = transcript.lower().count(description[:20])  # First 20 chars
        summary_count = summary.lower().count(description[:20])
        
        # Base relevance on confidence and occurrence
        base_relevance = task.get('confidence', 0.5)
        
        # Boost if mentioned multiple times
        if transcript_count > 1:
            base_relevance += 0.1
        if summary_count > 0:
            base_relevance += 0.2
        
        # Boost based on priority
        priority = task.get('priority', 'medium')
        priority_boost = {'high': 0.3, 'medium': 0.1, 'low': 0.0}
        base_relevance += priority_boost[priority]
        
        return max(0.0, min(1.0, base_relevance))
    
    def _parse_deadline(self, deadline_text: str) -> Optional[datetime]:
        """Parse deadline text into a datetime object."""
        deadline_lower = deadline_text.lower().strip()
        
        try:
            now = datetime.now()
            # Handle relative dates and business terms
            if any(term in deadline_lower for term in ['eod', 'end of day']):
                return now.replace(hour=17, minute=0, second=0, microsecond=0)
            if any(term in deadline_lower for term in ['cob', 'close of business']):
                return now.replace(hour=17, minute=0, second=0, microsecond=0)
            if 'tomorrow' in deadline_lower:
                return now.replace(hour=17, minute=0, second=0, microsecond=0) + timedelta(days=1)
            elif 'today' in deadline_lower:
                return now.replace(hour=17, minute=0, second=0, microsecond=0)
            elif 'next week' in deadline_lower:
                # --- FIX 1 ---
                return (now + timedelta(weeks=1)).replace(hour=17, minute=0, second=0, microsecond=0)
            elif 'this week' in deadline_lower or 'end of week' in deadline_lower or 'eow' in deadline_lower:
                end_of_week = now + timedelta(days=(4 - now.weekday())) # End of Friday
                return end_of_week.replace(hour=17, minute=0, second=0, microsecond=0)
            
            # Handle specific dates (simple patterns)
            date_pattern = r'(\d{1,2})[\/\-](\d{1,2})(?:[\/\-](\d{2,4}))?'
            match = re.search(date_pattern, deadline_text)
            if match:
                month, day = int(match.group(1)), int(match.group(2))
                year = int(match.group(3)) if match.group(3) else datetime.now().year
                if year < 100:  # Handle 2-digit years
                    year += 2000
                return datetime(year, month, day, 17, 0) # Default to 5pm

            # Month name date like "12 October 2025"
            month_match = re.search(r'(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{2,4}))?', deadline_lower)
            if month_match:
                day = int(month_match.group(1))
                month_name = month_match.group(2)
                year = int(month_match.group(3)) if month_match.group(3) else datetime.now().year
                if year < 100:
                    year += 2000
                month_map = {'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,'july':7,'august':8,'september':9,'october':10,'november':11,'december':12}
                return datetime(year, month_map[month_name], day, 17, 0) # Default to 5pm

            # Time like "by 5pm" today
            time_match = re.search(r'(?:by|before|until)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', deadline_lower)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2)) if time_match.group(2) else 0
                ampm = time_match.group(3)
                if ampm == 'pm' and hour < 12:
                    hour += 12
                if ampm == 'am' and hour == 12:
                    hour = 0
                return now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
        except (ValueError, TypeError):
            self.logger.debug(f"Could not parse deadline: {deadline_text}")
        
        return None
    
    def _categorize_task(self, description: str) -> str:
        """Categorize task based on description."""
        description_lower = description.lower()
        
        categories = {
            'academic': ['study', 'read', 'homework', 'assignment', 'research', 'write', 'exam', 'paper', 'thesis'],
            'communication': ['email', 'call', 'message', 'contact', 'meeting', 'discuss', 'present', 'report'],
            'administrative': ['schedule', 'book', 'register', 'submit', 'file', 'form', 'organize', 'plan'],
            'development': ['code', 'develop', 'build', 'create', 'design', 'implement', 'fix', 'debug'],
            'review': ['review', 'check', 'verify', 'test', 'validate', 'evaluate', 'approve']
        }
        
        for category, keywords in categories.items():
            if any(keyword in description_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def _estimate_effort(self, description: str) -> str:
        """Estimate effort required for task."""
        description_lower = description.lower()
        word_count = len(description.split())
        
        # Simple heuristics based on description
        # --- FIX 2 ---
        high_effort_keywords = ['project', 'develop', 'create', 'design', 'research', 'write', 'implement', 'build']
        medium_effort_keywords = ['review', 'prepare', 'organize', 'plan', 'study', 'test', 'analyze']
        
        if any(keyword in description_lower for keyword in high_effort_keywords):
            return 'high'
        elif any(keyword in description_lower for keyword in medium_effort_keywords):
            return 'medium'
        elif word_count > 10: # Longer tasks often medium
            return 'medium'
        else:
            return 'low'
    
    def _filter_by_focus(self, tasks: List[Dict], focus: str) -> List[Dict]:
        """Filter tasks based on focus area."""
        if focus == 'general_todos':
            return tasks
        
        focus_filters = {
            'assignments_deadlines': lambda t: 'academic' in t['category'] or t.get('deadline'),
            'action_items': lambda t: t['category'] in ['communication', 'development', 'administrative'] or 'action' in t['description'].lower(),
            'study_tasks': lambda t: 'academic' in t['category'] or 'review' in t['category']
        }
        
        filter_func = focus_filters.get(focus)
        if filter_func:
            return [task for task in tasks if filter_func(task)]
        
        return tasks
    
    def _sort_tasks(self, tasks: List[Dict]) -> List[Dict]:
        """Sort tasks by priority and relevance."""
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        
        def sort_key(task):
            priority_score = priority_order.get(task.get('priority', 'medium'), 2)
            relevance_score = task.get('relevance_score', 0.5)
            confidence_score = task.get('confidence', 0.5)
            
            # Boost tasks with deadlines
            deadline_boost = 0.5 if task.get('deadline') else 0
            
            # Combine scores
            # Priority is the most important factor
            return (priority_score * 10) + relevance_score + confidence_score + deadline_boost
        
        return sorted(tasks, key=sort_key, reverse=True)
"""
Fine-tuned summarization model using LoRA (Low-Rank Adaptation) for domain-specific summarization.

This module implements a fine-tuned language model specifically trained for converting
audio transcripts into high-quality summaries with different styles based on content type.

This version includes a Map-Reduce strategy for handling long transcripts.
"""

import logging
import json
import torch
from typing import Dict, List, Optional, Any
from pathlib import Path
import time
import math
import copy  # For deepcopy

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import numpy as np


class FineTunedSummarizer:
    """
    Fine-tuned summarization model with LoRA adaptation for audio transcript summarization.

    This model is specifically fine-tuned for different content types (academic, meetings, etc.)
    and can adapt its summarization style accordingly.
    """

    def __init__(self, config: Dict):
        """
        Initialize the fine-tuned summarizer.

        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Model configuration
        self.base_model_name = config.get('base_model', 'facebook/bart-large-cnn')
        self.model_cache_dir = config.get('cache_dir', './models/cache')
        self.fine_tuned_model_path = config.get('fine_tuned_path', './models/fine_tuned')
        self.max_input_length = config.get('max_input_length', 2048)  # Increased for longer inputs

        # LoRA configuration
        self.lora_config = LoraConfig(
            r=config.get('lora_r', 16),  # Rank
            lora_alpha=config.get('lora_alpha', 32),  # Alpha parameter
            target_modules=config.get('lora_targets', ["q_proj", "v_proj", "k_proj", "out_proj"]),
            lora_dropout=config.get('lora_dropout', 0.1),
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )

        # Generation parameters
        self.generation_config = GenerationConfig(
            max_length=config.get('max_summary_length', 1024),
            min_length=config.get('min_summary_length', 200),
            length_penalty=config.get('length_penalty', 2.0),
            num_beams=config.get('num_beams', 4),
            early_stopping=True,
            no_repeat_ngram_size=3,
            do_sample=False,
            temperature=0.7,
            top_p=0.9
        )

        # Load or initialize models
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._is_fine_tuned = False  # Track if we're using fine-tuned model

        self._initialize_models()

        # Style templates for different content types
        self.style_templates = {
            'structured_academic': {
                'prefix': "Create a detailed summary of this academic lecture. Focus on key concepts, definitions, and learning objectives. The summary should be at least 8 sentences long: ",
                'format_instructions': "Structure the summary with: 1) Main topics, 2) Key concepts, 3) Important details"
            },
            'action_oriented': {
                'prefix': "Summarize this meeting focusing on decisions made and action items. Be detailed and list all actions clearly: ",
                'format_instructions': "Highlight: 1) Decisions made, 2) Action items, 3) Next steps"
            },
            'key_points': {
                'prefix': "Summarize this study session highlighting the most important points in a detailed manner: ",
                'format_instructions': "Focus on: 1) Key concepts studied, 2) Areas needing review, 3) Study recommendations"
            },
            'comprehensive': {
                'prefix': "Create a detailed and comprehensive summary of the following content, covering all major topics discussed. The summary should be a few paragraphs long: ",
                'format_instructions': "Provide a balanced overview covering all major topics discussed"
            },
            # --- NEW PROMPTS FOR MAP-REDUCE ---
            'chunk_summary': {
                'prefix': "Summarize this section of the transcript in detail, covering all key points, decisions, and action items. Be thorough and comprehensive: ",
                'format_instructions': "Extract all main ideas from this chunk in detail."
            },
            'final_summary': {
                'prefix': "The following text consists of several detailed summaries from a long transcript. Combine them into a single, comprehensive, and coherent final summary that is at least several paragraphs long. Include all key topics, details, decisions, and action items from the entire transcript: ",
                'format_instructions': "Create a detailed, comprehensive final summary combining all the partial summaries."
            }
        }

    def _initialize_models(self):
        """Initialize tokenizer and model (base or fine-tuned)."""
        try:
            self.logger.info(f"Loading tokenizer for {self.base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                cache_dir=self.model_cache_dir,
                model_max_length=self.max_input_length # Set max length for tokenizer
            )

            # Check if fine-tuned model exists and has required files
            fine_tuned_path = Path(self.fine_tuned_model_path)
            required_files = ['adapter_config.json', 'adapter_model.safetensors']
            has_required_files = (
                fine_tuned_path.exists() and 
                all((fine_tuned_path / f).exists() for f in required_files)
            )
            
            if has_required_files:
                self.logger.info("Fine-tuned model files found. Loading fine-tuned model...")
                self._load_fine_tuned_model()
            else:
                missing_files = [f for f in required_files if not (fine_tuned_path / f).exists()]
                if fine_tuned_path.exists():
                    self.logger.warning(
                        f"Fine-tuned model directory exists but missing required files: {missing_files}. "
                        "Loading base model instead."
                    )
                else:
                    self.logger.info("Fine-tuned model not found, loading base model...")
                self._load_base_model()

        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise

    def _load_base_model(self):
        """Load the base model and prepare for fine-tuning."""
        self.logger.info(f"Loading base model: {self.base_model_name}")
        self._is_fine_tuned = False  # We're loading the base model

        # --- THIS IS THE FIX ---
        # Set device_map to None when on CPU to prevent the 'accelerate' error
        model_kwargs = {
            "cache_dir": self.model_cache_dir,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        # ------------------------
            
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.base_model_name,
            cache_dir=self.model_cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None 
        )

        # Apply LoRA if not already fine-tuned
        if not hasattr(self.model, 'peft_config'):
            self.logger.info("Applying LoRA configuration to base model...")
            self.model = get_peft_model(self.model, self.lora_config)
            self.model.print_trainable_parameters()

        # Manually move to CPU if no device_map was used
        if self.device.type == 'cpu' and "device_map" not in model_kwargs:
             self.model.to(self.device)
             self.logger.info("Manually moved base model to CPU.")
        
        self.logger.info("Base model loaded (not fine-tuned - will use base BART behavior)")


    def _load_fine_tuned_model(self):
        """Load the fine-tuned model with LoRA weights."""
        import traceback
        
        try:
            self.logger.info(f"Attempting to load fine-tuned model from: {self.fine_tuned_model_path}")
            self.logger.info(f"Device: {self.device}, CUDA available: {torch.cuda.is_available()}")
            
            # Set device_map to None when on CPU to avoid accelerate issues
            use_device_map = torch.cuda.is_available()
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            self.logger.info("Step 1: Loading base model...")
            # Load base model
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.base_model_name,
                cache_dir=self.model_cache_dir,
                torch_dtype=torch_dtype,
                device_map="auto" if use_device_map else None
            )
            
            if not use_device_map:
                base_model.to(self.device)
                self.logger.info("Base model moved to CPU.")
            
            self.logger.info("Step 2: Loading LoRA adapter weights...")
            # Load LoRA weights
            self.model = PeftModel.from_pretrained(
                base_model,
                self.fine_tuned_model_path,
                torch_dtype=torch_dtype,
            )
            
            # Manually move to CPU if no device_map was used
            if not use_device_map:
                self.model.to(self.device)
                self.logger.info("Fine-tuned model moved to CPU.")
            
            # Verify the model was loaded correctly
            if hasattr(self.model, 'peft_config') and self.model.peft_config:
                self.logger.info("Fine-tuned model loaded successfully!")
                self.logger.info(f"LoRA config: {list(self.model.peft_config.keys())}")
                # Set a flag to track that we're using fine-tuned model
                self._is_fine_tuned = True
            else:
                raise ValueError("Model loaded but PEFT config is missing. Model may not be properly fine-tuned.")
                
        except FileNotFoundError as e:
            error_msg = f"Fine-tuned model file not found: {e}"
            self.logger.error(error_msg)
            self.logger.error(f"Expected path: {self.fine_tuned_model_path}")
            self.logger.error(traceback.format_exc())
            self.logger.info("Falling back to base model...")
            self._load_base_model()
        except Exception as e:
            error_msg = f"Failed to load fine-tuned model: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error("Full traceback:")
            self.logger.error(traceback.format_exc())
            self.logger.info("Falling back to base model...")
            self._load_base_model()

    async def _generate_single_pass(self, transcript: str, config: Dict) -> Dict[str, Any]:
        """
        Generate a summary from a single transcript chunk (the original method).
        """
        start_time = time.time()
        style = config.get('style', 'comprehensive')
        
        try:
            # Prepare input (truncates if necessary)
            prepared_input = self._prepare_input(transcript, style)

            # Generate summary
            summary = await self._generate_with_model(prepared_input, config)

            # Post-process summary
            processed_summary = self._post_process_summary(summary, style)

            # Calculate metrics
            processing_time = time.time() - start_time
            input_len = len(transcript)
            output_len = len(processed_summary)

            result = {
                'summary': processed_summary,
                'metadata': {
                    'style': style,
                    'processing_time': processing_time,
                    'input_length': input_len,
                    'output_length': output_len,
                    'compression_ratio': output_len / input_len if input_len > 0 else 0,
                    'model_used': 'fine_tuned' if self._is_fine_tuned else 'base',
                    'generation_config': self.generation_config.to_dict(),
                    'strategy': 'single_pass'
                }
            }
            return result
        except Exception as e:
            self.logger.error(f"Summary generation (single pass) failed: {e}")
            raise

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Breaks text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
            
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            
            if end >= len(text):
                break
                
            start += (chunk_size - overlap)
            
        return chunks

    async def generate_summary(self, transcript: str, config: Dict) -> Dict[str, Any]:
        """
        Generate a summary from transcript, handling long inputs with Map-Reduce.
        """
        start_time = time.time()
        
        # Calculate effective input length (model max_len - buffer for prompt)
        # Use a larger buffer for the prompt to be safe
        prompt_buffer = 200  # Increased buffer for longer prompts 
        effective_max_input = self.max_input_length - prompt_buffer

        # 1. CHECK LENGTH: If transcript is short enough, run single-pass summarization
        if len(transcript) <= effective_max_input:
            self.logger.info("Transcript is short. Running single-pass summarization.")
            return await self._generate_single_pass(transcript, config)

        # 2. MAP-REDUCE: If transcript is long
        self.logger.warning(f"Transcript length ({len(transcript)}) exceeds limit. Starting Map-Reduce summarization.")

        # --- MAP STEP ---
        # Break the long transcript into overlapping chunks
        # Use a slightly smaller chunk size to account for the 'chunk_summary' prompt
        chunk_size = effective_max_input
        overlap = 150  # ~2-3 sentences of overlap
        text_chunks = self._chunk_text(transcript, chunk_size, overlap)
        self.logger.info(f"Split transcript into {len(text_chunks)} chunks.")

        chunk_summaries = []
        map_config = config.copy()
        map_config['style'] = 'chunk_summary'
        # Increased chunk summary lengths for better detail preservation
        map_config['min_length'] = 80 # Longer chunk summaries to preserve more detail
        map_config['max_length'] = 300 # Increased max length for chunks


        for i, chunk in enumerate(text_chunks):
            self.logger.info(f"Summarizing chunk {i+1}/{len(text_chunks)}...")
            try:
                chunk_summary_result = await self._generate_single_pass(chunk, map_config)
                chunk_summaries.append(chunk_summary_result['summary'])
            except Exception as e:
                self.logger.error(f"Failed to summarize chunk {i+1}: {e}")
                chunk_summaries.append(f"[Error summarizing chunk {i+1}]")

        # --- REDUCE STEP ---
        # Combine all the chunk summaries into one document
        combined_summary_text = "\n".join(chunk_summaries)
        self.logger.info(f"Generating final summary from combined text of length {len(combined_summary_text)}...")

        # Prepare config for the final "reduce" summarization
        reduce_config = config.copy()
        reduce_config['style'] = 'final_summary'
        # Use the original min/max length from the agent for the final summary
        # Ensure min_length is at least 200 for longer summaries
        default_min = max(200, config.get('min_length', self.config.get('min_summary_length', 200)))
        reduce_config['min_length'] = default_min
        # Increase max_length significantly for final summary
        default_max = max(800, config.get('max_length', self.config.get('max_summary_length', 1024)))
        reduce_config['max_length'] = default_max

        # --- RECURSIVE FIX ---
        # Run the final summarization recursively
        # This will handle the case where the combined_summary_text is *still* too long
        final_summary_result = await self.generate_summary(combined_summary_text, reduce_config)
        # ---------------------

        # Update metadata to reflect the Map-Reduce strategy
        total_time = time.time() - start_time
        # Only update metadata if this is the top-level call, not a recursive one
        if config.get('style') != 'final_summary':
            final_summary_result['metadata']['strategy'] = 'map_reduce'
            final_summary_result['metadata']['chunks_processed'] = len(text_chunks)
            final_summary_result['metadata']['processing_time'] = total_time
            final_summary_result['metadata']['original_input_length'] = len(transcript)
            final_summary_result['metadata']['compression_ratio'] = len(final_summary_result['summary']) / len(transcript) if len(transcript) > 0 else 0
        
        self.logger.info(f"Map-Reduce summarization step completed in {total_time:.2f}s")
        return final_summary_result


    async def _generate_with_model(self, input_text: str, config: Dict) -> str:
        """Generate summary using the model."""
        model_type = "fine-tuned" if self._is_fine_tuned else "base"
        self.logger.debug(f"Generating summary using {model_type} model")
        self.logger.debug(f"Input text length: {len(input_text)} characters")
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length, # Use the class attribute
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        self.logger.debug(f"Tokenized input shape: {inputs['input_ids'].shape}")

        # Adjust generation config if specified
        # Create a deep copy to avoid modifying the class default
        generation_config = copy.deepcopy(self.generation_config) 
        if 'max_length' in config:
            generation_config.max_length = config['max_length']
        # --- FIX: Apply min_length from config ---
        if 'min_length' in config:
            generation_config.min_length = config['min_length']
        # ----------------------------------------

        self.logger.debug(f"Generation config: max_length={generation_config.max_length}, min_length={generation_config.min_length}")

        # Generate summary
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            except Exception as e:
                self.logger.error(f"Error during model generation: {e}")
                self.logger.error(f"Using {model_type} model, device: {self.device}")
                raise

        # Decode output
        summary = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        self.logger.debug(f"Generated summary length: {len(summary)} characters")
        # Log first 200 chars of output for debugging
        if len(summary) > 200:
            self.logger.debug(f"Summary preview: {summary[:200]}...")
        else:
            self.logger.debug(f"Summary: {summary}")
        
        # Check if output looks suspicious (matches input exactly or contains the prompt)
        if summary.strip() == input_text.strip() or input_text.strip().startswith(summary.strip()):
            self.logger.warning(
                f"WARNING: Generated summary appears to match or be part of the input! "
                f"This may indicate the {model_type} model is not working correctly. "
                f"Using model: {model_type}, Device: {self.device}"
            )

        return summary

    def _prepare_input(self, transcript: str, style: str) -> str:
        """Prepare input text with style-specific formatting."""
        template = self.style_templates.get(style, self.style_templates['comprehensive'])

        # Add style prefix
        formatted_input = template['prefix'] + transcript

        # Truncate if too long for model context
        # Use the class attribute for max_input_length
        if len(formatted_input) > self.max_input_length:
            # Keep the prefix and truncate the transcript
            prefix = template['prefix']
            available_length = self.max_input_length - len(prefix) - 10  # Buffer
            
            # Ensure available_length is positive
            if available_length > 0:
                truncated_transcript = transcript[:available_length] + "..."
                formatted_input = prefix + truncated_transcript
            else:
                # Prefix itself is too long, just truncate the prefix
                formatted_input = prefix[:self.max_input_length - 3] + "..."

            self.logger.warning(f"Input truncated to {self.max_input_length} characters for model input.")

        return formatted_input


    def _post_process_summary(self, summary: str, style: str) -> str:
        """Post-process the generated summary.

        Besides normal cleanup, aggressively strips any instruction-style prefix
        that some base models may echo (e.g., when fine-tuned adapter fails to load).
        """
        # Remove any residual prefix from the output (match against all known templates)
        template = self.style_templates.get(style, self.style_templates['comprehensive'])
        prefix = template['prefix']

        # First try exact match for this style
        if summary.startswith(prefix):
            summary = summary[len(prefix):].strip()
        else:
            # Fall back: check against all style prefixes (case-insensitive)
            lowered = summary.lower().lstrip()
            for tpl in self.style_templates.values():
                p = tpl['prefix']
                if lowered.startswith(p.lower()):
                    # Compute offset accounting for any leading whitespace removed by lstrip
                    strip_offset = len(summary) - len(lowered)
                    summary = summary[strip_offset + len(p):].strip()
                    break

            # Extra guard: strip common instruction phrasing seen in older prompts
            common_instr = [
                "the following text consists of several detailed summaries from a long transcript",
                "summarize each section of the transcript in detail",
                "combine them into a single, comprehensive, and coherent final summary",
            ]
            lowered = summary.lower().lstrip()
            for instr in common_instr:
                if lowered.startswith(instr):
                    # Remove up to the first colon if present; otherwise remove the matched phrase
                    colon_idx = summary.find(":")
                    if colon_idx != -1 and colon_idx < 400:  # avoid chopping real content
                        summary = summary[colon_idx + 1:].strip()
                    else:
                        strip_offset = len(summary) - len(lowered)
                        summary = summary[strip_offset + len(instr):].strip()
                    break

        # Clean up the summary
        summary = self._clean_summary_text(summary)

        # Add style-specific formatting if needed
        if style == 'structured_academic':
            summary = self._format_academic_summary(summary)
        elif style == 'action_oriented':
            summary = self._format_action_summary(summary)

        return summary

    def _clean_summary_text(self, text: str) -> str:
        """Clean and normalize summary text."""
        import re

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Fix punctuation spacing issues
        text = re.sub(r'\s+([,.!?;:])', r'\1', text) # Remove space before punctuation

        # Ensure sentences start with capital letters
        # Use regex to find sentence endings followed by optional whitespace and a lowercase letter
        def capitalize_sentence(match):
            return match.group(1) + match.group(2).upper()
        
        text = re.sub(r'([.?!]\s+)([a-z])', capitalize_sentence, text)

        # Capitalize the very first letter if it's lowercase
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        return text.strip()

    def _format_academic_summary(self, summary: str) -> str:
        """Format summary for academic content."""
        # Add basic structure if not present
        if '1)' not in summary and 'topics' in summary.lower():
            # Simple restructuring attempt
            sentences = summary.split('. ')
            if len(sentences) >= 3:
                # Ensure last sentence has a period if it was split
                last_sentence = sentences[-1] if sentences[-1].endswith('.') else sentences[-1] + '.'
                middle_sentences = '. '.join(sentences[1:-1])
                # Ensure middle part has a period if not empty
                if middle_sentences: middle_sentences += '.'

                return f"Main Topics: {sentences[0]}. Key Concepts: {middle_sentences} Important Details: {last_sentence}"

        return summary

    def _format_action_summary(self, summary: str) -> str:
        """Format summary for action-oriented content."""
        # Highlight action items if present
        action_words = ['decided', 'agreed', 'assigned', 'action', 'next steps', 'follow up', 'task', 'to do'] # Added more keywords

        sentences = summary.split('. ')
        action_sentences = []
        other_sentences = []

        for sentence in sentences:
            if not sentence: continue # Skip empty strings
            if any(word in sentence.lower() for word in action_words):
                action_sentences.append(sentence)
            else:
                other_sentences.append(sentence)

        if action_sentences:
            # Join sentences, ensuring proper punctuation
            formatted = '. '.join(filter(None, other_sentences))
            if formatted and not formatted.endswith('.'): formatted += '.'
            
            action_part = '. '.join(filter(None, action_sentences))
            if action_part and not action_part.endswith('.'): action_part += '.'

            if formatted:
                formatted += ' Action Items: ' + action_part
            else:
                formatted = 'Action Items: ' + action_part
            return formatted

        return summary

    async def fine_tune_model(self, training_data: List[Dict], validation_data: List[Dict] = None):
        """
        Fine-tune the model on custom data using LoRA.

        Args:
            training_data: List of {'input': str, 'target': str} dictionaries
            validation_data: Optional validation data in same format
        """
        self.logger.info("Starting model fine-tuning...")

        try:
            # Prepare datasets
            train_dataset = self._prepare_dataset(training_data)
            eval_dataset = self._prepare_dataset(validation_data) if validation_data else None

            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=self.fine_tuned_model_path,
                per_device_train_batch_size=self.config.get('batch_size', 4),
                per_device_eval_batch_size=self.config.get('batch_size', 4),
                gradient_accumulation_steps=self.config.get('gradient_accumulation', 2),
                learning_rate=self.config.get('learning_rate', 1e-4),
                num_train_epochs=self.config.get('num_epochs', 3),
                warmup_steps=self.config.get('warmup_steps', 100),
                logging_steps=50,
                evaluation_strategy="steps" if eval_dataset else "no",
                eval_steps=100 if eval_dataset else None,
                save_steps=200,
                save_total_limit=3,
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model="eval_loss" if eval_dataset else None,
                fp16=torch.cuda.is_available(),
                dataloader_pin_memory=False,
                remove_unused_columns=False,
            )

            # Data collator
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True,
                max_length=self.max_input_length # Use class attribute
            )

            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )

            # Train the model
            self.logger.info("Starting training...")
            trainer.train()

            # Save the model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.fine_tuned_model_path)

            self.logger.info(f"Fine-tuning completed. Model saved to {self.fine_tuned_model_path}")

        except Exception as e:
            self.logger.error(f"Fine-tuning failed: {e}")
            raise

    def _prepare_dataset(self, data: List[Dict]) -> Dataset:
        """Prepare dataset for training."""
        def tokenize_function(examples):
            # Tokenize inputs
            model_inputs = self.tokenizer(
                examples['input'],
                max_length=self.max_input_length, # Use class attribute
                truncation=True,
                padding=True
            )

            # Tokenize targets
            # Use text_target for labels
            labels = self.tokenizer(
                text_target=examples['target'], # Use text_target
                max_length=self.config.get('max_output_length', 512),
                truncation=True,
                padding=True
            )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Filter out empty or None data
        valid_data = [item for item in data if item.get('input') and item.get('target')]
        if len(valid_data) < len(data):
            self.logger.warning(f"Removed {len(data) - len(valid_data)} empty training examples.")
            
        if not valid_data:
            self.logger.error("No valid training data provided.")
            return None # Return None if no valid data

        # Convert to Hugging Face dataset
        dataset_dict = {
            'input': [item['input'] for item in valid_data],
            'target': [item['target'] for item in valid_data]
        }
        
        try:
             dataset = Dataset.from_dict(dataset_dict)
        except Exception as e:
             self.logger.error(f"Failed to create dataset from dict: {e}")
             self.logger.error(f"Data causing error (first item): {dataset_dict['input'][0] if dataset_dict['input'] else 'N/A'}")
             raise

        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        return tokenized_dataset

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'base_model': self.base_model_name,
            'is_fine_tuned': self._is_fine_tuned,
            'has_peft_config': hasattr(self.model, 'peft_config') if self.model else False,
            'lora_config': self.lora_config.__dict__ if self.lora_config else None,
            'device': str(self.device),
            'generation_config': self.generation_config.to_dict(),
            'supported_styles': list(self.style_templates.keys()),
            'fine_tuned_path': self.fine_tuned_model_path
        }
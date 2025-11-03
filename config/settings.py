"""
Configuration management for the Audio-to-Summary AI Agent.

This module provides a centralized configuration system that loads settings
from YAML files and environment variables.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class AudioProcessingConfig:
    """Configuration for audio processing."""
    sample_rate: int = 16000
    chunk_duration: int = 30  # seconds
    overlap_duration: int = 2  # seconds
    whisper_model: str = 'base'
    noise_reduction: bool = True
    silence_removal: bool = True


@dataclass
class ModelConfig:
    """Configuration for fine-tuned models."""
    base_model: str = 'facebook/bart-large-cnn'
    cache_dir: str = './models/cache'
    fine_tuned_path: str = './models/fine_tuned'
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_targets: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "out_proj"])
    
    # Generation parameters
    max_summary_length: int = 1024
    min_summary_length: int = 200
    length_penalty: float = 2.0
    num_beams: int = 4
    max_input_length: int = 2048
    max_output_length: int = 512
    
    # Training parameters
    batch_size: int = 4
    gradient_accumulation: int = 2
    learning_rate: float = 1e-4
    num_epochs: int = 3
    warmup_steps: int = 100


@dataclass
class TaskExtractionConfig:
    """Configuration for task extraction."""
    confidence_threshold: float = 0.5
    similarity_threshold: float = 0.8
    max_tasks_per_content: int = 20
    min_task_length: int = 3
    max_task_length: int = 200


@dataclass
class EvaluationConfig:
    """Configuration for quality evaluation."""
    transcript_threshold: float = 0.8
    summary_threshold: float = 0.7
    task_threshold: float = 0.75
    overall_threshold: float = 0.7
    
    # Evaluation weights
    transcript_weight: float = 0.3
    summary_weight: float = 0.4
    task_weight: float = 0.3


@dataclass
class UIConfig:
    """Configuration for user interface."""
    host: str = "localhost"
    port: int = 8501
    debug: bool = False
    max_file_size_mb: int = 100


@dataclass
class VideoOCRConfig:
    frame_sample_rate: float = 1.0
    min_text_length: int = 3
    languages: list = field(default_factory=lambda: ["en"])
    detect_captions_only: bool = True


@dataclass
class VideoProcessingConfig:
    ocr: VideoOCRConfig = field(default_factory=VideoOCRConfig)


class Settings:
    """
    Main settings class that manages all configuration.
    """
    
    def __init__(self):
        """Initialize settings with default values."""
        self.audio_processing = AudioProcessingConfig()
        self.model = ModelConfig()
        self.task_extraction = TaskExtractionConfig()
        self.evaluation = EvaluationConfig()
        self.ui = UIConfig()
        self.video_processing = VideoProcessingConfig()
        
        # General settings
        self.log_level = "INFO"
        self.log_file = "audio_agent.log"
        self.data_dir = "./data"
        self.output_dir = "./output"
        
        # Environment-specific settings
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.debug = self.environment == 'development'
        
        # Load environment variables
        self._load_from_env()
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Settings':
        """
        Load settings from a YAML configuration file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Settings instance with loaded configuration
        """
        settings = cls()
        
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                settings._update_from_dict(config_data)
                logging.info(f"Configuration loaded from {config_path}")
            else:
                logging.warning(f"Configuration file {config_path} not found, using defaults")
                
        except Exception as e:
            logging.error(f"Failed to load configuration from {config_path}: {e}")
            logging.info("Using default configuration")
        
        return settings
    
    def _load_from_env(self):
        """Load settings from environment variables."""
        # General settings
        self.log_level = os.getenv('LOG_LEVEL', self.log_level)
        self.data_dir = os.getenv('DATA_DIR', self.data_dir)
        self.output_dir = os.getenv('OUTPUT_DIR', self.output_dir)
        
        # Model settings
        self.model.base_model = os.getenv('BASE_MODEL', self.model.base_model)
        self.model.cache_dir = os.getenv('MODEL_CACHE_DIR', self.model.cache_dir)
        self.model.fine_tuned_path = os.getenv('FINE_TUNED_MODEL_PATH', self.model.fine_tuned_path)
        
        # Audio processing
        if os.getenv('WHISPER_MODEL'):
            self.audio_processing.whisper_model = os.getenv('WHISPER_MODEL')
        
        if os.getenv('SAMPLE_RATE'):
            self.audio_processing.sample_rate = int(os.getenv('SAMPLE_RATE'))
        
        # UI settings
        self.ui.host = os.getenv('UI_HOST', self.ui.host)
        if os.getenv('UI_PORT'):
            self.ui.port = int(os.getenv('UI_PORT'))
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """Update settings from a dictionary."""
        # Update audio processing settings
        if 'audio_processing' in config_dict:
            audio_config = config_dict['audio_processing']
            for key, value in audio_config.items():
                if hasattr(self.audio_processing, key):
                    setattr(self.audio_processing, key, value)
        
        # Update model settings
        if 'model' in config_dict:
            model_config = config_dict['model']
            for key, value in model_config.items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
        
        # Update task extraction settings
        if 'task_extraction' in config_dict:
            task_config = config_dict['task_extraction']
            for key, value in task_config.items():
                if hasattr(self.task_extraction, key):
                    setattr(self.task_extraction, key, value)
        
        # Update evaluation settings
        if 'evaluation' in config_dict:
            eval_config = config_dict['evaluation']
            for key, value in eval_config.items():
                if hasattr(self.evaluation, key):
                    setattr(self.evaluation, key, value)
        
        # Update UI settings
        if 'ui' in config_dict:
            ui_config = config_dict['ui']
            for key, value in ui_config.items():
                if hasattr(self.ui, key):
                    setattr(self.ui, key, value)
        
        # Update general settings
        for key in ['log_level', 'log_file', 'data_dir', 'output_dir', 'environment', 'debug']:
            if key in config_dict:
                setattr(self, key, config_dict[key])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary format."""
        return {
            'audio_processing': {
                'sample_rate': self.audio_processing.sample_rate,
                'chunk_duration': self.audio_processing.chunk_duration,
                'overlap_duration': self.audio_processing.overlap_duration,
                'whisper_model': self.audio_processing.whisper_model,
                'noise_reduction': self.audio_processing.noise_reduction,
                'silence_removal': self.audio_processing.silence_removal,
            },
            'model': {
                'base_model': self.model.base_model,
                'cache_dir': self.model.cache_dir,
                'fine_tuned_path': self.model.fine_tuned_path,
                'lora_r': self.model.lora_r,
                'lora_alpha': self.model.lora_alpha,
                'lora_dropout': self.model.lora_dropout,
                'lora_targets': self.model.lora_targets,
                'max_summary_length': self.model.max_summary_length,
                'min_summary_length': self.model.min_summary_length,
                'length_penalty': self.model.length_penalty,
                'num_beams': self.model.num_beams,
                'max_input_length': self.model.max_input_length,
                'max_output_length': self.model.max_output_length,
                'batch_size': self.model.batch_size,
                'gradient_accumulation': self.model.gradient_accumulation,
                'learning_rate': self.model.learning_rate,
                'num_epochs': self.model.num_epochs,
                'warmup_steps': self.model.warmup_steps,
            },
            'task_extraction': {
                'confidence_threshold': self.task_extraction.confidence_threshold,
                'similarity_threshold': self.task_extraction.similarity_threshold,
                'max_tasks_per_content': self.task_extraction.max_tasks_per_content,
                'min_task_length': self.task_extraction.min_task_length,
                'max_task_length': self.task_extraction.max_task_length,
            },
            'evaluation': {
                'transcript_threshold': self.evaluation.transcript_threshold,
                'summary_threshold': self.evaluation.summary_threshold,
                'task_threshold': self.evaluation.task_threshold,
                'overall_threshold': self.evaluation.overall_threshold,
                'transcript_weight': self.evaluation.transcript_weight,
                'summary_weight': self.evaluation.summary_weight,
                'task_weight': self.evaluation.task_weight,
            },
            'ui': {
                'host': self.ui.host,
                'port': self.ui.port,
                'debug': self.ui.debug,
                'max_file_size_mb': self.ui.max_file_size_mb,
            },
            'video_processing': {
                'ocr': {
                    'frame_sample_rate': self.video_processing.ocr.frame_sample_rate,
                    'min_text_length': self.video_processing.ocr.min_text_length,
                    'languages': self.video_processing.ocr.languages,
                    'detect_captions_only': self.video_processing.ocr.detect_captions_only,
                }
            },
            'general': {
                'log_level': self.log_level,
                'log_file': self.log_file,
                'data_dir': self.data_dir,
                'output_dir': self.output_dir,
                'environment': self.environment,
                'debug': self.debug,
            }
        }
    
    def save_to_yaml(self, output_path: str):
        """
        Save current settings to a YAML file.
        
        Args:
            output_path: Path where to save the configuration
        """
        try:
            config_dict = self.to_dict()
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logging.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Failed to save configuration to {output_path}: {e}")
            raise
    
    def validate(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        errors = []
        
        # Validate audio processing settings
        if self.audio_processing.sample_rate <= 0:
            errors.append("Sample rate must be positive")
        
        if self.audio_processing.chunk_duration <= 0:
            errors.append("Chunk duration must be positive")
        
        # Validate model settings
        if self.model.max_summary_length <= self.model.min_summary_length:
            errors.append("Max summary length must be greater than min summary length")
        
        if self.model.lora_r <= 0:
            errors.append("LoRA rank must be positive")
        
        if not (0 <= self.model.lora_dropout <= 1):
            errors.append("LoRA dropout must be between 0 and 1")
        
        # Validate evaluation thresholds
        for threshold_name in ['transcript_threshold', 'summary_threshold', 'task_threshold', 'overall_threshold']:
            threshold_value = getattr(self.evaluation, threshold_name)
            if not (0 <= threshold_value <= 1):
                errors.append(f"{threshold_name} must be between 0 and 1")
        
        # Validate weights sum to 1
        weights_sum = (self.evaluation.transcript_weight + 
                      self.evaluation.summary_weight + 
                      self.evaluation.task_weight)
        if abs(weights_sum - 1.0) > 0.01:
            errors.append("Evaluation weights must sum to 1.0")
        
        # Log errors if any
        if errors:
            for error in errors:
                logging.error(f"Configuration validation error: {error}")
            return False
        
        logging.info("Configuration validation passed")
        return True
    
    def create_directories(self):
        """Create necessary directories based on configuration."""
        directories = [
            self.data_dir,
            self.output_dir,
            self.model.cache_dir,
            self.model.fine_tuned_path,
            os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else '.'
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logging.debug(f"Created directory: {directory}")
            except Exception as e:
                logging.error(f"Failed to create directory {directory}: {e}")
    
    def get_config_dict_for_component(self, component: str) -> Dict[str, Any]:
        """
        Get configuration dictionary for a specific component.
        
        Args:
            component: Name of the component ('audio_processing', 'model', etc.)
            
        Returns:
            Configuration dictionary for the component
        """
        component_configs = {
            'audio_processing': self.audio_processing.__dict__,
            'model': self.model.__dict__,
            'task_extraction': self.task_extraction.__dict__,
            'evaluation': self.evaluation.__dict__,
            'ui': self.ui.__dict__
        }
        
        return component_configs.get(component, {})


# Global settings instance
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


def initialize_settings(config_path: Optional[str] = None) -> Settings:
    """
    Initialize global settings from configuration file.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Initialized settings instance
    """
    global _settings_instance
    
    if config_path:
        _settings_instance = Settings.from_yaml(config_path)
    else:
        _settings_instance = Settings()
    
    # Validate and create directories
    _settings_instance.validate()
    _settings_instance.create_directories()
    
    return _settings_instance

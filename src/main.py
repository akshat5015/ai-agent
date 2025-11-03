#!/usr/bin/env python3
"""
Main entry point for the Audio-to-Summary AI Agent.

This module orchestrates the complete pipeline from audio input to 
summary and task list generation using fine-tuned models.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any

from agents.audio_summary_agent import AudioSummaryAgent
from config.settings import Settings


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('audio_agent.log'),
            logging.StreamHandler()
        ]
    )


async def process_audio_file(
    input_path: str,
    output_path: str,
    settings: Settings
) -> Dict[str, Any]:
    """
    Process a single audio file through the complete pipeline.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output JSON file
        settings: Configuration settings
        
    Returns:
        Dictionary containing processing results
    """
    logger = logging.getLogger(__name__)
    
    # Initialize the AI agent
    agent = AudioSummaryAgent(settings)
    
    try:
        # Process the audio file
        logger.info(f"Starting processing of {input_path}")
        results = await agent.process_audio(input_path)
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processing complete. Results saved to {output_path}")
        return results
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}")
        raise


def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Audio-to-Summary AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input audio file"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path for output summary JSON file"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load settings
    settings = Settings.from_yaml(args.config)
    
    # Validate input file
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process the audio file
    try:
        results = asyncio.run(process_audio_file(
            args.input,
            args.output,
            settings
        ))
        
        # Print summary
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Transcript Length: {len(results.get('transcript', ''))} characters")
        print(f"Summary Length: {len(results.get('summary', ''))} characters")
        print(f"Tasks Identified: {len(results.get('tasks', []))}")
        print("="*50)
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to process audio file: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())

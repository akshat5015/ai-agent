"""
Training script for fine-tuning the summarization model using LoRA.

This script provides functionality to fine-tune the base model on
domain-specific data for better performance on audio transcripts.
"""

import asyncio
import logging
import json
import argparse
from pathlib import Path
import sys
from typing import List, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.fine_tuned_summarizer import FineTunedSummarizer
from config.settings import initialize_settings


def setup_logging(log_level: str = "INFO"):
    """Setup logging for training."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )


def load_training_data(data_path: str) -> List[Dict[str, str]]:
    """
    Load training data from JSON file.
    
    Expected format: List of {"input": "transcript text", "target": "summary text"}
    """
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logging.info(f"Loaded {len(data)} training examples from {data_path}")
        return data
        
    except Exception as e:
        logging.error(f"Failed to load training data from {data_path}: {e}")
        raise


def create_sample_training_data() -> List[Dict[str, str]]:
    """Create sample training data for demonstration."""
    return [
        {
            "input": "Today's lecture covered the fundamentals of machine learning. We discussed supervised learning, which involves training algorithms on labeled data. The professor explained that supervised learning includes classification tasks, where we predict categories, and regression tasks, where we predict continuous values. We also talked about unsupervised learning, which works with unlabeled data to find patterns. Key algorithms mentioned were linear regression, decision trees, and neural networks. The homework assignment is to implement a simple linear regression model and submit it by next Friday. We need to use the dataset provided on the course website.",
            "target": "Machine Learning Fundamentals: Covered supervised learning (classification and regression) and unsupervised learning. Key algorithms: linear regression, decision trees, neural networks. Assignment: Implement linear regression model, due next Friday using course dataset."
        },
        {
            "input": "In today's team meeting, we discussed the quarterly project status. Sarah reported that the backend API development is 80% complete and should be finished by end of this week. Mike mentioned that the frontend user interface needs two more weeks of work, with the login system being the main remaining task. We decided to prioritize the authentication module since it's blocking other features. The deployment to staging environment is scheduled for next Monday. Everyone agreed to increase code review frequency to twice per week. Action items: Sarah will complete API documentation, Mike will focus on login system, and the team will meet again on Thursday to review progress.",
            "target": "Project Status Update: Backend API 80% complete (finish this week), Frontend UI needs 2 weeks (login system priority). Staging deployment: next Monday. Decisions: Prioritize authentication, increase code reviews to twice weekly. Action Items: Sarah - API documentation, Mike - login system, Team meeting Thursday."
        },
        {
            "input": "Study session for tomorrow's chemistry exam focused on organic chemistry reactions. We reviewed nucleophilic substitution reactions, specifically SN1 and SN2 mechanisms. SN1 reactions proceed through carbocation intermediates and work better with tertiary substrates, while SN2 reactions have a concerted mechanism and prefer primary substrates. We practiced identifying reaction conditions that favor each mechanism. The group worked through several practice problems from chapter 8. Areas that need more review include stereochemistry effects and the role of leaving groups. Plan to study elimination reactions tonight and meet again tomorrow morning for a final review session before the 2 PM exam.",
            "target": "Chemistry Exam Prep: Studied nucleophilic substitution (SN1 vs SN2 mechanisms). SN1: carbocation intermediate, tertiary substrates. SN2: concerted, primary substrates. Completed chapter 8 problems. Need more review: stereochemistry, leaving groups. Tonight: elimination reactions. Tomorrow: final review before 2 PM exam."
        },
        {
            "input": "Product development meeting to discuss the new mobile app features. The team presented wireframes for the social sharing functionality. Key features include photo sharing, comment threads, and user profiles. Technical requirements include implementing OAuth authentication, setting up cloud storage for images, and designing a scalable database schema. Marketing suggested adding gamification elements like user badges and achievement systems. The development timeline estimates 6 weeks for core features and an additional 2 weeks for gamification. Budget approval needed for cloud storage costs estimated at $500 monthly. Next steps include getting stakeholder approval and starting the development sprint.",
            "target": "Mobile App Development: Reviewed social sharing wireframes (photo sharing, comments, profiles). Technical needs: OAuth, cloud storage, scalable database. Marketing request: gamification (badges, achievements). Timeline: 6 weeks core features + 2 weeks gamification. Budget: $500/month cloud storage. Next: stakeholder approval, start development sprint."
        },
        {
            "input": "Research presentation on renewable energy technologies covered solar, wind, and hydroelectric power systems. Solar panels have improved efficiency rates reaching 22% in commercial applications, with costs decreasing 15% annually. Wind turbine technology advances include larger blade designs that capture more energy at lower wind speeds. Hydroelectric systems remain the most reliable renewable source but have limited expansion opportunities due to environmental concerns. The presenter discussed energy storage challenges, particularly battery technology limitations and grid integration issues. Government incentives are driving adoption, with tax credits available through 2025. The class assignment is to analyze one renewable technology and write a 2000-word report due in two weeks.",
            "target": "Renewable Energy Research: Solar efficiency at 22%, costs down 15% yearly. Wind: larger blades for low-speed capture. Hydro: reliable but environmentally limited. Storage challenges: battery tech, grid integration. Government incentives through 2025. Assignment: 2000-word technology analysis, due in 2 weeks."
        }
    ]


async def train_model(
    training_data_path: str,
    validation_data_path: str = None,
    config_path: str = None,
    output_path: str = None
):
    """Train the fine-tuned summarization model."""
    
    # Initialize settings
    if config_path:
        settings = initialize_settings(config_path)
    else:
        # Use default config
        default_config_path = Path(__file__).parent.parent / "config" / "default.yaml"
        settings = initialize_settings(str(default_config_path))
    
    # Load training data
    if training_data_path == "sample":
        logging.info("Using sample training data")
        training_data = create_sample_training_data()
        validation_data = training_data[-1:]  # Use last sample for validation
        training_data = training_data[:-1]  # Use rest for training
    else:
        training_data = load_training_data(training_data_path)
        validation_data = None
        
        if validation_data_path:
            validation_data = load_training_data(validation_data_path)
    
    # Initialize model
    model_config = settings.get_config_dict_for_component('model')
    if output_path:
        model_config['fine_tuned_path'] = output_path
    
    summarizer = FineTunedSummarizer(model_config)
    
    # Start training
    logging.info("Starting model fine-tuning...")
    await summarizer.fine_tune_model(training_data, validation_data)
    
    logging.info("Fine-tuning completed successfully!")
    
    # Test the fine-tuned model
    logging.info("Testing fine-tuned model...")
    test_input = training_data[0]['input']
    test_config = {
        'style': 'comprehensive',
        'max_length': 200
    }
    
    result = await summarizer.generate_summary(test_input, test_config)
    
    logging.info("Test generation completed:")
    logging.info(f"Input length: {len(test_input)} characters")
    logging.info(f"Generated summary: {result['summary']}")
    logging.info(f"Summary length: {len(result['summary'])} characters")
    logging.info(f"Compression ratio: {result['metadata']['compression_ratio']:.2f}")


def create_training_dataset_template():
    """Create a template for training dataset."""
    template_data = [
        {
            "input": "Your transcript text goes here. This should be the full audio transcript that you want to summarize.",
            "target": "Your target summary goes here. This should be the ideal summary for the transcript."
        },
        {
            "input": "Another example transcript...",
            "target": "Another example summary..."
        }
    ]
    
    output_path = Path("training_data_template.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template_data, f, indent=2, ensure_ascii=False)
    
    print(f"Training data template created at: {output_path}")
    print("Fill in your own transcript-summary pairs and use this file for training.")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Fine-tune the summarization model")
    
    parser.add_argument(
        "--training-data",
        default="sample",
        help="Path to training data JSON file (use 'sample' for demo data)"
    )
    
    parser.add_argument(
        "--validation-data",
        help="Path to validation data JSON file"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output-path",
        help="Output path for fine-tuned model"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--create-template",
        action="store_true",
        help="Create a template for training data"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    if args.create_template:
        create_training_dataset_template()
        return
    
    # Run training
    try:
        asyncio.run(train_model(
            args.training_data,
            args.validation_data,
            args.config,
            args.output_path
        ))
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("Your fine-tuned model is ready to use.")
        print("You can now run the main application with improved performance.")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())

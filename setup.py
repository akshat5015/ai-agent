"""
Setup script for the Audio-to-Summary AI Agent.

This script handles environment setup, dependency installation,
and initial configuration for the AI agent.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing Python dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install main requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    return True


def download_nltk_data():
    """Download required NLTK data."""
    print("\n📚 Downloading NLTK data...")
    
    nltk_downloads = [
        'punkt',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words',
        'stopwords'
    ]
    
    for data in nltk_downloads:
        command = f"{sys.executable} -c \"import nltk; nltk.download('{data}', quiet=True)\""
        if not run_command(command, f"Downloading NLTK {data}"):
            print(f"⚠️ Warning: Failed to download NLTK {data}, will attempt during runtime")
    
    return True


def setup_spacy_model():
    """Download spaCy English model."""
    print("\n🧠 Setting up spaCy model...")
    
    # Try to download the English model
    command = f"{sys.executable} -m spacy download en_core_web_sm"
    if not run_command(command, "Downloading spaCy English model"):
        print("⚠️ Warning: spaCy model download failed, will attempt during runtime")
        return False
    
    return True


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating project directories...")
    
    directories = [
        "data/audio",
        "data/transcripts", 
        "data/summaries",
        "models/cache",
        "models/fine_tuned",
        "output",
        "logs"
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False
    
    return True


def create_env_file():
    """Create example environment file."""
    print("\n🔧 Creating environment configuration...")
    
    env_content = """# Environment variables for Audio-to-Summary AI Agent
# Copy this to .env and customize as needed

# General Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
DATA_DIR=./data
OUTPUT_DIR=./output

# Model Settings
BASE_MODEL=facebook/bart-large-cnn
MODEL_CACHE_DIR=./models/cache
FINE_TUNED_MODEL_PATH=./models/fine_tuned
WHISPER_MODEL=base

# Audio Processing
SAMPLE_RATE=16000

# UI Settings
UI_HOST=localhost
UI_PORT=8501

# Optional: Set to enable GPU acceleration
# CUDA_VISIBLE_DEVICES=0
"""
    
    try:
        with open(".env.example", "w") as f:
            f.write(env_content)
        print("✅ Created .env.example file")
        print("   Copy to .env and customize as needed")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env.example: {e}")
        return False


def run_basic_test():
    """Run a basic test to verify installation."""
    print("\n🧪 Running basic installation test...")
    
    test_script = """
import sys
sys.path.append('src')

try:
    # Test imports
    from config.settings import Settings
    from processors.audio_processor import AudioProcessor
    from models.fine_tuned_summarizer import FineTunedSummarizer
    from extractors.task_extractor import TaskExtractor
    from evaluators.quality_assessor import QualityAssessor
    from agents.audio_summary_agent import AudioSummaryAgent
    
    # Test configuration
    settings = Settings()
    print("✅ All core modules imported successfully")
    print("✅ Configuration system working")
    print("✅ Installation test passed")
    
except Exception as e:
    print(f"❌ Installation test failed: {e}")
    sys.exit(1)
"""
    
    try:
        with open("test_installation.py", "w") as f:
            f.write(test_script)
        
        result = subprocess.run([sys.executable, "test_installation.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
            os.remove("test_installation.py")
            return True
        else:
            print(f"❌ Test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Could not run test: {e}")
        return False


def print_usage_instructions():
    """Print usage instructions."""
    print(f"""
{'='*60}
🎉 INSTALLATION COMPLETED SUCCESSFULLY! 🎉
{'='*60}

📋 Quick Start Guide:

1. 🎤 Command Line Usage:
   python src/main.py --input audio_file.wav --output results.json

2. 🌐 Web Interface:
   streamlit run src/ui/app.py

3. 🔧 Fine-tune Model (optional):
   python scripts/train_model.py --training-data sample

4. 📚 Example Commands:

   # Process an audio file
   python src/main.py -i "meeting_recording.mp3" -o "meeting_summary.json"
   
   # Launch web interface
   streamlit run src/ui/app.py
   
   # Train model with custom data
   python scripts/train_model.py --training-data training_data.json

5. 📁 Project Structure:
   - src/: Core AI agent code
   - config/: Configuration files
   - data/: Input data directory
   - models/: Model storage
   - scripts/: Utility scripts

6. 🔗 Supported Audio Formats:
   .wav, .mp3, .m4a, .flac, .ogg, .aac

7. 📖 Documentation:
   See README.md for detailed documentation

{'='*60}
Ready to transform your audio into summaries and tasks! 🚀
{'='*60}
""")


def main():
    """Main setup function."""
    print("🎤 Audio-to-Summary AI Agent Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found. Please run this from the project root directory.")
        sys.exit(1)
    
    success_steps = []
    
    # Run setup steps
    steps = [
        (install_dependencies, "Install Dependencies"),
        (create_directories, "Create Directories"),
        (download_nltk_data, "Download NLTK Data"),
        (setup_spacy_model, "Setup spaCy Model"),
        (create_env_file, "Create Environment File"),
        (run_basic_test, "Run Installation Test")
    ]
    
    for step_func, step_name in steps:
        if step_func():
            success_steps.append(step_name)
        else:
            print(f"\n⚠️ Warning: {step_name} encountered issues")
            print("You may need to address these manually")
    
    # Summary
    print(f"\n📊 Setup Summary:")
    print(f"✅ Successful steps: {len(success_steps)}/{len(steps)}")
    
    for step in success_steps:
        print(f"   ✅ {step}")
    
    if len(success_steps) >= len(steps) - 1:  # Allow one failure
        print_usage_instructions()
    else:
        print("\n❌ Setup encountered multiple issues")
        print("Please check the error messages above and resolve them manually")
        print("You may need to:")
        print("- Check your internet connection")
        print("- Update Python and pip")
        print("- Install system dependencies")
        sys.exit(1)


if __name__ == "__main__":
    main()

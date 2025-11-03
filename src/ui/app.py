"""
Streamlit web interface for the Audio-to-Summary AI Agent.

This provides a user-friendly web interface for uploading audio files,
configuring processing options, and viewing results.
"""

import streamlit as st
import asyncio
import json
import os
import tempfile
from pathlib import Path
import time
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

# Add the parent directory to the path to import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.audio_summary_agent import AudioSummaryAgent
from config.settings import Settings, initialize_settings


def initialize_app():
    """Initialize the Streamlit application."""
    st.set_page_config(
        page_title="Meeting Helper AI Agent",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize settings
    if 'settings' not in st.session_state:
        config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
        st.session_state.settings = initialize_settings(str(config_path))
    
    # Initialize agent
    if 'agent' not in st.session_state:
        st.session_state.agent = AudioSummaryAgent(st.session_state.settings.to_dict())
    
    # Initialize session state variables
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []


def create_sidebar():
    """Create the sidebar with configuration options."""
    st.sidebar.title("🎤 AI Agent Settings")
    
    # Audio processing settings
    st.sidebar.subheader("File Processing")
    
    whisper_model = st.sidebar.selectbox(
        "Whisper Model",
        ["tiny", "base", "small", "medium", "large"],
        index=1,  # default to "base"
        help="Larger models are more accurate but slower"
    )
    
    chunk_duration = st.sidebar.slider(
        "Chunk Duration (seconds)",
        min_value=10,
        max_value=60,
        value=30,
        help="Length of audio chunks for processing"
    )
    
    # Summarization settings
    st.sidebar.subheader("Summarization")
    
    summary_style = st.sidebar.selectbox(
        "Summary Style",
        ["comprehensive", "structured_academic", "action_oriented", "key_points"],
        help="Choose summary style based on content type"
    )
    
    max_summary_length = st.sidebar.slider(
        "Max Summary Length",
        min_value=100,
        max_value=1000,
        value=512,
        help="Maximum length of generated summary"
    )
    
    # Task extraction settings
    st.sidebar.subheader("Task Extraction")
    
    task_focus = st.sidebar.selectbox(
        "Task Focus",
        ["general_todos", "assignments_deadlines", "action_items", "study_tasks"],
        help="Focus area for task extraction"
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Minimum confidence for task inclusion"
    )
    
    # Quality thresholds
    st.sidebar.subheader("Quality Thresholds")
    
    overall_threshold = st.sidebar.slider(
        "Overall Quality Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Minimum quality score for acceptance"
    )
    
    return {
        'whisper_model': whisper_model,
        'chunk_duration': chunk_duration,
        'summary_style': summary_style,
        'max_summary_length': max_summary_length,
        'task_focus': task_focus,
        'confidence_threshold': confidence_threshold,
        'overall_threshold': overall_threshold
    }


def create_file_uploader():
    """Create the file upload interface."""
    st.header("📁 Upload Media File")
    
    uploaded_file = st.file_uploader(
        "Choose an audio or video file",
        type=['wav', 'mp3', 'm4a', 'flac', 'ogg', 'aac', 'mp4', 'mkv', 'mov', 'avi', 'webm', 'm4v'],
        help="Upload your audio/video file for processing"
    )
    
    if uploaded_file is not None:
        # Display file information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("File Name", uploaded_file.name)
        
        with col2:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.metric("File Size", f"{file_size_mb:.2f} MB")
        
        with col3:
            file_type = uploaded_file.type
            st.metric("File Type", file_type)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        return temp_file_path, uploaded_file.name
    
    return None, None


async def process_audio_file(file_path: str, file_name: str, config: Dict[str, Any]):
    """Process the uploaded media file (audio/video)."""
    try:
        # Update agent configuration
        agent_config = st.session_state.settings.to_dict()
        
        # Update with UI settings
        agent_config['audio_processing']['whisper_model'] = config['whisper_model']
        agent_config['audio_processing']['chunk_duration'] = config['chunk_duration']
        agent_config['summarization'] = {
            'style': config['summary_style'],
            'max_length': config['max_summary_length']
        }
        agent_config['task_extraction'] = {
            'focus': config['task_focus'],
            'confidence_threshold': config['confidence_threshold']
        }
        agent_config['evaluation']['overall_threshold'] = config['overall_threshold']
        
        # Process media (auto-detect video/audio)
        results = await st.session_state.agent.process_media(file_path)
        
        # Add processing metadata
        results['processing_metadata'] = {
            'file_name': file_name,
            'config_used': config,
            'processing_timestamp': time.time()
        }
        
        return results
        
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(file_path)
        except:
            pass


def display_results(results: Dict[str, Any]):
    """Display the processing results."""
    if not results:
        return
    
    st.header("📊 Processing Results")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        transcript_length = len(results.get('transcript', ''))
        st.metric("Transcript Length", f"{transcript_length:,} chars")
    
    with col2:
        summary_length = len(results.get('summary', ''))
        st.metric("Summary Length", f"{summary_length:,} chars")
    
    with col3:
        task_count = len(results.get('tasks', []))
        st.metric("Tasks Identified", task_count)
    
    with col4:
        overall_quality = results.get('quality_scores', {}).get('overall_quality', {}).get('score', 0)
        st.metric("Quality Score", f"{overall_quality:.2f}")
    
    # Tabs for different result sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Transcript", "📋 Summary", "✅ Tasks", "📈 Quality", "🔍 Insights"])
    
    with tab1:
        display_transcript(results)
    
    with tab2:
        display_summary(results)
    
    with tab3:
        display_tasks(results)
    
    with tab4:
        display_quality_metrics(results)
    
    with tab5:
        display_insights(results)


def display_transcript(results: Dict[str, Any]):
    """Display the transcript section."""
    st.subheader("📝 File Transcript")
    
    transcript = results.get('transcript', 'No transcript available')
    transcript_metadata = results.get('transcript_metadata', {})
    
    # Transcript metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence = transcript_metadata.get('confidence', 0)
        st.metric("Confidence", f"{confidence:.2f}")
    
    with col2:
        duration = transcript_metadata.get('duration', 0)
        st.metric("Duration", f"{duration:.1f}s")
    
    with col3:
        processing_time = transcript_metadata.get('processing_time', 0)
        st.metric("Processing Time", f"{processing_time:.1f}s")
    
    # Transcript text
    st.text_area(
        "Transcript",
        value=transcript,
        height=300,
        help="The complete transcribed text from your file"
    )


def display_summary(results: Dict[str, Any]):
    """Display the summary section."""
    st.subheader("📋 Generated Summary")
    
    summary = results.get('summary', 'No summary available')
    summary_metadata = results.get('summary_metadata', {})
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        compression_ratio = summary_metadata.get('compression_ratio', 0)
        st.metric("Compression Ratio", f"{compression_ratio:.2f}")
    
    with col2:
        style = summary_metadata.get('style', 'unknown')
        st.metric("Style", style)
    
    with col3:
        processing_time = summary_metadata.get('processing_time', 0)
        st.metric("Generation Time", f"{processing_time:.1f}s")
    
    # Summary text
    st.text_area(
        "Summary",
        value=summary,
        height=200,
        help="AI-generated summary of your file content"
    )


def display_tasks(results: Dict[str, Any]):
    """Display the extracted tasks section."""
    st.subheader("✅ Extracted Tasks")
    
    tasks = results.get('tasks', [])
    task_metadata = results.get('task_metadata', {})
    
    if not tasks:
        st.info("No actionable tasks were identified in this content.")
        return
    
    # Task overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Tasks", len(tasks))
    
    with col2:
        high_priority_count = sum(1 for task in tasks if task.get('priority') == 'high')
        st.metric("High Priority", high_priority_count)
    
    with col3:
        with_deadlines = sum(1 for task in tasks if task.get('deadline'))
        st.metric("With Deadlines", with_deadlines)
    
    # Task list
    for i, task in enumerate(tasks, 1):
        with st.expander(f"Task {i}: {task.get('description', 'No description')[:50]}..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Description:**")
                st.write(task.get('description', 'No description available'))
                
                if task.get('context'):
                    st.write("**Context:**")
                    st.write(task.get('context'))
            
            with col2:
                st.write("**Priority:**", task.get('priority', 'medium'))
                st.write("**Category:**", task.get('category', 'general'))
                
                if task.get('deadline'):
                    st.write("**Deadline:**", task.get('deadline'))
                
                if task.get('assignee'):
                    st.write("**Assignee:**", task.get('assignee'))
                
                confidence = task.get('confidence', 0)
                st.write("**Confidence:**", f"{confidence:.2f}")


def display_quality_metrics(results: Dict[str, Any]):
    """Display quality assessment metrics."""
    st.subheader("📈 Quality Assessment")
    
    quality_scores = results.get('quality_scores', {})
    
    if not quality_scores:
        st.warning("Quality assessment not available.")
        return
    
    # Overall quality
    overall_quality = quality_scores.get('overall_quality', {})
    overall_score = overall_quality.get('score', 0)
    
    # Quality gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = overall_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Quality Score"},
        delta = {'reference': 0.7},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.5], 'color': "lightgray"},
                {'range': [0.5, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.7
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Component scores
    col1, col2, col3 = st.columns(3)
    
    with col1:
        transcript_quality = quality_scores.get('transcript_quality', {}).get('quality_score', 0)
        st.metric("Transcript Quality", f"{transcript_quality:.2f}")
    
    with col2:
        summary_quality = quality_scores.get('summary_quality', {}).get('quality_score', 0)
        st.metric("Summary Quality", f"{summary_quality:.2f}")
    
    with col3:
        task_quality = quality_scores.get('task_quality', {}).get('quality_score', 0)
        st.metric("Task Quality", f"{task_quality:.2f}")
    
    # Detailed metrics
    st.subheader("Detailed Metrics")
    
    # ROUGE scores for summary
    summary_quality_detail = quality_scores.get('summary_quality', {})
    rouge_scores = summary_quality_detail.get('rouge_scores', {})
    
    if rouge_scores:
        st.write("**Summary ROUGE Scores:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("ROUGE-1", f"{rouge_scores.get('rouge1_f', 0):.3f}")
        with col2:
            st.metric("ROUGE-2", f"{rouge_scores.get('rouge2_f', 0):.3f}")
        with col3:
            st.metric("ROUGE-L", f"{rouge_scores.get('rougeL_f', 0):.3f}")
    
    # Task quality breakdown
    task_quality_detail = quality_scores.get('task_quality', {})
    if task_quality_detail:
        st.write("**Task Quality Breakdown:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            relevance = task_quality_detail.get('average_relevance', 0)
            st.metric("Avg Relevance", f"{relevance:.2f}")
        with col2:
            clarity = task_quality_detail.get('average_clarity', 0)
            st.metric("Avg Clarity", f"{clarity:.2f}")
        with col3:
            actionability = task_quality_detail.get('average_actionability', 0)
            st.metric("Avg Actionability", f"{actionability:.2f}")


def display_insights(results: Dict[str, Any]):
    """Display AI-generated insights."""
    st.subheader("🔍 AI-Generated Insights")
    
    insights = results.get('insights', {})
    
    if not insights:
        st.info("No insights available for this processing result.")
        return
    
    # Content analysis
    content_analysis = insights.get('content_analysis', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Content Statistics:**")
        st.write(f"- Transcript length: {content_analysis.get('transcript_length', 0):,} characters")
        st.write(f"- Compression ratio: {content_analysis.get('summary_compression_ratio', 0):.2f}")
        st.write(f"- Tasks identified: {content_analysis.get('tasks_identified', 0)}")
        st.write(f"- Overall quality: {content_analysis.get('quality_score', 0):.2f}")
    
    with col2:
        # Processing info
        processing_info = results.get('processing_info', {})
        st.write("**Processing Information:**")
        st.write(f"- Content type: {processing_info.get('content_type', 'unknown')}")
        st.write(f"- Agent version: {processing_info.get('agent_version', 'unknown')}")
        
        timestamp = processing_info.get('processing_timestamp', 0)
        if timestamp:
            import datetime
            dt = datetime.datetime.fromtimestamp(timestamp)
            st.write(f"- Processed: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Recommendations
    recommendations = insights.get('recommendations', [])
    if recommendations:
        st.write("**Recommendations:**")
        for rec in recommendations:
            st.write(f"• {rec}")
    
    # Quality insights from quality assessor
    quality_insights = results.get('quality_scores', {}).get('insights', {})
    
    if quality_insights:
        col1, col2 = st.columns(2)
        
        with col1:
            strengths = quality_insights.get('strengths', [])
            if strengths:
                st.write("**Strengths:**")
                for strength in strengths:
                    st.success(f"✅ {strength}")
        
        with col2:
            weaknesses = quality_insights.get('weaknesses', [])
            if weaknesses:
                st.write("**Areas for Improvement:**")
                for weakness in weaknesses:
                    st.warning(f"⚠️ {weakness}")


def display_processing_history():
    """Display processing history."""
    st.header("📚 Processing History")
    
    if not st.session_state.processing_history:
        st.info("No processing history available.")
        return
    
    for i, result in enumerate(reversed(st.session_state.processing_history)):
        metadata = result.get('processing_metadata', {})
        file_name = metadata.get('file_name', f'Processing {i+1}')
        timestamp = metadata.get('processing_timestamp', 0)
        
        if timestamp:
            import datetime
            dt = datetime.datetime.fromtimestamp(timestamp)
            timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp_str = 'Unknown time'
        
        with st.expander(f"{file_name} - {timestamp_str}"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                transcript_length = len(result.get('transcript', ''))
                st.metric("Transcript", f"{transcript_length:,} chars")
            
            with col2:
                task_count = len(result.get('tasks', []))
                st.metric("Tasks", task_count)
            
            with col3:
                quality = result.get('quality_scores', {}).get('overall_quality', {}).get('score', 0)
                st.metric("Quality", f"{quality:.2f}")
            
            with col4:
                if st.button(f"View Details", key=f"view_{i}"):
                    st.session_state.processing_results = result


def main():
    """Main application function."""
    initialize_app()
    
    # Title and description
    st.title("🎤 Meeting Helper AI Agent")
    st.markdown("""
    Transform your audio or video recordings into concise summaries and actionable task lists using advanced AI.
    Perfect for lectures, meetings, study sessions, and more!
    """)
    
    # Sidebar configuration
    config = create_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        file_path, file_name = create_file_uploader()
        
        # Process button
        if file_path and st.button("🚀 Process file", type="primary"):
            with st.spinner("Processing file... This may take a few minutes."):
                # Run async processing
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(process_audio_file(file_path, file_name, config))
                
                if results:
                    st.session_state.processing_results = results
                    st.session_state.processing_history.append(results)
                    st.success("Processing completed successfully!")
                else:
                    st.error("Processing failed. Please check your file and try again.")
        
        # Display results
        if st.session_state.processing_results:
            display_results(st.session_state.processing_results)
    
    with col2:
        # Agent status
        st.subheader("🤖 Agent Status")
        agent_status = st.session_state.agent.get_agent_status()
        
        st.metric("Total Processed", agent_status['metrics']['total_processed'])
        st.metric("Success Rate", f"{agent_status['metrics']['success_rate']:.2f}")
        st.metric("Avg Time", f"{agent_status['metrics']['average_processing_time']:.1f}s")
        
        # Processing history
        if st.session_state.processing_history:
            st.subheader("📚 Recent Results")
            for result in st.session_state.processing_history[-3:]:  # Show last 3
                metadata = result.get('processing_metadata', {})
                file_name = metadata.get('file_name', 'Unknown file')
                quality = result.get('quality_scores', {}).get('overall_quality', {}).get('score', 0)
                
                with st.container():
                    st.write(f"**{file_name[:20]}...**")
                    st.write(f"Quality: {quality:.2f}")
                    st.write("---")


if __name__ == "__main__":
    main()
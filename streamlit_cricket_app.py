#!/usr/bin/env python3
"""
Streamlit Web Application for Cricket Analytics
Upload videos, analyze technique, and view results in browser
"""

import streamlit as st
import os
import tempfile
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from enhanced_cricket_analyzer import EnhancedCricketAnalyzer
import base64

# Page configuration
st.set_page_config(
    page_title="🏏 AthleteRise Cricket Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}
.metric-card {
    background: white;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #2a5298;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 10px 0;
}
.grade-a { color: #28a745; font-weight: bold; }
.grade-b { color: #ffc107; font-weight: bold; }
.grade-c { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def get_grade_class(grade):
    """Get CSS class for grade styling"""
    if grade.startswith('A'):
        return 'grade-a'
    elif grade.startswith('B'):
        return 'grade-b'
    else:
        return 'grade-c'

def create_radar_chart(scores_data):
    """Create radar chart for technique scores"""
    categories = list(scores_data.keys())
    scores = [scores_data[cat]["score"] for cat in categories]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Your Technique',
        line_color='rgb(42, 82, 152)',
        fillcolor='rgba(42, 82, 152, 0.3)'
    ))
    
    # Add ideal reference line
    ideal_scores = [8.5] * len(categories)  # Professional benchmark
    fig.add_trace(go.Scatterpolar(
        r=ideal_scores,
        theta=categories,
        fill=None,
        name='Professional Benchmark',
        line_color='rgb(255, 193, 7)',
        line_dash='dash'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        title="Technique Analysis Radar Chart",
        showlegend=True,
        height=500
    )
    
    return fig

def create_temporal_chart(metrics_data):
    """Create temporal analysis chart from metrics data"""
    if not metrics_data:
        return None
    
    # Extract time series data
    timestamps = [m["timestamp"] for m in metrics_data if "timestamp" in m]
    elbow_angles = [m.get("front_elbow_angle", 0) for m in metrics_data]
    spine_leans = [m.get("spine_lean", 0) for m in metrics_data]
    
    if not timestamps:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Elbow Angle Over Time', 'Spine Lean Over Time'),
        vertical_spacing=0.1
    )
    
    # Elbow angle
    fig.add_trace(
        go.Scatter(x=timestamps, y=elbow_angles, mode='lines+markers',
                  name='Elbow Angle', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Spine lean
    fig.add_trace(
        go.Scatter(x=timestamps, y=spine_leans, mode='lines+markers',
                  name='Spine Lean', line=dict(color='red', width=2)),
        row=2, col=1
    )
    
    fig.update_layout(height=600, title_text="Temporal Analysis")
    fig.update_xaxes(title_text="Time (seconds)")
    fig.update_yaxes(title_text="Angle (degrees)")
    
    return fig

def display_contact_analysis(contact_moments):
    """Display contact moment analysis"""
    if not contact_moments:
        st.warning("No contact moments detected in this video.")
        return
    
    st.subheader("⚡ Contact Moment Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Contacts Detected", len(contact_moments))
    
    with col2:
        avg_confidence = np.mean([c["confidence"] for c in contact_moments])
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    with col3:
        avg_velocity = np.mean([c.get("wrist_velocity", 0) for c in contact_moments])
        st.metric("Avg Wrist Velocity", f"{avg_velocity:.3f}")
    
    # Contact moments table
    if len(contact_moments) > 0:
        df = pd.DataFrame(contact_moments)
        df['timestamp'] = df['timestamp'].round(2)
        df['confidence'] = df['confidence'].round(3)
        st.dataframe(df, use_container_width=True)

def display_skill_assessment(skill_data):
    """Display skill level assessment"""
    st.subheader("🎯 Skill Level Assessment")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Skill grade with color coding
        grade = skill_data["grade"]
        score = skill_data["score"]
        
        if grade == "Advanced":
            st.success(f"**{grade}** - {score}/10")
        elif grade == "Intermediate":
            st.warning(f"**{grade}** - {score}/10")
        else:
            st.info(f"**{grade}** - {score}/10")
        
        st.write(f"*Confidence: {skill_data['confidence']:.1%}*")
    
    with col2:
        st.write("**Assessment Description:**")
        st.write(skill_data["description"])
        
        # Component scores
        if "component_scores" in skill_data:
            st.write("**Component Breakdown:**")
            components = skill_data["component_scores"]
            for component, score in components.items():
                st.write(f"• {component.title()}: {score}/10")

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏏 AthleteRise Cricket Analytics</h1>
        <p>Advanced AI-Powered Cricket Technique Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type:",
        ["Upload & Analyze", "Sample Analysis", "About"]
    )
    
    if page == "Upload & Analyze":
        st.header("📹 Video Upload & Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a cricket video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video showing cricket cover drive technique"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.success(f"File uploaded: {uploaded_file.name}")
            file_size = len(uploaded_file.getvalue()) / (1024*1024)
            st.info(f"File size: {file_size:.1f} MB")
            
            # Analysis options
            st.subheader("Analysis Options")
            
            col1, col2 = st.columns(2)
            with col1:
                include_advanced = st.checkbox("Include Advanced Features", value=True)
                generate_charts = st.checkbox("Generate Temporal Charts", value=True)
            
            with col2:
                contact_detection = st.checkbox("Contact Moment Detection", value=True)
                skill_assessment = st.checkbox("Skill Level Assessment", value=True)
            
            # Analyze button
            if st.button("🚀 Start Analysis", type="primary"):
                with st.spinner("Processing video... This may take a few minutes."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_path = tmp_file.name
                        
                        # Initialize analyzer
                        analyzer = EnhancedCricketAnalyzer()
                        
                        # Run analysis
                        results = analyzer.analyze_video(temp_path)
                        
                        # Clean up temp file
                        os.unlink(temp_path)
                        
                        # Display results
                        display_results(results)
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        st.error("Please check your video file and try again.")
    
    elif page == "Sample Analysis":
        st.header("📊 Sample Analysis Results")
        
        # Load sample results if available
        sample_file = "output/enhanced_evaluation.json"
        if os.path.exists(sample_file):
            with open(sample_file, 'r') as f:
                results = json.load(f)
            display_results(results)
        else:
            st.info("No sample analysis available. Please run an analysis first.")
            
            # Show demo data
            st.subheader("Demo Analysis Preview")
            demo_scores = {
                "footwork": {"score": 9.2, "grade": "A", "feedback": "Excellent foot positioning"},
                "head_position": {"score": 7.8, "grade": "B+", "feedback": "Good head stability"},
                "swing_control": {"score": 8.5, "grade": "A-", "feedback": "Smooth swing execution"},
                "balance": {"score": 8.0, "grade": "B+", "feedback": "Good balance throughout"},
                "follow_through": {"score": 8.7, "grade": "A-", "feedback": "Strong follow-through"}
            }
            
            # Display demo radar chart
            fig = create_radar_chart(demo_scores)
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # About page
        st.header("About AthleteRise Cricket Analytics")
        
        st.markdown("""
        ### 🏏 What is AthleteRise?
        
        AthleteRise is an advanced AI-powered cricket analytics system that provides comprehensive 
        biomechanical analysis of batting technique, specifically focusing on cover drive execution.
        
        ### 🚀 Key Features
        
        **Core Analysis:**
        - Real-time pose estimation using MediaPipe
        - Biomechanical metrics calculation
        - Live video annotations with feedback
        - Comprehensive scoring across 5 categories
        
        **Advanced Features:**
        - Automatic phase segmentation (Stance → Impact → Follow-through)
        - Contact moment detection using motion analysis
        - Temporal smoothness analysis with interactive charts
        - Reference technique comparison
        - Skill level assessment (Beginner/Intermediate/Advanced)
        - Personalized improvement recommendations
        
        ### 📊 Scoring Categories
        
        1. **Footwork** - Stability and positioning consistency
        2. **Head Position** - Head-over-knee alignment accuracy  
        3. **Swing Control** - Elbow angle consistency and timing
        4. **Balance** - Core stability throughout the shot
        5. **Follow-through** - Stroke completion quality
        
        ### 🎯 How It Works
        
        1. Upload your cricket video (MP4, AVI, MOV formats)
        2. AI analyzes every frame using computer vision
        3. Biomechanical metrics are calculated in real-time
        4. Advanced algorithms detect phases and contact moments
        5. Comprehensive report with scores and recommendations
        
        ### ⚡ Performance
        
        - Processes at 15-25 FPS on modern CPUs
        - Handles videos up to 1080p resolution
        - Professional-grade accuracy for technique analysis
        
        ### 🔬 Technology Stack
        
        - **Computer Vision**: MediaPipe, OpenCV
        - **Machine Learning**: TensorFlow Lite
        - **Data Analysis**: NumPy, SciPy, Pandas
        - **Visualization**: Plotly, Matplotlib
        - **Web Interface**: Streamlit
        
        ### 📈 Use Cases
        
        - **Players**: Improve technique with detailed feedback
        - **Coaches**: Analyze player performance objectively  
        - **Academies**: Standardize technique assessment
        - **Scouts**: Evaluate batting talent systematically
        
        ---
        
        *Built with ❤️ for cricket excellence*
        """)

def display_results(results):
    """Display comprehensive analysis results"""
    
    # Overview metrics
    st.header("📊 Analysis Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        overall_score = np.mean([score["score"] for score in results["traditional_scores"].values()])
        st.metric("Overall Score", f"{overall_score:.1f}/10")
    
    with col2:
        processing_fps = results["processing_info"]["average_processing_fps"]
        st.metric("Processing FPS", f"{processing_fps:.1f}")
    
    with col3:
        if "enhanced_analysis" in results:
            skill_grade = results["enhanced_analysis"]["skill_assessment"]["grade"]
            st.metric("Skill Level", skill_grade)
        else:
            st.metric("Analysis Type", "Standard")
    
    with col4:
        contacts = len(results.get("enhanced_analysis", {}).get("contact_moments", []))
        st.metric("Contact Moments", contacts)
    
    # Technique scores with radar chart
    st.header("🎯 Technique Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Radar chart
        radar_fig = create_radar_chart(results["traditional_scores"])
        st.plotly_chart(radar_fig, use_container_width=True)
    
    with col2:
        st.subheader("Score Breakdown")
        for category, data in results["traditional_scores"].items():
            score = data["score"]
            grade = data["grade"]
            
            # Score display with color coding
            grade_class = get_grade_class(grade)
            st.markdown(f"""
            <div class="metric-card">
                <strong>{category.replace('_', ' ').title()}</strong><br>
                <span class="{grade_class}">{score}/10 ({grade})</span><br>
                <small>{data["feedback"]}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced analysis if available
    if "enhanced_analysis" in results:
        enhanced = results["enhanced_analysis"]
        
        # Skill assessment
        if "skill_assessment" in enhanced:
            display_skill_assessment(enhanced["skill_assessment"])
        
        # Contact analysis
        if "contact_moments" in enhanced:
            display_contact_analysis(enhanced["contact_moments"])
        
        # Smoothness analysis
        if "smoothness_scores" in enhanced:
            st.header("📈 Smoothness Analysis")
            
            col1, col2, col3 = st.columns(3)
            smoothness = enhanced["smoothness_scores"]
            
            with col1:
                elbow_smooth = smoothness["elbow_smoothness"]
                st.metric("Elbow Smoothness", f"{elbow_smooth:.2f}")
                if elbow_smooth > 0.8:
                    st.success("Excellent")
                elif elbow_smooth > 0.6:
                    st.warning("Good")
                else:
                    st.error("Needs Work")
            
            with col2:
                spine_smooth = smoothness["spine_smoothness"]
                st.metric("Spine Smoothness", f"{spine_smooth:.2f}")
                if spine_smooth > 0.8:
                    st.success("Excellent")
                elif spine_smooth > 0.6:
                    st.warning("Good")
                else:
                    st.error("Needs Work")
            
            with col3:
                overall_smooth = smoothness["overall_smoothness"]
                st.metric("Overall Smoothness", f"{overall_smooth:.2f}")
                if overall_smooth > 0.8:
                    st.success("Excellent")
                elif overall_smooth > 0.6:
                    st.warning("Good")
                else:
                    st.error("Needs Work")
        
        # Reference comparison
        if "reference_comparison" in enhanced:
            st.header("🏆 Professional Comparison")
            comparison = enhanced["reference_comparison"]
            
            if "elbow_analysis" in comparison:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Elbow Technique")
                    elbow_data = comparison["elbow_analysis"]
                    st.write(f"Your Average: {elbow_data['average_angle']}°")
                    st.write(f"Professional Standard: {elbow_data['ideal_angle']}°")
                    st.write(f"Deviation: {elbow_data['deviation']}°")
                    
                    grade = elbow_data["grade"]
                    if grade == "excellent":
                        st.success("Excellent - Professional level!")
                    elif grade == "good":
                        st.warning("Good - Minor adjustments needed")
                    else:
                        st.info("Needs improvement - Focus on consistency")
                
                with col2:
                    if "spine_analysis" in comparison:
                        st.subheader("Balance & Posture")
                        spine_data = comparison["spine_analysis"]
                        st.write(f"Your Average: {spine_data['average_lean']}°")
                        st.write(f"Professional Standard: {spine_data['ideal_lean']}°")
                        st.write(f"Deviation: {spine_data['deviation']}°")
                        
                        grade = spine_data["grade"]
                        if grade == "excellent":
                            st.success("Excellent - Professional level!")
                        elif grade == "good":
                            st.warning("Good - Minor adjustments needed")
                        else:
                            st.info("Needs improvement - Work on balance")
    
    # Recommendations
    if "recommendations" in results:
        st.header("💡 Personalized Recommendations")
        
        recommendations = results["recommendations"]
        for i, rec in enumerate(recommendations, 1):
            st.write(f"**{i}.** {rec}")
    
    # Advanced insights
    if "advanced_insights" in results:
        st.header("🔬 Advanced Technical Insights")
        insights = results["advanced_insights"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Technique Consistency", f"{insights['technique_consistency']}/10")
            st.metric("Balance Stability", f"{insights['balance_stability']}/10")
        
        with col2:
            st.metric("Timing Precision", f"{insights['timing_precision']:.2f}")
            st.metric("Technical Grade", insights['overall_technical_grade'])
    
    
    # Video download section
    st.header("📁 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Annotated video download
        video_path = "output/enhanced_annotated_video.mp4"
        if not os.path.exists(video_path):
            video_path = "output/annotated_video.mp4"
        
        if os.path.exists(video_path):
            with open(video_path, "rb") as video_file:
                video_bytes = video_file.read()
            
            st.download_button(
                label="📹 Download Annotated Video",
                data=video_bytes,
                file_name="cricket_analysis_video.mp4",
                mime="video/mp4"
            )
        else:
            st.info("Annotated video not available")
    
    with col2:
        # JSON report download
        report_data = json.dumps(results, indent=2)
        st.download_button(
            label="📋 Download Detailed Report",
            data=report_data,
            file_name="cricket_analysis_report.json",
            mime="application/json"
        )
    
    # Processing information
    with st.expander("ℹ️ Processing Information"):
        processing = results["processing_info"]
        st.write(f"**Frames Processed:** {processing['frames_processed']}")
        st.write(f"**Processing Speed:** {processing['average_processing_fps']:.1f} FPS")
        st.write(f"**Total Time:** {processing['total_processing_time']:.1f} seconds")
        st.write(f"**Performance Grade:** {processing['performance_grade']}")
        
        # Video information
        if "video_info" in results:
            video_info = results["video_info"]
            st.write(f"**Video Dimensions:** {video_info['dimensions']}")
            st.write(f"**Video FPS:** {video_info['fps']}")
            st.write(f"**Duration:** {video_info['duration_seconds']:.1f} seconds")

if __name__ == "__main__":
    main()
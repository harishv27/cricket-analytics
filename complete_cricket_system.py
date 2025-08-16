#!/usr/bin/env python3
"""
Complete Cricket Analytics System
Integrates all features: Enhanced analysis, Streamlit app, and Report generation
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
    'mediapipe', 'numpy', 'matplotlib', 
    'streamlit', 'plotly', 'scipy', 'jinja2']

    # Handle OpenCV separately
    try:
        import cv2
    except ImportError:
        print("❌ Missing required package: opencv-python (or opencv-python-headless)")
        missing.append("opencv-python")


    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print(f"\n💡 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def setup_project_structure():
    """Ensure project structure exists"""
    directories = ['output', 'config', 'temp']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Project structure validated")

def run_basic_analysis(video_path):
    """Run basic cricket analysis"""
    print("🚀 Starting basic analysis...")
    
    try:
        from cover_drive_analysis_realtime import CricketAnalyzer
        
        analyzer = CricketAnalyzer()
        analyzer.validate_video_path(video_path)
        results = analyzer.analyze_video(video_path)
        
        print("✅ Basic analysis completed")
        return results
        
    except ImportError:
        print("❌ Basic analyzer not found. Please ensure cover_drive_analysis_realtime.py exists.")
        return None
    except Exception as e:
        print(f"❌ Basic analysis failed: {e}")
        return None

def run_enhanced_analysis(video_path):
    """Run enhanced cricket analysis with all bonus features"""
    print("🚀 Starting enhanced analysis...")
    
    try:
        from enhanced_cricket_analyzer import EnhancedCricketAnalyzer
        
        analyzer = EnhancedCricketAnalyzer()
        analyzer.validate_video_path(video_path)
        results = analyzer.analyze_video(video_path)
        
        print("✅ Enhanced analysis completed")
        return results
        
    except ImportError:
        print("❌ Enhanced analyzer not found. Please ensure enhanced_cricket_analyzer.py exists.")
        return None
    except Exception as e:
        print(f"❌ Enhanced analysis failed: {e}")
        return None

def generate_html_report():
    """Generate professional HTML report"""
    print("📄 Generating HTML report...")
    
    try:
        from report_generator import CricketReportGenerator
        
        # Load latest results
        results_file = "output/enhanced_evaluation.json"
        if not os.path.exists(results_file):
            results_file = "output/evaluation.json"
        
        if not os.path.exists(results_file):
            print("❌ No analysis results found for report generation")
            return None
        
        import json
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        generator = CricketReportGenerator()
        report_path = generator.generate_report(results)
        
        print(f"✅ HTML report generated: {report_path}")
        return report_path
        
    except ImportError:
        print("❌ Report generator not found. Please ensure report_generator.py exists.")
        return None
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return None

def launch_streamlit_app():
    """Launch the Streamlit web application"""
    print("🌐 Launching Streamlit web application...")
    
    app_file = "streamlit_cricket_app.py"
    if not os.path.exists(app_file):
        print(f"❌ Streamlit app not found: {app_file}")
        return False
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_file], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Streamlit app: {e}")
        return False
    except FileNotFoundError:
        print("❌ Streamlit not installed. Install with: pip install streamlit")
        return False

def display_system_info():
    """Display system capabilities and features"""
    print("🏏" + "="*60 + "🏏")
    print("   COMPLETE CRICKET ANALYTICS SYSTEM")
    print("   Advanced AI-Powered Technique Analysis")
    print("🏏" + "="*60 + "🏏")
    
    print("\n🎯 AVAILABLE FEATURES:")
    print("   ✅ Real-time pose estimation & biomechanical analysis")
    print("   ✅ Advanced phase detection (Stance → Impact → Follow-through)")
    print("   ✅ Contact moment detection using motion analysis")
    print("   ✅ Temporal smoothness analysis with interactive charts")
    print("   ✅ Professional technique comparison & benchmarking")
    print("   ✅ Skill level assessment (Beginner/Intermediate/Advanced)")
    print("   ✅ Personalized improvement recommendations")
    print("   ✅ Interactive web interface for video upload")
    print("   ✅ Professional HTML report generation")
    print("   ✅ Performance optimization (15-25+ FPS processing)")
    
    print("\n🚀 ANALYSIS MODES:")
    print("   1. Basic Analysis    - Core biomechanical assessment")
    print("   2. Enhanced Analysis - All advanced features included")
    print("   3. Web Interface     - Upload videos via browser")
    print("   4. Report Generation - Professional HTML reports")
    
    print("\n📊 SCORING CATEGORIES:")
    print("   • Footwork (Stability & positioning)")
    print("   • Head Position (Alignment accuracy)")
    print("   • Swing Control (Consistency & timing)")
    print("   • Balance (Core stability)")
    print("   • Follow-through (Completion quality)")
    
    print("\n⚡ PERFORMANCE TARGETS:")
    print("   • Processing Speed: 15-25 FPS on modern CPUs")
    print("   • Analysis Accuracy: Professional-grade biomechanics")
    print("   • Report Quality: Publication-ready HTML documents")

def main():
    """Main system controller"""
    parser = argparse.ArgumentParser(
        description="Complete Cricket Analytics System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'mode',
        choices=['info', 'basic', 'enhanced', 'web', 'report', 'all'],
        help='Analysis mode to run'
    )
    
    parser.add_argument(
        '--video', '-v',
        type=str,
        default='input_video.mp4',
        help='Path to cricket video file'
    )
    
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Skip dependency check'
    )
    
    args = parser.parse_args()
    
    # Display system information
    if args.mode == 'info':
        display_system_info()
        return
    
    # Check dependencies
    if not args.skip_deps:
        print("🔍 Checking dependencies...")
        if not check_dependencies():
            return
        print("✅ All dependencies satisfied")
    
    # Setup project structure
    setup_project_structure()
    
    # Execute based on mode
    if args.mode == 'basic':
        results = run_basic_analysis(args.video)
        if results:
            print(f"\n🎯 Analysis complete! Overall score: {results.get('summary', {}).get('total_score', 'N/A')}/10")
    
    elif args.mode == 'enhanced':
        results = run_enhanced_analysis(args.video)
        if results:
            skill_grade = results.get('enhanced_analysis', {}).get('skill_assessment', {}).get('grade', 'N/A')
            overall_score = results.get('summary', {}).get('total_score', 'N/A')
            print(f"\n🎯 Enhanced analysis complete!")
            print(f"   Overall Score: {overall_score}/10")
            print(f"   Skill Level: {skill_grade}")
    
    elif args.mode == 'web':
        launch_streamlit_app()
    
    elif args.mode == 'report':
        generate_html_report()
    
    elif args.mode == 'all':
        print("🚀 Running complete analysis pipeline...")
        
        # Step 1: Enhanced analysis
        results = run_enhanced_analysis(args.video)
        if not results:
            print("❌ Analysis failed, cannot continue pipeline")
            return
        
        # Step 2: Generate report
        report_path = generate_html_report()
        
        # Step 3: Display summary
        print("\n" + "="*50)
        print("🎉 COMPLETE ANALYSIS PIPELINE FINISHED!")
        print("="*50)
        
        skill_grade = results.get('enhanced_analysis', {}).get('skill_assessment', {}).get('grade', 'N/A')
        overall_score = results.get('summary', {}).get('total_score', 'N/A')
        contacts = len(results.get('enhanced_analysis', {}).get('contact_moments', []))
        
        print(f"📊 Results Summary:")
        print(f"   Overall Score: {overall_score}/10")
        print(f"   Skill Level: {skill_grade}")
        print(f"   Contact Moments: {contacts}")
        
        print(f"\n📁 Generated Files:")
        print(f"   📹 Annotated Video: output/enhanced_annotated_video.mp4")
        print(f"   📋 Detailed Analysis: output/enhanced_evaluation.json")
        if report_path:
            print(f"   📄 HTML Report: {report_path}")
        
        print(f"\n💡 Next Steps:")
        print(f"   • Review the annotated video for visual feedback")
        print(f"   • Open the HTML report in your browser")
        print(f"   • Use 'python complete_cricket_system.py web' for interactive analysis")

if __name__ == "__main__":
    main()
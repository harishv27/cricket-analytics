#!/usr/bin/env python3
"""
Professional HTML/PDF Report Generator for Cricket Analytics
Generates comprehensive reports with charts, insights, and recommendations
"""

import json
import os
from datetime import datetime
import base64
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template
import plotly.graph_objects as go
import plotly.io as pio

class CricketReportGenerator:
    def __init__(self):
        self.report_template = self.get_html_template()
    
    def generate_report(self, results_data, output_path="output/cricket_analysis_report.html"):
        """Generate comprehensive HTML report"""
        
        print("📄 Generating professional report...")
        
        # Prepare data for template
        report_data = self.prepare_report_data(results_data)
        
        # Generate charts
        charts = self.generate_charts(results_data)
        report_data.update(charts)
        
        # Render template
        template = Template(self.report_template)
        html_content = template.render(**report_data)
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Report generated: {output_path}")
        return output_path
    
    def prepare_report_data(self, results):
        """Prepare data for the report template"""
        
        # Basic information
        report_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video_info": results.get("video_info", {}),
            "processing_info": results.get("processing_info", {}),
        }
        
        # Scores processing
        traditional_scores = results.get("traditional_scores", {})
        overall_score = np.mean([score["score"] for score in traditional_scores.values()]) if traditional_scores else 0
        
        report_data.update({
            "overall_score": round(overall_score, 1),
            "scores": traditional_scores,
            "score_items": list(traditional_scores.items())
        })
        
        # Enhanced analysis
        if "enhanced_analysis" in results:
            enhanced = results["enhanced_analysis"]
            report_data.update({
                "has_enhanced": True,
                "skill_assessment": enhanced.get("skill_assessment", {}),
                "contact_moments": enhanced.get("contact_moments", []),
                "smoothness_scores": enhanced.get("smoothness_scores", {}),
                "reference_comparison": enhanced.get("reference_comparison", {})
            })
        else:
            report_data["has_enhanced"] = False
        
        # Advanced insights
        if "advanced_insights" in results:
            report_data["advanced_insights"] = results["advanced_insights"]
        
        # Recommendations
        report_data["recommendations"] = results.get("recommendations", [])
        
        # Grade mapping for display
        report_data["grade_colors"] = {
            "A+": "#28a745", "A": "#28a745", "A-": "#28a745",
            "B+": "#ffc107", "B": "#ffc107", "B-": "#ffc107", 
            "C+": "#dc3545", "C": "#dc3545", "C-": "#dc3545"
        }
        
        return report_data
    
    def generate_charts(self, results):
        """Generate charts for the report"""
        charts = {}
        
        # Radar chart for technique scores
        if "traditional_scores" in results:
            radar_chart = self.create_radar_chart(results["traditional_scores"])
            charts["radar_chart"] = radar_chart
        
        # Temporal analysis chart
        if "enhanced_analysis" in results and "contact_moments" in results["enhanced_analysis"]:
            temporal_chart = self.create_temporal_summary_chart(results)
            charts["temporal_chart"] = temporal_chart
        
        # Performance chart
        performance_chart = self.create_performance_chart(results)
        charts["performance_chart"] = performance_chart
        
        return charts
    
    def create_radar_chart(self, scores_data):
        """Create radar chart as base64 encoded image"""
        categories = list(scores_data.keys())
        scores = [scores_data[cat]["score"] for cat in categories]
        
        # Create plotly figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=[cat.replace('_', ' ').title() for cat in categories],
            fill='toself',
            name='Your Technique',
            line_color='rgb(42, 82, 152)',
            fillcolor='rgba(42, 82, 152, 0.3)'
        ))
        
        # Professional benchmark
        ideal_scores = [8.5] * len(categories)
        fig.add_trace(go.Scatterpolar(
            r=ideal_scores,
            theta=[cat.replace('_', ' ').title() for cat in categories],
            fill=None,
            name='Professional Benchmark',
            line_color='rgb(255, 193, 7)',
            line_dash='dash'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickmode='linear',
                    tick0=0,
                    dtick=2
                )
            ),
            showlegend=True,
            width=500,
            height=400,
            margin=dict(l=80, r=80, t=80, b=80)
        )
        
        # Convert to base64
        img_bytes = pio.to_image(fig, format="png", width=500, height=400)
        img_base64 = base64.b64encode(img_bytes).decode()
        
        return f"data:image/png;base64,{img_base64}"
    
    def create_temporal_summary_chart(self, results):
        """Create a summary temporal chart"""
        if "enhanced_analysis" not in results:
            return None
        
        enhanced = results["enhanced_analysis"]
        contact_moments = enhanced.get("contact_moments", [])
        
        if not contact_moments:
            return None
        
        # Create a simple timeline chart
        fig = go.Figure()
        
        timestamps = [c["timestamp"] for c in contact_moments]
        confidences = [c["confidence"] for c in contact_moments]
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=confidences,
            mode='markers+lines',
            name='Contact Confidence',
            marker=dict(size=10, color='red'),
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Contact Moments Timeline",
            xaxis_title="Time (seconds)",
            yaxis_title="Detection Confidence",
            width=600,
            height=300,
            margin=dict(l=60, r=60, t=60, b=60)
        )
        
        # Convert to base64
        img_bytes = pio.to_image(fig, format="png", width=600, height=300)
        img_base64 = base64.b64encode(img_bytes).decode()
        
        return f"data:image/png;base64,{img_base64}"
    
    def create_performance_chart(self, results):
        """Create performance metrics chart"""
        processing_info = results.get("processing_info", {})
        
        # Create a gauge chart for processing performance
        fps = processing_info.get("average_processing_fps", 0)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = fps,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Processing Performance (FPS)"},
            delta = {'reference': 10, 'increasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 30]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 10], 'color': "lightgray"},
                    {'range': [10, 20], 'color': "yellow"},
                    {'range': [20, 30], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 10
                }
            }
        ))
        
        fig.update_layout(
            width=400,
            height=300,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # Convert to base64
        img_bytes = pio.to_image(fig, format="png", width=400, height=300)
        img_base64 = base64.b64encode(img_bytes).decode()
        
        return f"data:image/png;base64,{img_base64}"
    
    def get_html_template(self):
        """HTML template for the cricket analysis report"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cricket Technique Analysis Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .meta-info {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #667eea;
        }
        
        .section {
            margin-bottom: 40px;
        }
        
        .section h2 {
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 20px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .score-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .score-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 5px solid #667eea;
        }
        
        .score-card h3 {
            font-size: 1.3em;
            margin-bottom: 10px;
            color: #333;
        }
        
        .score-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .score-grade {
            padding: 5px 10px;
            border-radius: 15px;
            color: white;
            font-weight: bold;
            display: inline-block;
            margin: 10px 0;
        }
        
        .feedback {
            font-style: italic;
            color: #666;
            margin-top: 10px;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin: 30px 0;
        }
        
        .chart-container {
            text-align: center;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .chart-container h3 {
            margin-bottom: 20px;
            color: #667eea;
        }
        
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }
        
        .insights-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        
        .insight-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        
        .insight-card .value {
            font-size: 2em;
            font-weight: bold;
            display: block;
        }
        
        .insight-card .label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .recommendations {
            background-color: #e8f5e8;
            border: 1px solid #4caf50;
            border-radius: 8px;
            padding: 20px;
        }
        
        .recommendations h3 {
            color: #4caf50;
            margin-bottom: 15px;
        }
        
        .recommendations ul {
            list-style-type: none;
        }
        
        .recommendations li {
            padding: 8px 0;
            border-bottom: 1px solid #ddd;
        }
        
        .recommendations li:before {
            content: "💡 ";
            margin-right: 10px;
        }
        
        .contact-moments {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .performance-stats {
            display: flex;
            justify-content: space-around;
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        .stat {
            text-align: center;
        }
        
        .stat .value {
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat .label {
            font-size: 0.9em;
            color: #666;
        }
        
        .footer {
            text-align: center;
            padding: 30px;
            color: #666;
            border-top: 1px solid #ddd;
            margin-top: 50px;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .score-grid {
                grid-template-columns: 1fr;
            }
            
            .charts-grid {
                grid-template-columns: 1fr;
            }
            
            .performance-stats {
                flex-direction: column;
                gap: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🏏 Cricket Technique Analysis</h1>
            <div class="subtitle">Professional Biomechanical Assessment Report</div>
            <div style="margin-top: 20px; font-size: 1em;">
                Generated on {{ timestamp }}
            </div>
        </div>
        
        <!-- Meta Information -->
        <div class="meta-info">
            <h3>📹 Video Information</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 15px;">
                <div><strong>Dimensions:</strong> {{ video_info.dimensions or 'N/A' }}</div>
                <div><strong>Frame Rate:</strong> {{ video_info.fps or 'N/A' }} FPS</div>
                <div><strong>Duration:</strong> {{ "%.1f"|format(video_info.duration_seconds or 0) }} seconds</div>
                <div><strong>Total Frames:</strong> {{ video_info.total_frames or 'N/A' }}</div>
            </div>
        </div>
        
        <!-- Overall Score -->
        <div class="section">
            <h2>🎯 Overall Assessment</h2>
            <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px;">
                <div style="font-size: 3em; font-weight: bold;">{{ overall_score }}/10</div>
                <div style="font-size: 1.2em; margin-top: 10px;">
                    {% if overall_score >= 9 %}Elite Technique
                    {% elif overall_score >= 8 %}Excellent Technique  
                    {% elif overall_score >= 7 %}Good Technique
                    {% elif overall_score >= 6 %}Developing Technique
                    {% else %}Needs Improvement{% endif %}
                </div>
            </div>
        </div>
        
        <!-- Technique Scores -->
        <div class="section">
            <h2>📊 Technique Breakdown</h2>
            <div class="score-grid">
                {% for category, data in score_items %}
                <div class="score-card">
                    <h3>{{ category.replace('_', ' ').title() }}</h3>
                    <div class="score-value">{{ data.score }}/10</div>
                    <div class="score-grade" style="background-color: {{ grade_colors[data.grade] }};">
                        Grade {{ data.grade }}
                    </div>
                    <div class="feedback">{{ data.feedback }}</div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Charts -->
        {% if radar_chart %}
        <div class="section">
            <h2>📈 Visual Analysis</h2>
            <div class="charts-grid">
                <div class="chart-container">
                    <h3>Technique Radar Chart</h3>
                    <img src="{{ radar_chart }}" alt="Technique Radar Chart">
                </div>
                {% if temporal_chart %}
                <div class="chart-container">
                    <h3>Contact Moments Analysis</h3>
                    <img src="{{ temporal_chart }}" alt="Temporal Analysis">
                </div>
                {% endif %}
                {% if performance_chart %}
                <div class="chart-container">
                    <h3>Processing Performance</h3>
                    <img src="{{ performance_chart }}" alt="Performance Chart">
                </div>
                {% endif %}
            </div>
        </div>
        {% endif %}
        
        <!-- Enhanced Analysis -->
        {% if has_enhanced %}
        <div class="section">
            <h2>🔬 Advanced Analysis</h2>
            
            {% if skill_assessment %}
            <div style="background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h3>🎯 Skill Level Assessment</h3>
                <div style="font-size: 2em; margin: 10px 0;">{{ skill_assessment.grade }}</div>
                <div>Score: {{ skill_assessment.score }}/10</div>
                <div style="margin-top: 10px; opacity: 0.9;">{{ skill_assessment.description }}</div>
            </div>
            {% endif %}
            
            {% if contact_moments %}
            <div class="contact-moments">
                <h3>⚡ Contact Moment Analysis</h3>
                <div class="performance-stats">
                    <div class="stat">
                        <div class="value">{{ contact_moments|length }}</div>
                        <div class="label">Contacts Detected</div>
                    </div>
                    <div class="stat">
                        <div class="value">{{ "%.2f"|format(contact_moments|map(attribute='confidence')|list|sum / contact_moments|length) if contact_moments else 0 }}</div>
                        <div class="label">Avg Confidence</div>
                    </div>
                    <div class="stat">
                        <div class="value">{{ "%.3f"|format(contact_moments|map(attribute='wrist_velocity')|list|sum / contact_moments|length) if contact_moments else 0 }}</div>
                        <div class="label">Avg Wrist Velocity</div>
                    </div>
                </div>
            </div>
            {% endif %}
            
            {% if smoothness_scores %}
            <div class="insights-grid">
                <div class="insight-card">
                    <span class="value">{{ "%.2f"|format(smoothness_scores.elbow_smoothness) }}</span>
                    <span class="label">Elbow Smoothness</span>
                </div>
                <div class="insight-card">
                    <span class="value">{{ "%.2f"|format(smoothness_scores.spine_smoothness) }}</span>
                    <span class="label">Spine Smoothness</span>
                </div>
                <div class="insight-card">
                    <span class="value">{{ "%.2f"|format(smoothness_scores.overall_smoothness) }}</span>
                    <span class="label">Overall Smoothness</span>
                </div>
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        <!-- Professional Comparison -->
        {% if reference_comparison %}
        <div class="section">
            <h2>🏆 Professional Comparison</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                {% if reference_comparison.elbow_analysis %}
                <div class="score-card">
                    <h3>Elbow Technique</h3>
                    <div>Your Average: {{ reference_comparison.elbow_analysis.average_angle }}°</div>
                    <div>Professional: {{ reference_comparison.elbow_analysis.ideal_angle }}°</div>
                    <div>Deviation: {{ reference_comparison.elbow_analysis.deviation }}°</div>
                    <div class="score-grade" style="background-color: 
                        {% if reference_comparison.elbow_analysis.grade == 'excellent' %}#28a745
                        {% elif reference_comparison.elbow_analysis.grade == 'good' %}#ffc107
                        {% else %}#dc3545{% endif %};">
                        {{ reference_comparison.elbow_analysis.grade.title() }}
                    </div>
                </div>
                {% endif %}
                
                {% if reference_comparison.spine_analysis %}
                <div class="score-card">
                    <h3>Balance & Posture</h3>
                    <div>Your Average: {{ reference_comparison.spine_analysis.average_lean }}°</div>
                    <div>Professional: {{ reference_comparison.spine_analysis.ideal_lean }}°</div>
                    <div>Deviation: {{ reference_comparison.spine_analysis.deviation }}°</div>
                    <div class="score-grade" style="background-color: 
                        {% if reference_comparison.spine_analysis.grade == 'excellent' %}#28a745
                        {% elif reference_comparison.spine_analysis.grade == 'good' %}#ffc107
                        {% else %}#dc3545{% endif %};">
                        {{ reference_comparison.spine_analysis.grade.title() }}
                    </div>
                </div>
                {% endif %}
            </div>
        </div>
        {% endif %}
        
        <!-- Recommendations -->
        {% if recommendations %}
        <div class="section">
            <div class="recommendations">
                <h3>💡 Personalized Recommendations</h3>
                <ul>
                    {% for rec in recommendations %}
                    <li>{{ rec }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
        {% endif %}
        
        <!-- Processing Performance -->
        <div class="section">
            <h2>⚡ Processing Performance</h2>
            <div class="performance-stats">
                <div class="stat">
                    <div class="value">{{ "%.1f"|format(processing_info.average_processing_fps or 0) }}</div>
                    <div class="label">Processing FPS</div>
                </div>
                <div class="stat">
                    <div class="value">{{ "%.1f"|format(processing_info.total_processing_time or 0) }}</div>
                    <div class="label">Total Time (seconds)</div>
                </div>
                <div class="stat">
                    <div class="value">{{ processing_info.frames_processed or 0 }}</div>
                    <div class="label">Frames Processed</div>
                </div>
                <div class="stat">
                    <div class="value">{{ processing_info.performance_grade or 'N/A' }}</div>
                    <div class="label">Performance Grade</div>
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <div style="font-size: 1.1em; margin-bottom: 10px;">
                🏏 Generated by AthleteRise Cricket Analytics
            </div>
            <div style="color: #999; font-size: 0.9em;">
                Advanced AI-Powered Cricket Technique Analysis System<br>
                Professional Biomechanical Assessment & Performance Optimization
            </div>
        </div>
    </div>
</body>
</html>"""

def main():
    """Test the report generator with sample data"""
    
    # Check if analysis results exist
    results_file = "output/enhanced_evaluation.json"
    if not os.path.exists(results_file):
        print("❌ No analysis results found. Please run the cricket analysis first.")
        return
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Generate report
    generator = CricketReportGenerator()
    report_path = generator.generate_report(results)
    
    print(f"📄 Professional report generated: {report_path}")
    print("🌐 Open the HTML file in your browser to view the report.")

if __name__ == "__main__":
    main()
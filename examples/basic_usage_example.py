"""
Basic Usage Example: Behavioral Change Detection Pipeline
========================================================

This example demonstrates how to use the learning analytics pipeline
for analyzing behavioral changes in educational gaming sessions.

This script shows:
1. Data loading and preprocessing
2. Change point detection using multiple methods
3. Visualization and interpretation of results
4. Generating summary reports

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from change_detection.zscore_method import BehavioralChangeDetector
from visualization.session_plots import BehavioralVisualization
from data_processing.synthetic_data import generate_synthetic_session_data

def main():
    """
    Main execution function demonstrating the complete pipeline.
    """
    print("🎮 Learning Analytics Pipeline - Basic Usage Example")
    print("=" * 60)
    
    # Step 1: Generate or Load Sample Data
    print("\n📊 Step 1: Loading Sample Data")
    print("-" * 30)
    
    # For this example, we'll generate synthetic data
    # In practice, you would load your actual educational game data
    session_data = generate_synthetic_session_data(
        session_id=1,
        duration_minutes=15,
        n_events=200,
        players=['player1', 'player2']
    )
    
    print(f"✅ Loaded session data: {len(session_data)} events")
    print(f"   Session duration: {session_data['session_duration_mins'].iloc[0]:.1f} minutes")
    print(f"   Players: {list(session_data['player_inference_total'].unique())}")
    print(f"   Event types: {len(session_data['eventKey'].unique())} unique events")
    
    # Step 2: Initialize the Behavioral Change Detector
    print("\n🔍 Step 2: Initializing Change Detection Framework")
    print("-" * 50)
    
    # Load reference statistics (if available)
    # For this example, we'll use the detector without reference stats
    detector = BehavioralChangeDetector()
    
    print("✅ Detector initialized")
    print("   Method: Z-score based change point detection")
    print("   Analysis types: Reference-based + Session-specific")
    
    # Step 3: Run Complete Analysis
    print("\n⚡ Step 3: Running Behavioral Analysis")
    print("-" * 40)
    
    # Analyze the session
    results = detector.analyze_session(
        session_data=session_data,
        session_id=1,
        threshold=2.0  # Z-score threshold for change point detection
    )
    
    print("✅ Analysis complete!")
    print(f"   Reference method: {results['summary']['total_change_points_ref']} change points")
    print(f"   Session method: {results['summary']['total_change_points_sess']} change points")
    
    # Step 4: Examine Results in Detail
    print("\n📈 Step 4: Examining Results")
    print("-" * 30)
    
    ref_change_points = results['reference_analysis']['change_points']
    sess_change_points = results['session_analysis']['change_points']
    
    if not ref_change_points.empty:
        print("\n🔍 Reference Method Change Points:")
        for _, cp in ref_change_points.iterrows():
            time_pct = cp['position_in_session_pct'] * 100
            player = cp['player_inference_total_cleaned']
            intensity = cp['change_intensity']
            events = cp.get('key_events', [])
            
            print(f"   • Player {player} at {time_pct:.1f}% through session")
            print(f"     Intensity: {intensity:.2f}, Key events: {events[:3]}")
    
    if not sess_change_points.empty:
        print("\n🔍 Session-Specific Method Change Points:")
        for _, cp in sess_change_points.iterrows():
            time_pct = cp['position_in_session_pct'] * 100
            player = cp['player_inference_total_cleaned']
            intensity = cp['change_intensity']
            events = cp.get('key_events', [])
            
            print(f"   • Player {player} at {time_pct:.1f}% through session")
            print(f"     Intensity: {intensity:.2f}, Key events: {events[:3]}")
    
    # Step 5: Create Visualizations
    print("\n🎨 Step 5: Generating Visualizations")
    print("-" * 40)
    
    # Initialize visualization toolkit
    visualizer = BehavioralVisualization()
    
    # Create main session analysis plot
    session_fig = visualizer.plot_session_analysis(
        ref_zscores_df=results['reference_analysis']['zscores'],
        sess_zscores_df=results['session_analysis']['zscores'],
        change_points_ref=ref_change_points,
        change_points_sess=sess_change_points,
        session_id=1,
        save_path='output/example_session_analysis.png'
    )
    
    print("✅ Session analysis plot created: output/example_session_analysis.png")
    
    # Create change point distribution analysis
    if not ref_change_points.empty:
        dist_fig = visualizer.plot_change_point_distribution(
            change_points_df=ref_change_points,
            method_name="Reference Method",
            save_path='output/example_change_distribution.png'
        )
        print("✅ Distribution analysis plot created: output/example_change_distribution.png")
    
    # Step 6: Generate Interactive Dashboard
    print("\n🌐 Step 6: Creating Interactive Dashboard")
    print("-" * 45)
    
    interactive_fig = visualizer.create_interactive_dashboard(
        ref_zscores_df=results['reference_analysis']['zscores'],
        sess_zscores_df=results['session_analysis']['zscores'],
        change_points_ref=ref_change_points,
        change_points_sess=sess_change_points
    )
    
    # Save interactive plot
    os.makedirs('output', exist_ok=True)
    interactive_fig.write_html('output/example_interactive_dashboard.html')
    print("✅ Interactive dashboard created: output/example_interactive_dashboard.html")
    
    # Step 7: Export Results
    print("\n💾 Step 7: Exporting Results")
    print("-" * 30)
    
    # Save analysis results
    detector.save_analysis_results(results, 'output')
    
    # Create summary statistics
    summary_stats = create_summary_statistics(results)
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv('output/example_summary_statistics.csv', index=False)
    
    print("✅ Results exported to output/ directory")
    print("   • Change points CSV files")
    print("   • Summary statistics")
    print("   • Visualization files")
    
    # Step 8: Interpretation Guide
    print("\n💡 Step 8: Interpreting Results")
    print("-" * 35)
    
    print_interpretation_guide(results)
    
    print("\n🎉 Analysis Complete!")
    print("=" * 60)
    print("Check the 'output/' directory for all generated files.")
    print("Open 'example_interactive_dashboard.html' in your browser for interactive exploration.")


def create_summary_statistics(results):
    """
    Create summary statistics from analysis results.
    
    Parameters:
    -----------
    results : dict
        Analysis results from BehavioralChangeDetector
        
    Returns:
    --------
    dict
        Summary statistics
    """
    ref_cp = results['reference_analysis']['change_points']
    sess_cp = results['session_analysis']['change_points']
    
    stats = {
        'session_id': results['session_id'],
        'session_duration_minutes': results['summary']['session_duration'],
        'total_events': results['summary']['total_events'],
        
        # Reference method stats
        'ref_total_change_points': len(ref_cp),
        'ref_avg_intensity': ref_cp['change_intensity'].mean() if not ref_cp.empty else 0,
        'ref_max_intensity': ref_cp['change_intensity'].max() if not ref_cp.empty else 0,
        'ref_early_changes': len(ref_cp[ref_cp['position_in_session_pct'] <= 0.33]),
        'ref_mid_changes': len(ref_cp[(ref_cp['position_in_session_pct'] > 0.33) & 
                                     (ref_cp['position_in_session_pct'] <= 0.67)]),
        'ref_late_changes': len(ref_cp[ref_cp['position_in_session_pct'] > 0.67]),
        
        # Session method stats
        'sess_total_change_points': len(sess_cp),
        'sess_avg_intensity': sess_cp['change_intensity'].mean() if not sess_cp.empty else 0,
        'sess_max_intensity': sess_cp['change_intensity'].max() if not sess_cp.empty else 0,
        'sess_early_changes': len(sess_cp[sess_cp['position_in_session_pct'] <= 0.33]),
        'sess_mid_changes': len(sess_cp[(sess_cp['position_in_session_pct'] > 0.33) & 
                                       (sess_cp['position_in_session_pct'] <= 0.67)]),
        'sess_late_changes': len(sess_cp[sess_cp['position_in_session_pct'] > 0.67]),
    }
    
    return stats


def print_interpretation_guide(results):
    """
    Print interpretation guide for the analysis results.
    
    Parameters:
    -----------
    results : dict
        Analysis results from BehavioralChangeDetector
    """
    ref_cp = results['reference_analysis']['change_points']
    sess_cp = results['session_analysis']['change_points']
    
    print("🔍 How to Interpret These Results:")
    print()
    
    print("📊 Change Point Detection Methods:")
    print("   • Reference Method: Compares behavior to a large reference dataset")
    print("   • Session Method: Identifies unusual behavior within this specific session")
    print()
    
    print("📈 Change Intensity:")
    print("   • Low (0-2): Normal behavioral variation")
    print("   • Medium (2-4): Notable behavioral shift")
    print("   • High (4+): Significant strategy or engagement change")
    print()
    
    print("⏰ Timing Interpretation:")
    print("   • Early changes (0-33%): Initial strategy exploration")
    print("   • Mid changes (33-67%): Learning adaptation or strategy shifts")
    print("   • Late changes (67-100%): Final optimization or engagement changes")
    print()
    
    # Provide specific interpretation for this session
    total_ref = len(ref_cp)
    total_sess = len(sess_cp)
    
    if total_ref == 0 and total_sess == 0:
        print("📝 Session Interpretation:")
        print("   ✅ Stable behavioral patterns throughout the session")
        print("   ✅ Consistent engagement and strategy")
        print("   ⚠️  May indicate either optimal learning flow OR lack of adaptation")
    
    elif total_ref > 5 or total_sess > 5:
        print("📝 Session Interpretation:")
        print("   ⚠️  High number of behavioral changes detected")
        print("   ⚠️  May indicate exploration, confusion, or learning struggles")
        print("   💡 Consider examining specific events that triggered changes")
    
    else:
        print("📝 Session Interpretation:")
        print("   ✅ Moderate behavioral adaptation")
        print("   ✅ Evidence of strategic learning and adjustment")
        print("   💡 Normal learning progression patterns")
    
    print()
    print("🎯 Key Questions to Ask:")
    print("   • Do change points align with known learning milestones?")
    print("   • Are behavioral shifts associated with success or struggle?")
    print("   • How do the two methods complement each other?")
    print("   • What specific events drove the most significant changes?")


class AdvancedAnalysisExample:
    """
    Advanced usage example showing multi-session analysis and method comparison.
    """
    
    def __init__(self):
        self.detector = BehavioralChangeDetector()
        self.visualizer = BehavioralVisualization()
        
    def run_multi_session_analysis(self, n_sessions=5):
        """
        Demonstrate analysis across multiple sessions.
        
        Parameters:
        -----------
        n_sessions : int
            Number of sessions to analyze
        """
        print(f"\n🔄 Advanced Example: Multi-Session Analysis ({n_sessions} sessions)")
        print("=" * 70)
        
        all_results = []
        all_summaries = []
        
        for session_id in range(1, n_sessions + 1):
            print(f"\n📊 Analyzing Session {session_id}...")
            
            # Generate varied session data
            session_data = generate_synthetic_session_data(
                session_id=session_id,
                duration_minutes=np.random.uniform(8, 25),  # Varied duration
                n_events=np.random.randint(150, 300),       # Varied complexity
                players=['player1', 'player2'],
                difficulty_level=np.random.choice(['easy', 'medium', 'hard'])
            )
            
            # Analyze session
            results = self.detector.analyze_session(
                session_data=session_data,
                session_id=session_id,
                threshold=2.0
            )
            
            all_results.append(results)
            all_summaries.append(create_summary_statistics(results))
            
            print(f"   ✅ Session {session_id}: {results['summary']['total_change_points_ref']} change points")
        
        # Create comparative analysis
        self.create_cross_session_analysis(all_results, all_summaries)
        
        return all_results, all_summaries
    
    def create_cross_session_analysis(self, all_results, all_summaries):
        """
        Create comparative analysis across multiple sessions.
        """
        print("\n📈 Cross-Session Analysis")
        print("-" * 30)
        
        # Combine all summaries
        summary_df = pd.DataFrame(all_summaries)
        
        # Session-level insights
        print("\n📊 Session-Level Patterns:")
        avg_changes_ref = summary_df['ref_total_change_points'].mean()
        avg_changes_sess = summary_df['sess_total_change_points'].mean()
        
        print(f"   • Average change points (Reference): {avg_changes_ref:.1f}")
        print(f"   • Average change points (Session): {avg_changes_sess:.1f}")
        print(f"   • Most active session: Session {summary_df.loc[summary_df['ref_total_change_points'].idxmax(), 'session_id']}")
        print(f"   • Most stable session: Session {summary_df.loc[summary_df['ref_total_change_points'].idxmin(), 'session_id']}")
        
        # Temporal patterns
        early_changes = summary_df['ref_early_changes'].sum()
        mid_changes = summary_df['ref_mid_changes'].sum()
        late_changes = summary_df['ref_late_changes'].sum()
        total_changes = early_changes + mid_changes + late_changes
        
        if total_changes > 0:
            print(f"\n⏰ Temporal Distribution of Changes:")
            print(f"   • Early session: {early_changes/total_changes*100:.1f}%")
            print(f"   • Mid session: {mid_changes/total_changes*100:.1f}%")
            print(f"   • Late session: {late_changes/total_changes*100:.1f}%")
        
        # Save comprehensive summary
        summary_df.to_csv('output/multi_session_summary.csv', index=False)
        print(f"\n💾 Multi-session summary saved to: output/multi_session_summary.csv")
        
        # Create summary visualization
        self.create_summary_visualization(summary_df)
    
    def create_summary_visualization(self, summary_df):
        """
        Create summary visualizations for multi-session analysis.
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Change points by session
        axes[0, 0].bar(summary_df['session_id'], summary_df['ref_total_change_points'], 
                      alpha=0.7, label='Reference Method')
        axes[0, 0].bar(summary_df['session_id'], summary_df['sess_total_change_points'], 
                      alpha=0.7, label='Session Method')
        axes[0, 0].set_title('Change Points by Session')
        axes[0, 0].set_xlabel('Session ID')
        axes[0, 0].set_ylabel('Number of Change Points')
        axes[0, 0].legend()
        
        # Plot 2: Intensity distributions
        axes[0, 1].scatter(summary_df['session_duration_minutes'], 
                          summary_df['ref_avg_intensity'], alpha=0.7)
        axes[0, 1].set_title('Average Intensity vs Session Duration')
        axes[0, 1].set_xlabel('Session Duration (minutes)')
        axes[0, 1].set_ylabel('Average Change Intensity')
        
        # Plot 3: Temporal distribution
        temporal_data = summary_df[['ref_early_changes', 'ref_mid_changes', 'ref_late_changes']].sum()
        axes[1, 0].pie(temporal_data, labels=['Early', 'Mid', 'Late'], autopct='%1.1f%%')
        axes[1, 0].set_title('Distribution of Changes by Session Phase')
        
        # Plot 4: Method comparison
        method_comparison = pd.DataFrame({
            'Reference': summary_df['ref_total_change_points'],
            'Session': summary_df['sess_total_change_points']
        })
        method_comparison.plot(kind='box', ax=axes[1, 1])
        axes[1, 1].set_title('Method Comparison: Change Point Detection')
        axes[1, 1].set_ylabel('Number of Change Points')
        
        plt.tight_layout()
        plt.savefig('output/multi_session_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Multi-session visualization saved to: output/multi_session_analysis.png")


def demonstrate_method_comparison():
    """
    Demonstrate comparison between different change detection methods.
    """
    print("\n🔬 Method Comparison Example")
    print("=" * 40)
    
    # Generate a complex session for comparison
    session_data = generate_synthetic_session_data(
        session_id=999,
        duration_minutes=20,
        n_events=300,
        players=['player1', 'player2'],
        complexity='high'
    )
    
    print("📊 Generating sample session with high complexity...")
    print(f"   • Duration: 20 minutes")
    print(f"   • Events: 300 total events")
    print(f"   • Players: 2 active players")
    
    # Initialize different detection methods
    methods = {
        'Z-Score (Reference)': BehavioralChangeDetector(),
        'Z-Score (Session-only)': BehavioralChangeDetector()
    }
    
    # Note: In a full implementation, you would also include:
    # - HMM-based detection
    # - Clustering-based detection
    # - Bayesian change point detection
    
    results_by_method = {}
    
    for method_name, detector in methods.items():
        print(f"\n🔍 Running {method_name} analysis...")
        
        results = detector.analyze_session(
            session_data=session_data,
            session_id=999,
            threshold=2.0
        )
        
        if 'Reference' in method_name:
            change_points = results['reference_analysis']['change_points']
        else:
            change_points = results['session_analysis']['change_points']
        
        results_by_method[method_name] = change_points
        
        print(f"   ✅ Detected {len(change_points)} change points")
    
    # Compare results
    print("\n📊 Method Comparison Results:")
    print("-" * 30)
    
    for method, cp_df in results_by_method.items():
        avg_intensity = cp_df['change_intensity'].mean() if not cp_df.empty else 0
        max_intensity = cp_df['change_intensity'].max() if not cp_df.empty else 0
        
        print(f"   {method}:")
        print(f"     • Total change points: {len(cp_df)}")
        print(f"     • Average intensity: {avg_intensity:.2f}")
        print(f"     • Maximum intensity: {max_intensity:.2f}")
    
    # Create comparison visualization
    if results_by_method:
        visualizer = BehavioralVisualization()
        comparison_fig = visualizer.plot_method_comparison(
            results_by_method,
            save_path='output/method_comparison.png'
        )
        print("\n📊 Method comparison plot saved to: output/method_comparison.png")


if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    
    # Run basic example
    main()
    
    # Run advanced examples
    print("\n" + "="*80)
    print("🚀 ADVANCED EXAMPLES")
    print("="*80)
    
    # Multi-session analysis
    advanced_example = AdvancedAnalysisExample()
    all_results, all_summaries = advanced_example.run_multi_session_analysis(n_sessions=5)
    
    # Method comparison
    demonstrate_method_comparison()
    
    print("\n🎉 All Examples Complete!")
    print("="*80)
    print("📁 Generated Files in output/ directory:")
    print("   • Individual session analyses")
    print("   • Multi-session comparative analysis")
    print("   • Method comparison visualizations")
    print("   • Interactive dashboards")
    print("   • Summary statistics (CSV)")
    print("\n💡 Next Steps:")
    print("   • Open HTML files in your browser for interactive exploration")
    print("   • Examine CSV files for detailed numerical results")
    print("   • Modify parameters to test different scenarios")
    print("   • Integrate with your own educational game data")
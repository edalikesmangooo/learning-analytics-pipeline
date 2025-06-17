"""
Advanced Visualization Module for Behavioral Change Analysis
===========================================================

This module provides comprehensive visualization tools for analyzing behavioral 
change points and learning patterns in educational gaming sessions.

Key Features:
- Multi-method change point comparison plots
- Interactive session timeline visualization
- Statistical summary dashboards
- Publication-ready figure generation

Author: [Your Name]
Institution: [Your Institution]
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from typing import Dict, List, Optional, Union, Tuple
import warnings

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


class BehavioralVisualization:
    """
    Comprehensive visualization toolkit for behavioral change analysis.
    
    Provides both static (matplotlib/seaborn) and interactive (plotly) 
    visualizations for research and presentation purposes.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Initialize visualization settings.
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Default figure size for matplotlib plots
        dpi : int
            Resolution for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'reference': '#3498db',      # Blue
            'session': '#2ecc71',        # Green  
            'threshold': '#e74c3c',      # Red
            'intensity': '#95a5a6',      # Gray
            'player1': '#9b59b6',        # Purple
            'player2': '#f39c12'         # Orange
        }
        
    def plot_session_analysis(self,
                             ref_zscores_df: pd.DataFrame,
                             sess_zscores_df: pd.DataFrame,
                             change_points_ref: pd.DataFrame,
                             change_points_sess: pd.DataFrame,
                             session_id: Union[str, int],
                             save_path: Optional[str] = None,
                             time_col: str = 'position_in_session_pct',
                             players: List[str] = ['player1', 'player2'],
                             threshold: float = 2.0) -> plt.Figure:
        """
        Create comprehensive session analysis plot showing change intensity 
        and detected change points for both analysis methods.

        Parameters:
        -----------
        ref_zscores_df : pd.DataFrame
            Reference-based Z-score analysis results
        sess_zscores_df : pd.DataFrame  
            Session-specific Z-score analysis results
        change_points_ref : pd.DataFrame
            Detected change points from reference method
        change_points_sess : pd.DataFrame
            Detected change points from session method
        session_id : Union[str, int]
            Session identifier for plot title
        save_path : str, optional
            Path to save the figure
        time_col : str
            Column name for time axis
        players : List[str]
            List of players to analyze
        threshold : float
            Threshold value for change point detection

        Returns:
        --------
        plt.Figure
            Generated matplotlib figure
        """
        fig, axs = plt.subplots(len(players), 1, figsize=self.figsize, sharex=True)
        if len(players) == 1:
            axs = [axs]

        for i, player in enumerate(players):
            ax = axs[i]

            # Filter data for this player
            ref_data = ref_zscores_df[
                (ref_zscores_df['session_id'] == session_id) & 
                (ref_zscores_df['player_inference_total_cleaned'] == player)
            ]
            sess_data = sess_zscores_df[
                (sess_zscores_df['session_id'] == session_id) & 
                (sess_zscores_df['player_inference_total_cleaned'] == player)
            ]

            cp_ref = change_points_ref[
                (change_points_ref['session_id'] == session_id) & 
                (change_points_ref['player_inference_total_cleaned'] == player)
            ]
            cp_sess = change_points_sess[
                (change_points_sess['session_id'] == session_id) & 
                (change_points_sess['player_inference_total_cleaned'] == player)
            ]

            # Plot change intensity timeline
            if not ref_data.empty:
                ax.plot(ref_data[time_col], ref_data['change_intensity'], 
                       color=self.colors['intensity'], linewidth=2, 
                       label='Change Intensity', alpha=0.7)

            # Overlay change points
            if not cp_ref.empty:
                ax.scatter(cp_ref[time_col], cp_ref['change_intensity'], 
                          color=self.colors['reference'], marker='o', s=80,
                          label='Reference Method', alpha=0.8, edgecolors='white')
                
            if not cp_sess.empty:
                ax.scatter(cp_sess[time_col], cp_sess['change_intensity'], 
                          color=self.colors['session'], marker='s', s=80,
                          label='Session Method', alpha=0.8, edgecolors='white')

            # Add threshold line
            ax.axhline(y=threshold, linestyle='--', color=self.colors['threshold'], 
                      linewidth=2, label=f'Threshold ({threshold})', alpha=0.8)

            # Annotate key events for change points
            self._annotate_change_points(ax, cp_ref, time_col, self.colors['reference'], offset=10)
            self._annotate_change_points(ax, cp_sess, time_col, self.colors['session'], offset=-15)

            # Styling
            ax.set_ylabel('Change Intensity', fontsize=12)
            ax.set_title(f'Player {player.capitalize()} — Session {session_id}', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

        # Final styling
        axs[-1].set_xlabel('Session Progress (Normalized Time)', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig

    def _annotate_change_points(self, ax, change_points_df, time_col, color, offset=10):
        """Helper method to annotate change points with key events."""
        for _, row in change_points_df.iterrows():
            if 'key_events' in row and row['key_events']:
                events_text = ', '.join(row['key_events'][:2])
                if len(row['key_events']) > 2:
                    events_text += '...'
                    
                ax.annotate(events_text,
                           (row[time_col], row['change_intensity']),
                           textcoords='offset points', xytext=(0, offset), 
                           ha='center', fontsize=8, color=color,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   alpha=0.8, edgecolor=color))

    def plot_change_point_distribution(self,
                                     change_points_df: pd.DataFrame,
                                     method_name: str = "Analysis Method",
                                     time_col: str = 'position_in_session_pct',
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of change points across session timeline.

        Parameters:
        -----------
        change_points_df : pd.DataFrame
            Change points data
        method_name : str
            Name for plot title
        time_col : str
            Column for temporal analysis
        save_path : str, optional
            Path to save figure

        Returns:
        --------
        plt.Figure
            Generated figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[2, 1])

        # Histogram of change point timing
        ax1.hist(change_points_df[time_col], bins=20, color=self.colors['reference'], 
                alpha=0.7, edgecolor='black', linewidth=1)
        ax1.set_title(f'{method_name}: Change Points Across Session Timeline', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Change Points', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Add phase markers
        ax1.axvline(x=0.33, color='red', linestyle='--', alpha=0.6, label='Early/Mid')
        ax1.axvline(x=0.67, color='red', linestyle='--', alpha=0.6, label='Mid/Late')
        ax1.legend()

        # Box plot by session phase
        change_points_df['phase'] = pd.cut(change_points_df[time_col], 
                                         bins=[0, 0.33, 0.67, 1.0], 
                                         labels=['Early', 'Middle', 'Late'])
        
        sns.boxplot(data=change_points_df, x='phase', y='change_intensity', ax=ax2)
        ax2.set_title('Change Intensity by Session Phase', fontsize=12)
        ax2.set_xlabel('Session Phase', fontsize=12)
        ax2.set_ylabel('Change Intensity', fontsize=12)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig

    def create_interactive_dashboard(self,
                                   ref_zscores_df: pd.DataFrame,
                                   sess_zscores_df: pd.DataFrame,
                                   change_points_ref: pd.DataFrame,
                                   change_points_sess: pd.DataFrame) -> go.Figure:
        """
        Create interactive Plotly dashboard for exploration.

        Parameters:
        -----------
        ref_zscores_df, sess_zscores_df : pd.DataFrame
            Z-score analysis results
        change_points_ref, change_points_sess : pd.DataFrame
            Change point detection results

        Returns:
        --------
        go.Figure
            Interactive Plotly figure
        """
        sessions = ref_zscores_df['session_id'].unique()
        
        fig = make_subplots(
            rows=len(sessions), cols=2,
            subplot_titles=[f'Session {sid} - Player 1' for sid in sessions] + 
                          [f'Session {sid} - Player 2' for sid in sessions],
            vertical_spacing=0.05
        )

        for i, session_id in enumerate(sessions):
            for j, player in enumerate(['player1', 'player2']):
                row, col = i + 1, j + 1
                
                # Filter data
                ref_data = ref_zscores_df[
                    (ref_zscores_df['session_id'] == session_id) & 
                    (ref_zscores_df['player_inference_total_cleaned'] == player)
                ]
                
                cp_ref = change_points_ref[
                    (change_points_ref['session_id'] == session_id) & 
                    (change_points_ref['player_inference_total_cleaned'] == player)
                ]

                # Add change intensity line
                if not ref_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=ref_data['position_in_session_pct'],
                            y=ref_data['change_intensity'],
                            mode='lines',
                            name=f'S{session_id}-{player}',
                            line=dict(color=self.colors['intensity'], width=2),
                            showlegend=False
                        ),
                        row=row, col=col
                    )

                # Add change points
                if not cp_ref.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=cp_ref['position_in_session_pct'],
                            y=cp_ref['change_intensity'],
                            mode='markers',
                            marker=dict(
                                color=self.colors['reference'],
                                size=10,
                                symbol='circle'
                            ),
                            text=[', '.join(events[:2]) for events in cp_ref['key_events']],
                            hovertemplate='<b>Change Point</b><br>' +
                                        'Time: %{x:.2f}<br>' +
                                        'Intensity: %{y:.2f}<br>' +
                                        'Events: %{text}<extra></extra>',
                            showlegend=False
                        ),
                        row=row, col=col
                    )

        fig.update_layout(
            height=300 * len(sessions),
            title_text="Interactive Behavioral Change Analysis Dashboard",
            title_x=0.5
        )
        
        return fig

    def plot_method_comparison(self,
                             results_dict: Dict[str, pd.DataFrame],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare results from different change detection methods.

        Parameters:
        -----------
        results_dict : Dict[str, pd.DataFrame]
            Dictionary mapping method names to their change point results
        save_path : str, optional
            Path to save figure

        Returns:
        --------
        plt.Figure
            Comparison plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Method comparison statistics
        method_stats = {}
        for method, cp_df in results_dict.items():
            method_stats[method] = {
                'total_change_points': len(cp_df),
                'avg_intensity': cp_df['change_intensity'].mean(),
                'max_intensity': cp_df['change_intensity'].max(),
                'sessions_with_changes': cp_df['session_id'].nunique()
            }
        
        stats_df = pd.DataFrame(method_stats).T
        
        # Plot 1: Total change points by method
        axes[0, 0].bar(stats_df.index, stats_df['total_change_points'], 
                      color=list(self.colors.values())[:len(stats_df)])
        axes[0, 0].set_title('Total Change Points by Method')
        axes[0, 0].set_ylabel('Count')
        
        # Plot 2: Average intensity by method
        axes[0, 1].bar(stats_df.index, stats_df['avg_intensity'],
                      color=list(self.colors.values())[:len(stats_df)])
        axes[0, 1].set_title('Average Change Intensity by Method')
        axes[0, 1].set_ylabel('Intensity')
        
        # Plot 3: Change point timing distribution
        for i, (method, cp_df) in enumerate(results_dict.items()):
            axes[1, 0].hist(cp_df['position_in_session_pct'], 
                           alpha=0.6, label=method, bins=15)
        axes[1, 0].set_title('Change Point Timing Distribution')
        axes[1, 0].set_xlabel('Session Progress')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Plot 4: Intensity distribution by method
        intensity_data = []
        method_labels = []
        for method, cp_df in results_dict.items():
            intensity_data.append(cp_df['change_intensity'])
            method_labels.extend([method] * len(cp_df))
        
        combined_df = pd.DataFrame({
            'intensity': pd.concat(intensity_data),
            'method': method_labels
        })
        
        sns.boxplot(data=combined_df, x='method', y='intensity', ax=axes[1, 1])
        axes[1, 1].set_title('Change Intensity Distribution by Method')
        axes[1, 1].set_ylabel('Change Intensity')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig

    def create_summary_report(self,
                            analysis_results: Dict,
                            save_dir: str) -> None:
        """
        Generate comprehensive visual summary report.

        Parameters:
        -----------
        analysis_results : Dict
            Complete analysis results from BehavioralChangeDetector
        save_dir : str
            Directory to save report figures
        """
        os.makedirs(save_dir, exist_ok=True)
        session_id = analysis_results['session_id']
        
        # 1. Main session analysis plot
        main_fig = self.plot_session_analysis(
            analysis_results['reference_analysis']['zscores'],
            analysis_results['session_analysis']['zscores'],
            analysis_results['reference_analysis']['change_points'],
            analysis_results['session_analysis']['change_points'],
            session_id
        )
        main_fig.savefig(f"{save_dir}/session_{session_id}_main_analysis.png", 
                        dpi=self.dpi, bbox_inches='tight')
        plt.close(main_fig)
        
        # 2. Change point distribution plots
        for method, label in [('reference_analysis', 'Reference'), 
                             ('session_analysis', 'Session')]:
            dist_fig = self.plot_change_point_distribution(
                analysis_results[method]['change_points'],
                f"{label} Method"
            )
            dist_fig.savefig(f"{save_dir}/session_{session_id}_{method}_distribution.png",
                           dpi=self.dpi, bbox_inches='tight')
            plt.close(dist_fig)
        
        # 3. Interactive dashboard (save as HTML)
        interactive_fig = self.create_interactive_dashboard(
            analysis_results['reference_analysis']['zscores'],
            analysis_results['session_analysis']['zscores'],
            analysis_results['reference_analysis']['change_points'],
            analysis_results['session_analysis']['change_points']
        )
        interactive_fig.write_html(f"{save_dir}/session_{session_id}_interactive.html")
        
        print(f"Summary report generated in {save_dir}/")


# Utility functions for batch processing
def plot_all_sessions(analysis_results_list: List[Dict],
                     save_dir: str,
                     visualizer: Optional[BehavioralVisualization] = None) -> None:
    """
    Generate plots for multiple sessions in batch.
    
    Parameters:
    -----------
    analysis_results_list : List[Dict]
        List of analysis results from multiple sessions
    save_dir : str
        Directory to save all plots
    visualizer : BehavioralVisualization, optional
        Custom visualizer instance
    """
    if visualizer is None:
        visualizer = BehavioralVisualization()
    
    os.makedirs(save_dir, exist_ok=True)
    
    for results in analysis_results_list:
        session_id = results['session_id']
        session_dir = os.path.join(save_dir, f"session_{session_id}")
        visualizer.create_summary_report(results, session_dir)
    
    print(f"Batch processing complete. Results saved to {save_dir}/")


def create_comparative_analysis(results_dict: Dict[str, List[Dict]],
                              save_path: str) -> None:
    """
    Create comparative analysis across different methods.
    
    Parameters:
    -----------
    results_dict : Dict[str, List[Dict]]
        Dictionary mapping method names to their results lists
    save_path : str
        Path to save comparative analysis
    """
    visualizer = BehavioralVisualization(figsize=(16, 12))
    
    # Combine change points from all methods
    combined_results = {}
    for method_name, results_list in results_dict.items():
        all_change_points = []
        for result in results_list:
            # Assuming we want reference analysis for comparison
            cp_df = result['reference_analysis']['change_points'].copy()
            cp_df['method'] = method_name
            all_change_points.append(cp_df)
        
        if all_change_points:
            combined_results[method_name] = pd.concat(all_change_points, ignore_index=True)
    
    # Generate comparison plot
    if combined_results:
        comparison_fig = visualizer.plot_method_comparison(combined_results)
        comparison_fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(comparison_fig)
        print(f"Comparative analysis saved to {save_path}")


# Example usage
if __name__ == "__main__":
    print("Behavioral Visualization Module")
    print("==============================")
    print("This module provides comprehensive visualization tools for")
    print("behavioral change analysis in educational gaming sessions.")
    print("\nKey features:")
    print("- Session timeline analysis plots")
    print("- Interactive dashboards")
    print("- Method comparison visualizations")
    print("- Automated report generation")
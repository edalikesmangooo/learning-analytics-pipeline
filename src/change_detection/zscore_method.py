"""
Behavioral Change Detection Using Z-Score Analysis
=================================================

This module implements a novel framework for detecting behavioral change points 
in educational game sessions using statistical z-score analysis.

Key Innovations:
- Dual z-score calculation (reference-based vs session-specific)
- Action vs. state separation methodology
- Multi-method change point detection

Author: [Your Name]
Institution: [Your Institution]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import json
import os
from datetime import datetime


class BehavioralChangeDetector:
    """
    A comprehensive framework for detecting behavioral change points in 
    educational gaming sessions using z-score analysis.
    
    This class implements the methodology described in:
    "Strategic Focus vs. Behavioral Scattering: A Novel Framework for 
    Learning Analytics in Educational Games"
    """
    
    def __init__(self, reference_stats: Optional[Dict] = None):
        """
        Initialize the behavioral change detector.
        
        Parameters:
        -----------
        reference_stats : dict, optional
            Dictionary containing 'means' and 'stds' for reference-based analysis
        """
        self.reference_means = reference_stats.get('means', {}) if reference_stats else {}
        self.reference_stds = reference_stats.get('stds', {}) if reference_stats else {}
        self.event_columns = []
        
    def clean_player_labels(self, 
                           df: pd.DataFrame, 
                           original_player_col: str = 'player_inference_total',
                           cleaned_player_col: str = 'player_inference_total_cleaned') -> pd.DataFrame:
        """
        Clean player inference labels, replacing 'none' with 'both' to capture
        collaborative gameplay moments.

        Parameters:
        -----------
        df : pd.DataFrame
            Session log data
        original_player_col : str
            Column indicating original player label
        cleaned_player_col : str
            Column name to save cleaned labels

        Returns:
        --------
        pd.DataFrame
            DataFrame with additional cleaned player column
        """
        df = df.copy()
        df[cleaned_player_col] = df[original_player_col].replace('none', 'both')
        return df

    def create_time_windows(self, 
                           session_df: pd.DataFrame, 
                           min_windows: int = 5, 
                           max_windows: int = 20) -> pd.DataFrame:
        """
        Create adaptive time windows based on session duration.
        
        For sessions ≥5 minutes: Uses 1-minute real-time windows
        For shorter sessions: Uses normalized percentage-based bins
        
        Parameters:
        -----------
        session_df : pd.DataFrame
            Session data with temporal information
        min_windows : int
            Minimum number of windows for short sessions
        max_windows : int
            Maximum number of windows for long sessions
            
        Returns:
        --------
        pd.DataFrame
            Data with time_window and position_in_session_pct columns
        """
        df = session_df.copy()

        # Normalize time for this session (0.0 to 1.0)
        df['position_in_session_pct'] = (
            (df['log_epoch'] - df['session_start_epoch']) /
            (df['session_end_epoch'] - df['session_start_epoch'])
        )

        # Determine duration and windowing strategy
        session_duration = df['session_duration_mins'].iloc[0]

        if session_duration >= 5:
            # Use 1-minute real-time windows for longer sessions
            df['minutes_since_start'] = (df['log_epoch'] - df['session_start_epoch']) / 60
            df['time_window'] = df['minutes_since_start'].astype(int)
        else:
            # Use normalized bins for shorter sessions
            bin_count = min_windows if session_duration < 2 else min(max_windows, 8)
            df['time_window'] = pd.cut(
                df['position_in_session_pct'].clip(0, 0.9999),
                bins=bin_count,
                labels=False
            )

        return df

    def create_event_count_matrix(self, 
                                 df: pd.DataFrame, 
                                 event_col: str = 'eventKey', 
                                 player_col: str = 'player_inference_total_cleaned', 
                                 time_col: str = 'time_window',
                                 all_event_keys: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create a time-windowed event count matrix per session and player.
        
        This method handles collaborative actions by duplicating 'both' player
        entries for both individual players, ensuring comprehensive coverage
        of collaborative gameplay moments.

        Parameters:
        -----------
        df : pd.DataFrame
            Preprocessed session log data with time_window column
        event_col : str
            Column name for event keys
        player_col : str
            Column identifying player
        time_col : str
            Column indicating time bin
        all_event_keys : List[str], optional
            Full list of possible event keys for consistent matrix shape

        Returns:
        --------
        pd.DataFrame
            Aggregated event count matrix with consistent column structure
        """
        # Handle collaborative actions: duplicate 'both' rows for both players
        df_both = df[df[player_col] == 'both'].copy()
        df_both[player_col] = 'player1'
        df_alt = df[df[player_col] == 'both'].copy()
        df_alt[player_col] = 'player2'

        # Combine with player-specific rows
        df_cleaned = pd.concat([
            df[df[player_col].isin(['player1', 'player2'])],
            df_both,
            df_alt
        ], ignore_index=True)

        # Group and pivot to create event count matrix
        grouped = df_cleaned.groupby(['session_id', player_col, time_col, event_col]).size()
        event_counts = grouped.unstack(fill_value=0).reset_index()

        # Ensure consistent column structure across sessions
        if all_event_keys is not None:
            current_events = set(event_counts.columns) - set(['session_id', player_col, time_col])
            missing_events = set(all_event_keys) - current_events
            
            for event in missing_events:
                event_counts[event] = 0
            
            # Reorder columns: metadata first, then event columns
            key_cols = ['session_id', player_col, time_col]
            event_cols = [e for e in all_event_keys]
            event_counts = event_counts[key_cols + event_cols]

        self.event_columns = [col for col in event_counts.columns 
                             if col not in ['session_id', player_col, time_col]]
        
        return event_counts

    def calculate_zscores_dual(self, 
                              event_counts_df: pd.DataFrame,
                              exclude_cols: List[str] = None,
                              min_data_points: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate Z-scores using both reference-based and session-specific methods.
        
        This dual approach allows comparison between:
        1. How unusual behavior is compared to a large reference dataset
        2. How unusual behavior is within the specific session context

        Parameters:
        -----------
        event_counts_df : pd.DataFrame
            Time-windowed event counts for analysis
        exclude_cols : List[str], optional
            Metadata columns not to Z-score
        min_data_points : int
            Minimum number of time windows required for session-based Z-score

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            (reference_zscores, session_zscores) - Two DataFrames with z-score columns
        """
        if exclude_cols is None:
            exclude_cols = ['session_id', 'player_inference_total_cleaned', 'time_window']
            
        ref_df = event_counts_df.copy()
        sess_df = event_counts_df.copy()
        
        event_cols = [col for col in event_counts_df.columns if col not in exclude_cols]
        enough_data = len(event_counts_df) >= min_data_points

        for col in event_cols:
            # 1. Reference-based Z-score (global comparison)
            if (self.reference_means and self.reference_stds and 
                col in self.reference_means and col in self.reference_stds):
                mu_ref = self.reference_means[col]
                sigma_ref = self.reference_stds[col] if self.reference_stds[col] > 0 else 1e-6
                ref_df[f'z_{col}'] = (event_counts_df[col] - mu_ref) / sigma_ref
            else:
                ref_df[f'z_{col}'] = np.nan

            # 2. Session-specific Z-score (internal comparison)
            if enough_data:
                mu_sess = event_counts_df[col].mean()
                sigma_sess = event_counts_df[col].std()
                if sigma_sess > 0:
                    sess_df[f'z_{col}'] = (event_counts_df[col] - mu_sess) / sigma_sess
                else:
                    sess_df[f'z_{col}'] = 0
            else:
                sess_df[f'z_{col}'] = np.nan

        return ref_df, sess_df

    def calculate_change_intensity(self, 
                                  zscore_df: pd.DataFrame, 
                                  method: str = 'max') -> pd.DataFrame:
        """
        Calculate change intensity from Z-score columns in each time window.
        
        Change intensity represents how unusual a player's behavior is in 
        that specific time window across all tracked events.

        Parameters:
        -----------
        zscore_df : pd.DataFrame
            Z-score-enhanced DataFrame
        method : str
            'max' for maximum |z| value, 'sum' for total |z| across events

        Returns:
        --------
        pd.DataFrame
            DataFrame with new 'change_intensity' column
        """
        df = zscore_df.copy()
        z_cols = [col for col in df.columns if col.startswith('z_')]

        if method == 'max':
            df['change_intensity'] = df[z_cols].abs().max(axis=1)
        elif method == 'sum':
            df['change_intensity'] = df[z_cols].abs().sum(axis=1)
        else:
            raise ValueError("Invalid method. Use 'max' or 'sum'.")

        return df

    def detect_change_points(self, 
                            zscore_df: pd.DataFrame, 
                            threshold: float = 2.0) -> pd.DataFrame:
        """
        Identify time windows where change intensity exceeds the threshold.
        
        Change points represent moments of significant behavioral shift,
        indicating potential learning transitions, strategy changes, or
        engagement pattern modifications.

        Parameters:
        -----------
        zscore_df : pd.DataFrame
            Must contain 'change_intensity' column
        threshold : float
            Value above which we consider a time window to be a behavioral shift

        Returns:
        --------
        pd.DataFrame
            Subset of original rows representing detected change points
        """
        return zscore_df[zscore_df['change_intensity'] > threshold].copy()

    def extract_key_events(self, 
                          change_points_df: pd.DataFrame, 
                          zscore_threshold: float = 2.0) -> pd.DataFrame:
        """
        Annotate each change point with events whose Z-scores exceeded the threshold.
        
        This provides interpretable insight into what specific behaviors
        contributed to each detected change point.

        Parameters:
        -----------
        change_points_df : pd.DataFrame
            Subset of Z-score data where change_intensity > threshold
        zscore_threshold : float
            Threshold to consider an event as a key contributor

        Returns:
        --------
        pd.DataFrame
            DataFrame with new 'key_events' column containing contributing events
        """
        df = change_points_df.copy()
        z_cols = [col for col in df.columns if col.startswith('z_')]
        
        # For each row, find which z_event values exceeded the threshold
        df['key_events'] = df.apply(
            lambda row: [
                col.replace('z_', '') for col in z_cols 
                if abs(row[col]) > zscore_threshold
            ],
            axis=1
        )
        
        return df

    def analyze_session(self, 
                       session_data: pd.DataFrame,
                       session_id: Union[str, int],
                       threshold: float = 2.0) -> Dict:
        """
        Complete analysis pipeline for a single session.
        
        Parameters:
        -----------
        session_data : pd.DataFrame
            Raw session log data
        session_id : Union[str, int]
            Identifier for the session
        threshold : float
            Change point detection threshold
            
        Returns:
        --------
        Dict
            Complete analysis results including change points and visualizations
        """
        # Step 1: Data preparation
        cleaned_data = self.clean_player_labels(session_data)
        windowed_data = self.create_time_windows(cleaned_data)
        
        # Step 2: Feature extraction
        event_matrix = self.create_event_count_matrix(windowed_data)
        
        # Step 3: Z-score calculation
        ref_zscores, sess_zscores = self.calculate_zscores_dual(event_matrix)
        
        # Step 4: Change intensity calculation
        ref_zscores = self.calculate_change_intensity(ref_zscores)
        sess_zscores = self.calculate_change_intensity(sess_zscores)
        
        # Step 5: Change point detection
        ref_change_points = self.detect_change_points(ref_zscores, threshold)
        sess_change_points = self.detect_change_points(sess_zscores, threshold)
        
        # Step 6: Key event extraction
        ref_change_points = self.extract_key_events(ref_change_points)
        sess_change_points = self.extract_key_events(sess_change_points)
        
        return {
            'session_id': session_id,
            'raw_data': windowed_data,
            'event_matrix': event_matrix,
            'reference_analysis': {
                'zscores': ref_zscores,
                'change_points': ref_change_points
            },
            'session_analysis': {
                'zscores': sess_zscores,
                'change_points': sess_change_points
            },
            'summary': {
                'total_change_points_ref': len(ref_change_points),
                'total_change_points_sess': len(sess_change_points),
                'session_duration': windowed_data['session_duration_mins'].iloc[0],
                'total_events': len(windowed_data)
            }
        }

    @staticmethod
    def load_reference_statistics(filepath: str) -> Dict:
        """
        Load reference statistics from JSON file.
        
        Parameters:
        -----------
        filepath : str
            Path to JSON file containing reference means and stds
            
        Returns:
        --------
        Dict
            Dictionary with 'means' and 'stds' keys
        """
        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def save_analysis_results(results: Dict, output_dir: str) -> None:
        """
        Save analysis results to CSV files.
        
        Parameters:
        -----------
        results : Dict
            Results from analyze_session method
        output_dir : str
            Directory to save output files
        """
        os.makedirs(output_dir, exist_ok=True)
        session_id = results['session_id']
        
        # Save change points
        ref_cp = results['reference_analysis']['change_points']
        sess_cp = results['session_analysis']['change_points']
        
        ref_cp.to_csv(f"{output_dir}/session_{session_id}_ref_change_points.csv", index=False)
        sess_cp.to_csv(f"{output_dir}/session_{session_id}_sess_change_points.csv", index=False)
        
        # Save summary
        summary_df = pd.DataFrame([results['summary']])
        summary_df.to_csv(f"{output_dir}/session_{session_id}_summary.csv", index=False)


# Example usage and demonstration
if __name__ == "__main__":
    # This would typically be replaced with actual data loading
    print("Behavioral Change Detection Framework")
    print("====================================")
    print("This module provides a comprehensive framework for detecting")
    print("behavioral change points in educational gaming sessions.")
    print("\nKey features:")
    print("- Dual z-score analysis (reference vs session-specific)")
    print("- Adaptive time windowing")
    print("- Collaborative action handling")
    print("- Interpretable change point extraction")
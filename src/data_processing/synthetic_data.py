"""
Synthetic Data Generator for Learning Analytics Pipeline Testing
===============================================================

This module generates realistic synthetic educational game session data
that mimics the structure and patterns of real multiplayer learning environments.

Key Features:
- Realistic temporal patterns with learning phases
- Multiple event types with varying frequencies
- Player collaboration and individual action modeling
- Configurable session complexity and duration
- Ground truth change points for validation

Author: [Your Name]
Institution: [Your Institution]
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple, Optional, Union
import json

# Define realistic educational game events
EDUCATIONAL_GAME_EVENTS = {
    # Core learning actions
    'PlantSeed': {'base_freq': 0.15, 'learning_phase': 'early', 'collaboration': 'individual'},
    'HarvestCrop': {'base_freq': 0.12, 'learning_phase': 'mid', 'collaboration': 'individual'},
    'WaterPlant': {'base_freq': 0.10, 'learning_phase': 'all', 'collaboration': 'individual'},
    
    # Programming/Logic actions
    'CreateProgram': {'base_freq': 0.08, 'learning_phase': 'mid', 'collaboration': 'both'},
    'RunProgram': {'base_freq': 0.09, 'learning_phase': 'mid', 'collaboration': 'individual'},
    'DebugCode': {'base_freq': 0.05, 'learning_phase': 'late', 'collaboration': 'both'},
    
    # Strategic actions
    'PlanStrategy': {'base_freq': 0.06, 'learning_phase': 'early', 'collaboration': 'both'},
    'ExecuteStrategy': {'base_freq': 0.08, 'learning_phase': 'mid', 'collaboration': 'individual'},
    'ModifyStrategy': {'base_freq': 0.04, 'learning_phase': 'late', 'collaboration': 'both'},
    
    # Resource management
    'CollectResource': {'base_freq': 0.11, 'learning_phase': 'all', 'collaboration': 'individual'},
    'TradeResource': {'base_freq': 0.03, 'learning_phase': 'late', 'collaboration': 'both'},
    'UseResource': {'base_freq': 0.09, 'learning_phase': 'all', 'collaboration': 'individual'},
    
    # Navigation and exploration
    'MovePlayer': {'base_freq': 0.20, 'learning_phase': 'all', 'collaboration': 'individual'},
    'ExploreArea': {'base_freq': 0.07, 'learning_phase': 'early', 'collaboration': 'individual'},
    'NavigateMap': {'base_freq': 0.05, 'learning_phase': 'all', 'collaboration': 'individual'},
    
    # Communication and collaboration
    'SendMessage': {'base_freq': 0.04, 'learning_phase': 'all', 'collaboration': 'both'},
    'ShareResource': {'base_freq': 0.02, 'learning_phase': 'mid', 'collaboration': 'both'},
    'RequestHelp': {'base_freq': 0.02, 'learning_phase': 'all', 'collaboration': 'both'},
    
    # System interactions
    'OpenMenu': {'base_freq': 0.08, 'learning_phase': 'all', 'collaboration': 'individual'},
    'SaveProgress': {'base_freq': 0.01, 'learning_phase': 'all', 'collaboration': 'individual'},
    'ViewTutorial': {'base_freq': 0.03, 'learning_phase': 'early', 'collaboration': 'individual'},
}


class SyntheticSessionGenerator:
    """
    Generator for realistic synthetic educational game session data.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the synthetic data generator.
        
        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducible generation
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.event_types = list(EDUCATIONAL_GAME_EVENTS.keys())
        
    def generate_session(self,
                        session_id: Union[str, int],
                        duration_minutes: float = 15.0,
                        n_events: int = 200,
                        players: List[str] = None,
                        difficulty_level: str = 'medium',
                        learning_success: str = 'partial',
                        complexity: str = 'medium') -> pd.DataFrame:
        """
        Generate a complete synthetic educational game session.
        
        Parameters:
        -----------
        session_id : Union[str, int]
            Unique identifier for the session
        duration_minutes : float
            Session duration in minutes
        n_events : int
            Approximate number of events to generate
        players : List[str], optional
            List of player identifiers
        difficulty_level : str
            Game difficulty ('easy', 'medium', 'hard')
        learning_success : str
            Learning outcome ('unsuccessful', 'partial', 'full')
        complexity : str
            Session complexity ('low', 'medium', 'high')
            
        Returns:
        --------
        pd.DataFrame
            Synthetic session data matching real data structure
        """
        if players is None:
            players = ['player1', 'player2']
        
        # Calculate session timing
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        start_epoch = start_time.timestamp()
        end_epoch = end_time.timestamp()
        
        # Generate events with realistic temporal patterns
        events = self._generate_events_with_phases(
            n_events=n_events,
            duration_minutes=duration_minutes,
            players=players,
            difficulty_level=difficulty_level,
            learning_success=learning_success,
            complexity=complexity
        )
        
        # Create the session DataFrame
        session_data = []
        
        for event in events:
            row = {
                'session_id': session_id,
                'session_start_epoch': start_epoch,
                'session_end_epoch': end_epoch,
                'session_duration_mins': duration_minutes,
                'log_epoch': event['timestamp'],
                'eventKey': event['event_type'],
                'player_inference_total': event['player'],
                'has_session': True,
                # Add additional realistic fields
                'siteName': 'SYNTHETIC_EDU_GAME',
                'gameState': event.get('game_state', {}),
                'difficulty': difficulty_level,
                'learning_phase': event.get('phase', 'unknown')
            }
            session_data.append(row)
        
        df = pd.DataFrame(session_data)
        
        # Sort by timestamp to ensure chronological order
        df = df.sort_values('log_epoch').reset_index(drop=True)
        
        return df
    
    def _generate_events_with_phases(self,
                                   n_events: int,
                                   duration_minutes: float,
                                   players: List[str],
                                   difficulty_level: str,
                                   learning_success: str,
                                   complexity: str) -> List[Dict]:
        """
        Generate events with realistic learning phase patterns.
        """
        events = []
        duration_seconds = duration_minutes * 60
        
        # Define learning phases with different characteristics
        phases = [
            {'name': 'early', 'start': 0.0, 'end': 0.33, 'exploration': 0.7, 'collaboration': 0.3},
            {'name': 'mid', 'start': 0.33, 'end': 0.67, 'exploration': 0.4, 'collaboration': 0.6},
            {'name': 'late', 'start': 0.67, 'end': 1.0, 'exploration': 0.2, 'collaboration': 0.5}
        ]
        
        # Adjust event distribution based on learning success
        success_modifiers = {
            'unsuccessful': {'change_points': 0.8, 'scattered_behavior': 0.7},
            'partial': {'change_points': 0.5, 'scattered_behavior': 0.4},
            'full': {'change_points': 0.3, 'scattered_behavior': 0.2}
        }
        
        modifier = success_modifiers.get(learning_success, success_modifiers['partial'])
        
        # Generate events for each phase
        events_per_phase = n_events // len(phases)
        
        for phase in phases:
            phase_events = self._generate_phase_events(
                n_events=events_per_phase,
                phase=phase,
                duration_seconds=duration_seconds,
                players=players,
                difficulty_level=difficulty_level,
                modifier=modifier,
                complexity=complexity
            )
            events.extend(phase_events)
        
        # Add some strategic change points based on learning success
        if random.random() < modifier['change_points']:
            change_events = self._generate_change_point_events(
                duration_seconds=duration_seconds,
                players=players,
                n_changes=random.randint(1, 3)
            )
            events.extend(change_events)
        
        return sorted(events, key=lambda x: x['timestamp'])
    
    def _generate_phase_events(self,
                             n_events: int,
                             phase: Dict,
                             duration_seconds: float,
                             players: List[str],
                             difficulty_level: str,
                             modifier: Dict,
                             complexity: str) -> List[Dict]:
        """
        Generate events for a specific learning phase.
        """
        events = []
        phase_start = phase['start'] * duration_seconds
        phase_end = phase['end'] * duration_seconds
        
        # Complexity affects event density and variety
        complexity_modifiers = {
            'low': {'variety': 0.6, 'density': 0.8},
            'medium': {'variety': 1.0, 'density': 1.0},
            'high': {'variety': 1.4, 'density': 1.2}
        }
        
        comp_mod = complexity_modifiers.get(complexity, complexity_modifiers['medium'])
        adjusted_events = int(n_events * comp_mod['density'])
        
        for _ in range(adjusted_events):
            # Generate timestamp within phase
            timestamp = np.random.uniform(phase_start, phase_end)
            
            # Select event type based on phase preferences
            event_type = self._select_event_for_phase(phase['name'], comp_mod['variety'])
            
            # Determine player assignment
            player = self._assign_player(event_type, players, phase)
            
            # Create event
            event = {
                'timestamp': timestamp,
                'event_type': event_type,
                'player': player,
                'phase': phase['name'],
                'game_state': self._generate_game_state(event_type, difficulty_level)
            }
            
            events.append(event)
        
        return events
    
    def _select_event_for_phase(self, phase_name: str, variety_modifier: float) -> str:
        """
        Select an appropriate event type for the given learning phase.
        """
        # Filter events suitable for this phase
        suitable_events = []
        for event_type, props in EDUCATIONAL_GAME_EVENTS.items():
            if props['learning_phase'] == phase_name or props['learning_phase'] == 'all':
                # Weight by base frequency and variety modifier
                weight = props['base_freq'] * variety_modifier
                suitable_events.extend([event_type] * int(weight * 100))
        
        if not suitable_events:
            suitable_events = list(EDUCATIONAL_GAME_EVENTS.keys())
        
        return random.choice(suitable_events)
    
    def _assign_player(self, event_type: str, players: List[str], phase: Dict) -> str:
        """
        Assign an event to a player based on collaboration patterns.
        """
        event_props = EDUCATIONAL_GAME_EVENTS.get(event_type, {})
        collaboration = event_props.get('collaboration', 'individual')
        
        if collaboration == 'both' and random.random() < phase['collaboration']:
            return 'both'
        elif collaboration == 'individual' or random.random() < 0.8:
            return random.choice(players)
        else:
            return 'both'
    
    def _generate_change_point_events(self,
                                    duration_seconds: float,
                                    players: List[str],
                                    n_changes: int) -> List[Dict]:
        """
        Generate events that create deliberate change points for validation.
        """
        change_events = []
        
        # Events that typically indicate strategy changes
        change_indicators = [
            'PlanStrategy', 'ModifyStrategy', 'DebugCode', 
            'CreateProgram', 'TradeResource'
        ]
        
        for i in range(n_changes):
            # Space change points throughout the session
            timestamp = (i + 1) * (duration_seconds / (n_changes + 1))
            
            # Create a burst of unusual activity
            for _ in range(random.randint(3, 7)):
                event = {
                    'timestamp': timestamp + random.uniform(-30, 30),
                    'event_type': random.choice(change_indicators),
                    'player': random.choice(players + ['both']),
                    'phase': 'change_point',
                    'game_state': {'change_indicator': True}
                }
                change_events.append(event)
        
        return change_events
    
    def _generate_game_state(self, event_type: str, difficulty_level: str) -> Dict:
        """
        Generate realistic game state information for an event.
        """
        base_state = {
            'event_type': event_type,
            'difficulty': difficulty_level,
            'timestamp_local': datetime.now().isoformat()
        }
        
        # Add event-specific state information
        if 'Plant' in event_type:
            base_state.update({
                'crop_type': random.choice(['corn', 'wheat', 'tomato', 'carrot']),
                'growth_stage': random.randint(0, 5)
            })
        elif 'Program' in event_type:
            base_state.update({
                'code_lines': random.randint(5, 50),
                'syntax_errors': random.randint(0, 3),
                'logic_complexity': random.choice(['simple', 'moderate', 'complex'])
            })
        elif 'Resource' in event_type:
            base_state.update({
                'resource_type': random.choice(['wood', 'stone', 'food', 'energy']),
                'quantity': random.randint(1, 20)
            })
        elif 'Move' in event_type:
            base_state.update({
                'x_coordinate': random.uniform(0, 100),
                'y_coordinate': random.uniform(0, 100),
                'movement_speed': random.uniform(1, 5)
            })
        
        return base_state


def generate_synthetic_session_data(session_id: Union[str, int],
                                  duration_minutes: float = 15.0,
                                  n_events: int = 200,
                                  players: List[str] = None,
                                  difficulty_level: str = 'medium',
                                  learning_success: str = 'partial',
                                  complexity: str = 'medium',
                                  seed: Optional[int] = None) -> pd.DataFrame:
    """
    Convenience function for generating synthetic session data.
    
    This function provides a simple interface for creating realistic
    educational game session data for testing and development.
    
    Parameters:
    -----------
    session_id : Union[str, int]
        Unique identifier for the session
    duration_minutes : float
        Session duration in minutes
    n_events : int
        Approximate number of events to generate
    players : List[str], optional
        List of player identifiers (defaults to ['player1', 'player2'])
    difficulty_level : str
        Game difficulty level ('easy', 'medium', 'hard')
    learning_success : str
        Simulated learning outcome ('unsuccessful', 'partial', 'full')
    complexity : str
        Session complexity level ('low', 'medium', 'high')
    seed : int, optional
        Random seed for reproducible generation
        
    Returns:
    --------
    pd.DataFrame
        Synthetic session data ready for analysis
        
    Examples:
    ---------
    >>> # Generate a basic session
    >>> session_data = generate_synthetic_session_data(session_id=1)
    
    >>> # Generate a complex, successful learning session
    >>> advanced_session = generate_synthetic_session_data(
    ...     session_id=2,
    ...     duration_minutes=25,
    ...     n_events=350,
    ...     learning_success='full',
    ...     complexity='high'
    ... )
    
    >>> # Generate reproducible data for testing
    >>> test_session = generate_synthetic_session_data(
    ...     session_id='test',
    ...     seed=42
    ... )
    """
    generator = SyntheticSessionGenerator(seed=seed)
    
    return generator.generate_session(
        session_id=session_id,
        duration_minutes=duration_minutes,
        n_events=n_events,
        players=players,
        difficulty_level=difficulty_level,
        learning_success=learning_success,
        complexity=complexity
    )


def generate_reference_statistics(n_sessions: int = 100,
                                save_path: Optional[str] = None) -> Dict[str, Dict]:
    """
    Generate reference statistics from multiple synthetic sessions.
    
    This creates realistic reference means and standard deviations
    that can be used for reference-based z-score analysis.
    
    Parameters:
    -----------
    n_sessions : int
        Number of sessions to generate for statistics
    save_path : str, optional
        Path to save reference statistics as JSON
        
    Returns:
    --------
    Dict[str, Dict]
        Dictionary containing 'means' and 'stds' for each event type
    """
    print(f"Generating reference statistics from {n_sessions} synthetic sessions...")
    
    generator = SyntheticSessionGenerator(seed=42)  # Fixed seed for consistency
    all_event_counts = []
    
    for session_id in range(1, n_sessions + 1):
        # Vary session parameters to create realistic distribution
        duration = np.random.uniform(10, 30)
        n_events = np.random.randint(150, 400)
        success = np.random.choice(['unsuccessful', 'partial', 'full'], 
                                 p=[0.2, 0.5, 0.3])
        complexity = np.random.choice(['low', 'medium', 'high'], 
                                    p=[0.3, 0.5, 0.2])
        
        session_data = generator.generate_session(
            session_id=session_id,
            duration_minutes=duration,
            n_events=n_events,
            learning_success=success,
            complexity=complexity
        )
        
        # Count events per time window (simplified aggregation)
        event_counts = session_data['eventKey'].value_counts().to_dict()
        
        # Ensure all event types are represented
        for event_type in EDUCATIONAL_GAME_EVENTS.keys():
            if event_type not in event_counts:
                event_counts[event_type] = 0
        
        all_event_counts.append(event_counts)
    
    # Calculate statistics
    event_df = pd.DataFrame(all_event_counts).fillna(0)
    
    reference_stats = {
        'means': event_df.mean().to_dict(),
        'stds': event_df.std().to_dict(),
        'metadata': {
            'n_sessions': n_sessions,
            'generation_date': datetime.now().isoformat(),
            'event_types': list(EDUCATIONAL_GAME_EVENTS.keys()),
            'total_events_analyzed': len(all_event_counts)
        }
    }
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(reference_stats, f, indent=2)
        print(f"Reference statistics saved to: {save_path}")
    
    print("Reference statistics generation complete!")
    print(f"  • Event types: {len(reference_stats['means'])}")
    print(f"  • Sessions analyzed: {n_sessions}")
    print(f"  • Mean events per session: {event_df.sum(axis=1).mean():.1f}")
    
    return reference_stats


def create_validation_dataset(n_sessions: int = 20,
                            with_ground_truth: bool = True,
                            save_dir: Optional[str] = None) -> List[Dict]:
    """
    Create a validation dataset with known change points for method testing.
    
    Parameters:
    -----------
    n_sessions : int
        Number of validation sessions to create
    with_ground_truth : bool
        Whether to include ground truth change point annotations
    save_dir : str, optional
        Directory to save validation dataset
        
    Returns:
    --------
    List[Dict]
        List of session data with optional ground truth annotations
    """
    print(f"Creating validation dataset with {n_sessions} sessions...")
    
    generator = SyntheticSessionGenerator(seed=123)  # Fixed seed for reproducibility
    validation_data = []
    
    for session_id in range(1, n_sessions + 1):
        # Create sessions with varied learning patterns
        learning_patterns = [
            {'success': 'full', 'complexity': 'medium', 'expected_changes': 'low'},
            {'success': 'partial', 'complexity': 'medium', 'expected_changes': 'medium'},
            {'success': 'unsuccessful', 'complexity': 'high', 'expected_changes': 'high'},
            {'success': 'full', 'complexity': 'high', 'expected_changes': 'medium'},
            {'success': 'partial', 'complexity': 'low', 'expected_changes': 'low'}
        ]
        
        pattern = learning_patterns[session_id % len(learning_patterns)]
        
        session_data = generator.generate_session(
            session_id=f"val_{session_id}",
            duration_minutes=np.random.uniform(12, 25),
            n_events=np.random.randint(180, 320),
            learning_success=pattern['success'],
            complexity=pattern['complexity']
        )
        
        validation_entry = {
            'session_id': f"val_{session_id}",
            'data': session_data,
            'metadata': {
                'learning_success': pattern['success'],
                'complexity': pattern['complexity'],
                'expected_change_level': pattern['expected_changes'],
                'duration_minutes': session_data['session_duration_mins'].iloc[0],
                'total_events': len(session_data)
            }
        }
        
        if with_ground_truth:
            # Add ground truth change points based on the session pattern
            ground_truth = generate_ground_truth_change_points(
                session_data, pattern['expected_changes']
            )
            validation_entry['ground_truth_change_points'] = ground_truth
        
        validation_data.append(validation_entry)
    
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for entry in validation_data:
            session_id = entry['session_id']
            
            # Save session data
            entry['data'].to_csv(f"{save_dir}/{session_id}_data.csv", index=False)
            
            # Save metadata and ground truth
            metadata = {
                'metadata': entry['metadata'],
                'ground_truth_change_points': entry.get('ground_truth_change_points', [])
            }
            
            with open(f"{save_dir}/{session_id}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Validation dataset saved to: {save_dir}/")
    
    print("Validation dataset creation complete!")
    print(f"  • Sessions: {n_sessions}")
    print(f"  • Ground truth included: {with_ground_truth}")
    
    return validation_data


def generate_ground_truth_change_points(session_data: pd.DataFrame, 
                                      expected_level: str) -> List[Dict]:
    """
    Generate ground truth change points for validation.
    
    Parameters:
    -----------
    session_data : pd.DataFrame
        Session data to analyze
    expected_level : str
        Expected number of change points ('low', 'medium', 'high')
        
    Returns:
    --------
    List[Dict]
        Ground truth change point annotations
    """
    duration = session_data['session_duration_mins'].iloc[0]
    
    # Define expected change point counts
    change_counts = {
        'low': (0, 2),
        'medium': (2, 4), 
        'high': (4, 7)
    }
    
    min_changes, max_changes = change_counts.get(expected_level, (1, 3))
    n_changes = np.random.randint(min_changes, max_changes + 1)
    
    ground_truth = []
    
    if n_changes > 0:
        # Generate change points at strategic times
        change_times = np.random.uniform(0.1, 0.9, n_changes)
        change_times = sorted(change_times)
        
        for i, time_pct in enumerate(change_times):
            # Determine the type of change
            change_types = [
                'strategy_shift', 'engagement_change', 'learning_transition',
                'collaboration_change', 'difficulty_adaptation'
            ]
            
            change_point = {
                'time_percentage': time_pct,
                'time_minutes': time_pct * duration,
                'change_type': np.random.choice(change_types),
                'intensity': np.random.uniform(2.0, 5.0),  # Above typical threshold
                'affected_players': np.random.choice([['player1'], ['player2'], ['player1', 'player2']]),
                'description': f"Ground truth change point {i+1}: {np.random.choice(change_types)}",
                'confidence': np.random.uniform(0.8, 1.0)
            }
            
            ground_truth.append(change_point)
    
    return ground_truth


class PerformanceTestDataGenerator:
    """
    Generator for performance testing with large-scale datasets.
    """
    
    @staticmethod
    def generate_stress_test_data(n_sessions: int = 1000,
                                session_duration_range: Tuple[float, float] = (5, 60),
                                events_per_minute_range: Tuple[int, int] = (10, 30)) -> List[pd.DataFrame]:
        """
        Generate large-scale data for performance testing.
        
        Parameters:
        -----------
        n_sessions : int
            Number of sessions to generate
        session_duration_range : Tuple[float, float]
            Min and max session duration in minutes
        events_per_minute_range : Tuple[int, int]
            Min and max events per minute
            
        Returns:
        --------
        List[pd.DataFrame]
            List of session DataFrames for stress testing
        """
        print(f"Generating stress test data: {n_sessions} sessions...")
        
        generator = SyntheticSessionGenerator()
        sessions = []
        
        for session_id in range(1, n_sessions + 1):
            duration = np.random.uniform(*session_duration_range)
            events_per_min = np.random.randint(*events_per_minute_range)
            n_events = int(duration * events_per_min)
            
            session_data = generator.generate_session(
                session_id=session_id,
                duration_minutes=duration,
                n_events=n_events,
                learning_success=np.random.choice(['unsuccessful', 'partial', 'full']),
                complexity=np.random.choice(['low', 'medium', 'high'])
            )
            
            sessions.append(session_data)
            
            if session_id % 100 == 0:
                print(f"  Generated {session_id}/{n_sessions} sessions...")
        
        print(f"Stress test data generation complete!")
        print(f"  • Total sessions: {n_sessions}")
        print(f"  • Average events per session: {sum(len(s) for s in sessions) / len(sessions):.1f}")
        
        return sessions


def demonstrate_synthetic_data():
    """
    Demonstrate the capabilities of the synthetic data generator.
    """
    print("🎲 Synthetic Data Generator Demonstration")
    print("=" * 50)
    
    # Example 1: Basic session generation
    print("\n📊 Example 1: Basic Session Generation")
    print("-" * 40)
    
    basic_session = generate_synthetic_session_data(
        session_id="demo_basic",
        duration_minutes=15,
        n_events=200,
        seed=42  # For reproducible results
    )
    
    print(f"✅ Generated basic session:")
    print(f"   • Duration: {basic_session['session_duration_mins'].iloc[0]} minutes")
    print(f"   • Total events: {len(basic_session)}")
    print(f"   • Event types: {basic_session['eventKey'].nunique()}")
    print(f"   • Players: {list(basic_session['player_inference_total'].unique())}")
    
    # Example 2: Advanced session with specific characteristics
    print("\n📊 Example 2: Advanced Session Configuration")
    print("-" * 45)
    
    advanced_session = generate_synthetic_session_data(
        session_id="demo_advanced",
        duration_minutes=25,
        n_events=350,
        learning_success='full',
        complexity='high',
        difficulty_level='hard',
        seed=42
    )
    
    print(f"✅ Generated advanced session:")
    print(f"   • Learning success: full")
    print(f"   • Complexity: high")
    print(f"   • Difficulty: hard")
    print(f"   • Events: {len(advanced_session)}")
    
    # Example 3: Event distribution analysis
    print("\n📊 Example 3: Event Distribution Analysis")
    print("-" * 40)
    
    event_dist = advanced_session['eventKey'].value_counts().head(10)
    print("Top 10 most frequent events:")
    for event, count in event_dist.items():
        percentage = (count / len(advanced_session)) * 100
        print(f"   • {event}: {count} times ({percentage:.1f}%)")
    
    # Example 4: Reference statistics generation
    print("\n📊 Example 4: Reference Statistics Generation")
    print("-" * 45)
    
    ref_stats = generate_reference_statistics(
        n_sessions=50,  # Smaller number for demo
        save_path='output/demo_reference_stats.json'
    )
    
    print("✅ Reference statistics generated:")
    print(f"   • Event types covered: {len(ref_stats['means'])}")
    print(f"   • Sessions analyzed: {ref_stats['metadata']['n_sessions']}")
    
    # Example 5: Validation dataset
    print("\n📊 Example 5: Validation Dataset Creation")
    print("-" * 40)
    
    validation_data = create_validation_dataset(
        n_sessions=5,  # Small number for demo
        with_ground_truth=True,
        save_dir='output/validation_demo'
    )
    
    print("✅ Validation dataset created:")
    print(f"   • Sessions: {len(validation_data)}")
    print(f"   • Ground truth included: Yes")
    
    # Show example ground truth
    if validation_data and 'ground_truth_change_points' in validation_data[0]:
        example_gt = validation_data[0]['ground_truth_change_points']
        print(f"   • Example change points: {len(example_gt)}")
        for cp in example_gt[:2]:  # Show first 2
            print(f"     - {cp['change_type']} at {cp['time_percentage']*100:.1f}%")
    
    print("\n🎉 Synthetic Data Generation Demo Complete!")
    print("Generated files saved to output/ directory")


if __name__ == "__main__":
    # Ensure output directory exists
    import os
    os.makedirs('output', exist_ok=True)
    
    # Run demonstration
    demonstrate_synthetic_data()
    
    print("\n💡 Usage Tips:")
    print("  • Use consistent seeds for reproducible testing")
    print("  • Vary learning_success and complexity for diverse datasets")
    print("  • Generate reference statistics from large sample sizes (100+ sessions)")
    print("  • Use validation datasets to test change detection accuracy")
    print("  • Adjust duration and event counts to match your real data characteristics")
#!/usr/bin/env python3
"""
Comparative Battery Visualizer Module
배터리 간 비교 및 벤치마킹 시각화 모듈
다양한 조건, 배터리 타입, 성능 지표를 비교하고 벤치마킹
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# Professional comparison styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

logger = logging.getLogger(__name__)

class ComparativeVisualizer:
    """배터리 비교 및 벤치마킹 시각화 클래스"""
    
    def __init__(self, output_dir: str = "analysis_output/comparative_plots"):
        """
        초기화
        
        Args:
            output_dir: 출력 디렉토리 경로
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Comparative color schemes
        self.battery_types_colors = sns.color_palette("Set1", n_colors=10)
        self.performance_colors = sns.color_palette("RdYlGn", n_colors=10)
        self.condition_colors = sns.color_palette("plasma", n_colors=15)
        self.benchmark_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Benchmark reference values (industry standards)
        self.benchmarks = {
            'capacity_retention_80_cycles': 800,  # Cycles to 80% retention
            'energy_density_target': 250,  # Wh/kg
            'power_density_target': 300,   # W/kg
            'round_trip_efficiency': 90,   # %
            'cycle_life_target': 3000      # cycles
        }
        
        logger.info(f"Comparative visualizer initialized, output: {self.output_dir}")
    
    def prepare_comparative_data(self, data_sources: Union[pd.DataFrame, List[pd.DataFrame]], 
                               labels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        비교 분석을 위한 데이터 준비
        
        Args:
            data_sources: 단일 데이터프레임 또는 비교할 데이터프레임 리스트
            labels: 각 데이터소스의 라벨
            
        Returns:
            비교 분석용 통합 데이터프레임
        """
        logger.info("Preparing comparative analysis data")
        
        # Handle single dataframe case
        if isinstance(data_sources, pd.DataFrame):
            data_sources = [data_sources]
            if labels is None:
                labels = ['Battery_1']
        
        # Default labels if not provided
        if labels is None:
            labels = [f'Battery_{i+1}' for i in range(len(data_sources))]
        
        # Create comparative dataset
        comparative_data = []
        
        for i, (df, label) in enumerate(zip(data_sources, labels)):
            # Add battery identifier
            df_copy = df.copy()
            df_copy['Battery_Type'] = label
            df_copy['Battery_ID'] = i
            
            # Calculate key metrics for comparison
            cycles = sorted(df_copy['TotalCycle'].unique())
            
            for cycle in cycles:
                cycle_data = df_copy[df_copy['TotalCycle'] == cycle]
                if len(cycle_data) > 0:
                    
                    # Basic metrics
                    voltage_mean = cycle_data['Voltage[V]'].mean()
                    voltage_std = cycle_data['Voltage[V]'].std()
                    current_mean = cycle_data['Current[A]'].mean()
                    capacity_charge = cycle_data['Chg_Capacity[Ah]'].max()
                    capacity_discharge = abs(cycle_data['Dchg_Capacity[Ah]'].min())
                    
                    # Advanced metrics
                    energy_density = voltage_mean * capacity_discharge  # Simplified Wh/kg
                    power_density = voltage_mean * np.abs(current_mean)  # Simplified W/kg
                    coulombic_efficiency = capacity_discharge / (capacity_charge + 1e-10)
                    
                    # Determine operating conditions
                    avg_current = np.abs(current_mean)
                    if avg_current < 1.0:
                        c_rate_category = 'Low (<1C)'
                    elif avg_current < 2.0:
                        c_rate_category = 'Medium (1-2C)'
                    else:
                        c_rate_category = 'High (>2C)'
                    
                    # Determine lifecycle phase
                    if cycle <= 100:
                        lifecycle_phase = 'Early'
                    elif cycle <= 500:
                        lifecycle_phase = 'Middle'
                    else:
                        lifecycle_phase = 'Late'
                    
                    comparative_data.append({
                        'Battery_Type': label,
                        'Battery_ID': i,
                        'Cycle': cycle,
                        'Voltage_Mean': voltage_mean,
                        'Voltage_Std': voltage_std,
                        'Current_Mean': current_mean,
                        'Capacity_Charge': capacity_charge,
                        'Capacity_Discharge': capacity_discharge,
                        'Energy_Density': energy_density,
                        'Power_Density': power_density,
                        'Coulombic_Efficiency': coulombic_efficiency,
                        'C_Rate_Category': c_rate_category,
                        'Lifecycle_Phase': lifecycle_phase,
                        'Duration': len(cycle_data),
                        'Performance_Score': self._calculate_performance_score(cycle_data)
                    })
        
        return pd.DataFrame(comparative_data)
    
    def _calculate_performance_score(self, cycle_data: pd.DataFrame) -> float:
        """성능 스코어 계산"""
        try:
            voltage_score = (cycle_data['Voltage[V]'].mean() - 2.5) / (4.5 - 2.5)
            capacity_score = cycle_data['Chg_Capacity[Ah]'].max() / 5.0
            efficiency = abs(cycle_data['Dchg_Capacity[Ah]'].min()) / (cycle_data['Chg_Capacity[Ah]'].max() + 1e-10)
            
            composite_score = (voltage_score * 0.3 + capacity_score * 0.4 + efficiency * 0.3) * 100
            return np.clip(composite_score, 0, 100)
        except:
            return 50.0
    
    def create_performance_radar_chart(self, comparative_data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        성능 레이더 차트 (다중 배터리 비교)
        
        Args:
            comparative_data: 비교 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating performance radar chart comparison")
        
        # Aggregate data by battery type
        battery_stats = []
        battery_types = comparative_data['Battery_Type'].unique()
        
        for battery_type in battery_types:
            battery_data = comparative_data[comparative_data['Battery_Type'] == battery_type]
            
            stats = {
                'Battery_Type': battery_type,
                'Avg_Voltage': battery_data['Voltage_Mean'].mean(),
                'Avg_Capacity': battery_data['Capacity_Discharge'].mean(),
                'Avg_Efficiency': battery_data['Coulombic_Efficiency'].mean(),
                'Avg_Energy_Density': battery_data['Energy_Density'].mean(),
                'Avg_Power_Density': battery_data['Power_Density'].mean(),
                'Voltage_Stability': 1 / (battery_data['Voltage_Std'].mean() + 1e-10),  # Inverse of std
                'Cycle_Count': battery_data['Cycle'].max()
            }
            battery_stats.append(stats)
        
        if len(battery_stats) < 2:
            logger.warning("Need at least 2 batteries for comparison")
            return plt.figure()
        
        stats_df = pd.DataFrame(battery_stats)
        
        # Normalize metrics for radar chart (0-1 scale)
        metrics = ['Avg_Voltage', 'Avg_Capacity', 'Avg_Efficiency', 
                  'Avg_Energy_Density', 'Avg_Power_Density', 'Voltage_Stability', 'Cycle_Count']
        
        scaler = MinMaxScaler()
        normalized_stats = stats_df.copy()
        normalized_stats[metrics] = scaler.fit_transform(stats_df[metrics])
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        
        for i, (_, row) in enumerate(normalized_stats.iterrows()):
            values = [row[metric] for metric in metrics]
            values += [values[0]]  # Complete the circle
            
            color = self.battery_types_colors[i % len(self.battery_types_colors)]
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Battery_Type'], 
                   color=color, alpha=0.8)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.replace('Avg_', '').replace('_', ' ') for metric in metrics])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True, alpha=0.3)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title('Battery Performance Radar Comparison', size=16, fontweight='bold', pad=30)
        
        if save:
            save_path = self.output_dir / 'performance_radar_chart.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_benchmark_comparison(self, comparative_data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        산업 표준 벤치마크와의 비교
        
        Args:
            comparative_data: 비교 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating benchmark comparison analysis")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Industry Benchmark Comparison', fontsize=16, fontweight='bold', y=0.98)
        
        battery_types = comparative_data['Battery_Type'].unique()
        
        # 1. Capacity Retention Benchmark
        ax1 = axes[0, 0]
        
        retention_data = []
        for battery_type in battery_types:
            battery_data = comparative_data[comparative_data['Battery_Type'] == battery_type]
            
            # Calculate capacity retention over cycles
            initial_capacity = battery_data['Capacity_Discharge'].iloc[0] if len(battery_data) > 0 else 4.0
            retention_80 = None
            
            for _, row in battery_data.iterrows():
                current_retention = (row['Capacity_Discharge'] / initial_capacity) * 100
                if current_retention <= 80 and retention_80 is None:
                    retention_80 = row['Cycle']
                    break
            
            if retention_80 is None:
                retention_80 = battery_data['Cycle'].max()  # Haven't reached 80% yet
            
            retention_data.append({
                'Battery_Type': battery_type,
                'Cycles_to_80': retention_80,
                'Performance': 'Excellent' if retention_80 > 1000 else 'Good' if retention_80 > 500 else 'Fair'
            })
        
        if retention_data:
            retention_df = pd.DataFrame(retention_data)
            
            bars = ax1.bar(retention_df['Battery_Type'], retention_df['Cycles_to_80'], 
                          alpha=0.7, color=self.performance_colors[:len(retention_df)])
            
            # Add benchmark line
            ax1.axhline(y=self.benchmarks['capacity_retention_80_cycles'], color='red', 
                       linestyle='--', linewidth=2, label='Industry Standard')
            
            # Add values on bars
            for bar, cycles in zip(bars, retention_df['Cycles_to_80']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                        f'{cycles}', ha='center', va='bottom', fontweight='bold')
            
            ax1.set_xlabel('Battery Type', fontweight='bold')
            ax1.set_ylabel('Cycles to 80% Retention', fontweight='bold')
            ax1.set_title('Capacity Retention Benchmark', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Energy vs Power Density Benchmark
        ax2 = axes[0, 1]
        
        for i, battery_type in enumerate(battery_types):
            battery_data = comparative_data[comparative_data['Battery_Type'] == battery_type]
            
            # Use representative values (e.g., cycle 100)
            cycle_100_data = battery_data[battery_data['Cycle'] == 100]
            if len(cycle_100_data) == 0:
                cycle_100_data = battery_data.iloc[:1]  # Use first available
            
            if len(cycle_100_data) > 0:
                energy_density = cycle_100_data['Energy_Density'].iloc[0]
                power_density = cycle_100_data['Power_Density'].iloc[0]
                
                color = self.battery_types_colors[i % len(self.battery_types_colors)]
                ax2.scatter(power_density, energy_density, s=100, alpha=0.7, 
                           color=color, label=battery_type)
        
        # Add benchmark targets
        ax2.axhline(y=self.benchmarks['energy_density_target'], color='red', 
                   linestyle='--', alpha=0.7, label='Energy Target')
        ax2.axvline(x=self.benchmarks['power_density_target'], color='blue', 
                   linestyle='--', alpha=0.7, label='Power Target')
        
        ax2.set_xlabel('Power Density (W/kg)', fontweight='bold')
        ax2.set_ylabel('Energy Density (Wh/kg)', fontweight='bold')
        ax2.set_title('Energy vs Power Density Benchmark', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Efficiency Benchmark
        ax3 = axes[1, 0]
        
        efficiency_data = []
        for battery_type in battery_types:
            battery_data = comparative_data[comparative_data['Battery_Type'] == battery_type]
            avg_efficiency = battery_data['Coulombic_Efficiency'].mean() * 100
            
            efficiency_data.append({
                'Battery_Type': battery_type,
                'Avg_Efficiency': avg_efficiency
            })
        
        if efficiency_data:
            eff_df = pd.DataFrame(efficiency_data)
            
            bars = ax3.bar(eff_df['Battery_Type'], eff_df['Avg_Efficiency'], 
                          alpha=0.7, color=self.performance_colors[:len(eff_df)])
            
            # Add benchmark line
            ax3.axhline(y=self.benchmarks['round_trip_efficiency'], color='red', 
                       linestyle='--', linewidth=2, label='Industry Target')
            
            # Add values on bars
            for bar, eff in zip(bars, eff_df['Avg_Efficiency']):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{eff:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax3.set_xlabel('Battery Type', fontweight='bold')
            ax3.set_ylabel('Round-Trip Efficiency (%)', fontweight='bold')
            ax3.set_title('Efficiency Benchmark', fontweight='bold')
            ax3.legend()
            ax3.set_ylim([80, 105])
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Overall Performance Score
        ax4 = axes[1, 1]
        
        performance_summary = []
        for battery_type in battery_types:
            battery_data = comparative_data[comparative_data['Battery_Type'] == battery_type]
            avg_performance = battery_data['Performance_Score'].mean()
            
            performance_summary.append({
                'Battery_Type': battery_type,
                'Performance_Score': avg_performance
            })
        
        if performance_summary:
            perf_df = pd.DataFrame(performance_summary)
            
            # Create horizontal bar chart
            bars = ax4.barh(perf_df['Battery_Type'], perf_df['Performance_Score'], 
                           alpha=0.7, color=self.performance_colors[:len(perf_df)])
            
            # Add benchmark line (assume 70% is industry average)
            ax4.axvline(x=70, color='red', linestyle='--', linewidth=2, label='Industry Average')
            
            # Add values on bars
            for bar, score in zip(bars, perf_df['Performance_Score']):
                ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{score:.1f}', ha='left', va='center', fontweight='bold')
            
            ax4.set_xlabel('Performance Score', fontweight='bold')
            ax4.set_title('Overall Performance Benchmark', fontweight='bold')
            ax4.legend()
            ax4.set_xlim([0, 100])
            ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'benchmark_comparison.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_condition_based_comparison(self, comparative_data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        운전 조건별 성능 비교
        
        Args:
            comparative_data: 비교 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating condition-based performance comparison")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Comparison by Operating Conditions', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Performance by C-rate category
        ax1 = axes[0, 0]
        
        sns.boxplot(data=comparative_data, x='C_Rate_Category', y='Performance_Score', 
                   hue='Battery_Type', ax=ax1, palette=self.battery_types_colors)
        ax1.set_xlabel('C-rate Category', fontweight='bold')
        ax1.set_ylabel('Performance Score', fontweight='bold')
        ax1.set_title('Performance by C-rate', fontweight='bold')
        ax1.legend(title='Battery Type', loc='upper right')
        
        # 2. Capacity fade by lifecycle phase
        ax2 = axes[0, 1]
        
        # Calculate capacity retention for each phase
        phase_retention = []
        for battery_type in comparative_data['Battery_Type'].unique():
            battery_data = comparative_data[comparative_data['Battery_Type'] == battery_type]
            
            for phase in ['Early', 'Middle', 'Late']:
                phase_data = battery_data[battery_data['Lifecycle_Phase'] == phase]
                if len(phase_data) > 0:
                    # Calculate retention relative to early phase
                    early_capacity = battery_data[battery_data['Lifecycle_Phase'] == 'Early']['Capacity_Discharge'].mean()
                    current_capacity = phase_data['Capacity_Discharge'].mean()
                    retention = (current_capacity / early_capacity) * 100 if early_capacity > 0 else 100
                    
                    phase_retention.append({
                        'Battery_Type': battery_type,
                        'Lifecycle_Phase': phase,
                        'Capacity_Retention': retention
                    })
        
        if phase_retention:
            retention_df = pd.DataFrame(phase_retention)
            
            sns.lineplot(data=retention_df, x='Lifecycle_Phase', y='Capacity_Retention', 
                        hue='Battery_Type', marker='o', linewidth=2, markersize=8, 
                        ax=ax2, palette=self.battery_types_colors)
            ax2.set_xlabel('Lifecycle Phase', fontweight='bold')
            ax2.set_ylabel('Capacity Retention (%)', fontweight='bold')
            ax2.set_title('Capacity Retention by Lifecycle Phase', fontweight='bold')
            ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% EOL')
            ax2.legend(title='Battery Type')
            ax2.grid(True, alpha=0.3)
        
        # 3. Efficiency distribution comparison
        ax3 = axes[1, 0]
        
        for i, battery_type in enumerate(comparative_data['Battery_Type'].unique()):
            battery_data = comparative_data[comparative_data['Battery_Type'] == battery_type]
            efficiency_data = battery_data['Coulombic_Efficiency'] * 100
            
            # Remove outliers
            q75, q25 = np.percentile(efficiency_data, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - (iqr * 1.5)
            upper_bound = q75 + (iqr * 1.5)
            clean_data = efficiency_data[(efficiency_data >= lower_bound) & (efficiency_data <= upper_bound)]
            
            if len(clean_data) > 0:
                ax3.hist(clean_data, bins=20, alpha=0.6, density=True, 
                        color=self.battery_types_colors[i], label=battery_type)
        
        ax3.set_xlabel('Coulombic Efficiency (%)', fontweight='bold')
        ax3.set_ylabel('Density', fontweight='bold')
        ax3.set_title('Efficiency Distribution Comparison', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Statistical comparison table (as heatmap)
        ax4 = axes[1, 1]
        
        # Create statistical summary
        stats_summary = []
        for battery_type in comparative_data['Battery_Type'].unique():
            battery_data = comparative_data[comparative_data['Battery_Type'] == battery_type]
            
            stats = {
                'Battery_Type': battery_type,
                'Mean_Performance': battery_data['Performance_Score'].mean(),
                'Std_Performance': battery_data['Performance_Score'].std(),
                'Mean_Efficiency': battery_data['Coulombic_Efficiency'].mean(),
                'Mean_Capacity': battery_data['Capacity_Discharge'].mean(),
                'Capacity_CV': battery_data['Capacity_Discharge'].std() / battery_data['Capacity_Discharge'].mean()
            }
            stats_summary.append(stats)
        
        if stats_summary:
            stats_df = pd.DataFrame(stats_summary)
            stats_df = stats_df.set_index('Battery_Type')
            
            # Create heatmap
            sns.heatmap(stats_df.T, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                       ax=ax4, cbar_kws={'label': 'Value'})
            ax4.set_title('Statistical Summary Heatmap', fontweight='bold')
            ax4.set_xlabel('Battery Type', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'condition_based_comparison.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_degradation_comparison(self, comparative_data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        배터리별 성능 저하 패턴 비교
        
        Args:
            comparative_data: 비교 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating degradation pattern comparison")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Battery Degradation Pattern Comparison', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        battery_types = comparative_data['Battery_Type'].unique()
        
        # 1. Capacity degradation trends
        ax1 = axes[0, 0]
        
        for i, battery_type in enumerate(battery_types):
            battery_data = comparative_data[comparative_data['Battery_Type'] == battery_type]
            
            # Calculate capacity retention
            if len(battery_data) > 0:
                initial_capacity = battery_data['Capacity_Discharge'].iloc[0]
                cycles = battery_data['Cycle'].values
                retention = (battery_data['Capacity_Discharge'] / initial_capacity) * 100
                
                color = self.battery_types_colors[i % len(self.battery_types_colors)]
                ax1.plot(cycles, retention, linewidth=2, alpha=0.8, 
                        color=color, label=battery_type)
                
                # Add trend line
                if len(cycles) > 5:
                    z = np.polyfit(cycles, retention, 1)
                    p = np.poly1d(z)
                    ax1.plot(cycles, p(cycles), '--', alpha=0.5, color=color,
                            label=f'{battery_type} trend ({z[0]:.4f}%/cycle)')
        
        ax1.set_xlabel('Cycle Number', fontweight='bold')
        ax1.set_ylabel('Capacity Retention (%)', fontweight='bold')
        ax1.set_title('Capacity Degradation Trends', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=80, color='red', linestyle=':', alpha=0.7, label='80% EOL')
        
        # 2. Voltage degradation analysis
        ax2 = axes[0, 1]
        
        for i, battery_type in enumerate(battery_types):
            battery_data = comparative_data[comparative_data['Battery_Type'] == battery_type]
            
            # Calculate moving average of voltage
            window_size = min(20, len(battery_data) // 5)
            if window_size > 2:
                voltage_ma = battery_data['Voltage_Mean'].rolling(window=window_size, center=True).mean()
                
                color = self.battery_types_colors[i % len(self.battery_types_colors)]
                ax2.plot(battery_data['Cycle'], voltage_ma, linewidth=2, alpha=0.8,
                        color=color, label=battery_type)
        
        ax2.set_xlabel('Cycle Number', fontweight='bold')
        ax2.set_ylabel('Average Voltage (V)', fontweight='bold')
        ax2.set_title('Voltage Evolution Comparison', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Degradation rate comparison
        ax3 = axes[1, 0]
        
        degradation_rates = []
        for battery_type in battery_types:
            battery_data = comparative_data[comparative_data['Battery_Type'] == battery_type]
            
            if len(battery_data) > 10:
                # Calculate degradation rate (capacity loss per cycle)
                cycles = battery_data['Cycle'].values
                capacity = battery_data['Capacity_Discharge'].values
                
                # Fit linear trend and extract slope
                if len(cycles) > 5:
                    initial_capacity = capacity[0] if capacity[0] > 0 else 4.0
                    retention = (capacity / initial_capacity) * 100
                    
                    slope, _, r_value, _, _ = stats.linregress(cycles, retention)
                    
                    degradation_rates.append({
                        'Battery_Type': battery_type,
                        'Degradation_Rate': abs(slope),  # %/cycle
                        'R_squared': r_value**2
                    })
        
        if degradation_rates:
            deg_df = pd.DataFrame(degradation_rates)
            
            bars = ax3.bar(deg_df['Battery_Type'], deg_df['Degradation_Rate'], 
                          alpha=0.7, color=self.performance_colors[:len(deg_df)])
            
            # Add R² values as text
            for bar, r2 in zip(bars, deg_df['R_squared']):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                        f'R²={r2:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax3.set_xlabel('Battery Type', fontweight='bold')
            ax3.set_ylabel('Degradation Rate (%/cycle)', fontweight='bold')
            ax3.set_title('Capacity Degradation Rate Comparison', fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Performance variability over time
        ax4 = axes[1, 1]
        
        for i, battery_type in enumerate(battery_types):
            battery_data = comparative_data[comparative_data['Battery_Type'] == battery_type]
            
            # Calculate rolling standard deviation of performance
            window_size = min(25, len(battery_data) // 4)
            if window_size > 2:
                performance_std = battery_data['Performance_Score'].rolling(
                    window=window_size, center=True).std()
                
                color = self.battery_types_colors[i % len(self.battery_types_colors)]
                ax4.plot(battery_data['Cycle'], performance_std, linewidth=2, alpha=0.8,
                        color=color, label=battery_type)
        
        ax4.set_xlabel('Cycle Number', fontweight='bold')
        ax4.set_ylabel('Performance Variability (Std)', fontweight='bold')
        ax4.set_title('Performance Variability Evolution', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'degradation_comparison.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_interactive_comparison_dashboard(self, comparative_data: pd.DataFrame, save: bool = True):
        """
        인터랙티브 비교 대시보드 (Plotly 사용)
        
        Args:
            comparative_data: 비교 데이터
            save: 파일 저장 여부
        """
        logger.info("Creating interactive comparison dashboard")
        
        # Create subplot structure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Capacity Evolution', 'Performance by Phase', 
                           'Efficiency Distribution', '3D Performance Space'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "scene"}]]
        )
        
        battery_types = comparative_data['Battery_Type'].unique()
        colors = px.colors.qualitative.Set1[:len(battery_types)]
        
        # 1. Capacity Evolution
        for i, battery_type in enumerate(battery_types):
            battery_data = comparative_data[comparative_data['Battery_Type'] == battery_type]
            
            fig.add_trace(
                go.Scatter(
                    x=battery_data['Cycle'],
                    y=battery_data['Capacity_Discharge'],
                    mode='lines+markers',
                    name=f'{battery_type} Capacity',
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=4),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # 2. Performance by Lifecycle Phase
        for i, battery_type in enumerate(battery_types):
            battery_data = comparative_data[comparative_data['Battery_Type'] == battery_type]
            
            phase_performance = battery_data.groupby('Lifecycle_Phase')['Performance_Score'].mean().reset_index()
            
            fig.add_trace(
                go.Bar(
                    x=phase_performance['Lifecycle_Phase'],
                    y=phase_performance['Performance_Score'],
                    name=f'{battery_type} Performance',
                    marker_color=colors[i],
                    opacity=0.7,
                    showlegend=True
                ),
                row=1, col=2
            )
        
        # 3. Efficiency Distribution
        for i, battery_type in enumerate(battery_types):
            battery_data = comparative_data[comparative_data['Battery_Type'] == battery_type]
            efficiency_data = battery_data['Coulombic_Efficiency'] * 100
            
            fig.add_trace(
                go.Histogram(
                    x=efficiency_data,
                    name=f'{battery_type} Efficiency',
                    marker_color=colors[i],
                    opacity=0.7,
                    nbinsx=20,
                    showlegend=True
                ),
                row=2, col=1
            )
        
        # 4. 3D Performance Space
        for i, battery_type in enumerate(battery_types):
            battery_data = comparative_data[comparative_data['Battery_Type'] == battery_type]
            
            # Sample data for better performance
            sample_size = min(100, len(battery_data))
            sample_data = battery_data.sample(n=sample_size, random_state=42) if len(battery_data) > sample_size else battery_data
            
            fig.add_trace(
                go.Scatter3d(
                    x=sample_data['Energy_Density'],
                    y=sample_data['Power_Density'],
                    z=sample_data['Coulombic_Efficiency'],
                    mode='markers',
                    name=f'{battery_type} 3D',
                    marker=dict(
                        size=5,
                        color=colors[i],
                        opacity=0.7
                    ),
                    showlegend=True
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Battery Comparison Dashboard",
            title_x=0.5,
            title_font_size=16,
            height=800,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Cycle Number", row=1, col=1)
        fig.update_yaxes(title_text="Capacity (Ah)", row=1, col=1)
        
        fig.update_xaxes(title_text="Lifecycle Phase", row=1, col=2)
        fig.update_yaxes(title_text="Performance Score", row=1, col=2)
        
        fig.update_xaxes(title_text="Coulombic Efficiency (%)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        # Update 3D scene
        fig.update_scenes(
            xaxis_title="Energy Density",
            yaxis_title="Power Density", 
            zaxis_title="Efficiency",
            row=2, col=2
        )
        
        if save:
            save_path = self.output_dir / 'interactive_comparison_dashboard.html'
            fig.write_html(save_path)
            logger.info(f"Saved interactive dashboard: {save_path}")
        
        return fig
    
    def create_all_comparative_visualizations(self, data_sources: Union[pd.DataFrame, List[pd.DataFrame]], 
                                            labels: Optional[List[str]] = None):
        """
        모든 비교 시각화 생성
        
        Args:
            data_sources: 비교할 데이터소스
            labels: 배터리 타입 라벨
        """
        logger.info("Creating all comparative visualizations...")
        
        # Prepare comparative data
        comparative_data = self.prepare_comparative_data(data_sources, labels)
        
        # Create all comparison visualizations
        self.create_performance_radar_chart(comparative_data)
        self.create_benchmark_comparison(comparative_data)
        self.create_condition_based_comparison(comparative_data)
        self.create_degradation_comparison(comparative_data)
        self.create_interactive_comparison_dashboard(comparative_data)
        
        logger.info(f"All comparative visualizations saved to: {self.output_dir}")


def main():
    """메인 실행 함수"""
    print("Comparative Battery Visualizer")
    print("=" * 60)
    
    # Load data
    data_path = Path("analysis_output/processed_data.csv")
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please run battery_pattern_analyzer.py first.")
        return
    
    # Load data
    data = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded {len(data):,} data points")
    
    # Create comparative visualizer
    visualizer = ComparativeVisualizer()
    
    # For demonstration, create comparison with the same data
    # In practice, you would load different battery datasets
    
    # Split data by cycle ranges to simulate different batteries
    max_cycle = data['TotalCycle'].max()
    battery_1 = data[data['TotalCycle'] <= max_cycle // 3]
    battery_2 = data[(data['TotalCycle'] > max_cycle // 3) & (data['TotalCycle'] <= 2 * max_cycle // 3)]
    battery_3 = data[data['TotalCycle'] > 2 * max_cycle // 3]
    
    data_sources = [battery_1, battery_2, battery_3]
    labels = ['Battery_Early_Cycles', 'Battery_Mid_Cycles', 'Battery_Late_Cycles']
    
    # Create all visualizations
    visualizer.create_all_comparative_visualizations(data_sources, labels)
    
    print(f"\nComparative analysis completed!")
    print(f"Output directory: {visualizer.output_dir}")


if __name__ == "__main__":
    main()
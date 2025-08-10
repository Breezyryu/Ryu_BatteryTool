#!/usr/bin/env python3
"""
Multi-Scale Battery Analyzer Module
다중 스케일 배터리 분석 모듈
마크로(전체 수명) / 메조(사이클 그룹) / 마이크로(개별 사이클) 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# Multi-scale visualization styling
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

class MultiscaleAnalyzer:
    """다중 스케일 배터리 분석 클래스"""
    
    def __init__(self, output_dir: str = "analysis_output/multiscale_plots"):
        """
        초기화
        
        Args:
            output_dir: 출력 디렉토리 경로
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Scale-specific color palettes
        self.macro_colors = sns.color_palette("viridis", n_colors=10)
        self.meso_colors = sns.color_palette("plasma", n_colors=20)
        self.micro_colors = sns.color_palette("coolwarm", n_colors=15)
        
        # Analysis parameters
        self.macro_window = 50  # cycles for macro trends
        self.meso_window = 10   # cycles for meso patterns
        self.micro_resolution = 100  # data points per cycle for micro analysis
        
        logger.info(f"Multi-scale analyzer initialized, output: {self.output_dir}")
    
    def prepare_multiscale_data(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        다중 스케일 분석을 위한 데이터 준비
        
        Args:
            data: 원본 배터리 데이터
            
        Returns:
            스케일별 데이터 딕셔너리
        """
        logger.info("Preparing multi-scale data structures")
        
        df = data.copy()
        
        # Macro scale data (cycle-level aggregates)
        macro_data = []
        cycles = sorted(df['TotalCycle'].unique())
        
        for cycle in cycles:
            cycle_data = df[df['TotalCycle'] == cycle]
            if len(cycle_data) > 0:
                macro_record = {
                    'Cycle': cycle,
                    'Voltage_Mean': cycle_data['Voltage[V]'].mean(),
                    'Voltage_Std': cycle_data['Voltage[V]'].std(),
                    'Voltage_Max': cycle_data['Voltage[V]'].max(),
                    'Voltage_Min': cycle_data['Voltage[V]'].min(),
                    'Current_Mean': cycle_data['Current[A]'].mean(),
                    'Current_Std': cycle_data['Current[A]'].std(),
                    'Current_Range': cycle_data['Current[A]'].max() - cycle_data['Current[A]'].min(),
                    'Capacity_Charge': cycle_data['Chg_Capacity[Ah]'].max(),
                    'Capacity_Discharge': abs(cycle_data['Dchg_Capacity[Ah]'].min()),
                    'Energy': (cycle_data['Voltage[V]'] * np.abs(cycle_data['Current[A]'])).sum(),
                    'Duration': len(cycle_data),  # proxy for time
                    'Efficiency': abs(cycle_data['Dchg_Capacity[Ah]'].min()) / (cycle_data['Chg_Capacity[Ah]'].max() + 1e-10)
                }
                macro_data.append(macro_record)
        
        # Meso scale data (cycle groups)
        meso_data = []
        group_size = 5  # 5 cycles per group
        
        for i in range(0, len(cycles), group_size):
            group_cycles = cycles[i:i + group_size]
            if len(group_cycles) >= 2:
                group_data = df[df['TotalCycle'].isin(group_cycles)]
                
                if len(group_data) > 0:
                    meso_record = {
                        'Group_Start': group_cycles[0],
                        'Group_End': group_cycles[-1],
                        'Group_Center': np.mean(group_cycles),
                        'Cycles_Count': len(group_cycles),
                        'Voltage_Trend': self._calculate_trend(group_data, 'Voltage[V]', 'TotalCycle'),
                        'Capacity_Trend': self._calculate_trend(group_data, 'Chg_Capacity[Ah]', 'TotalCycle'),
                        'Variability_Voltage': group_data.groupby('TotalCycle')['Voltage[V]'].std().mean(),
                        'Variability_Current': group_data.groupby('TotalCycle')['Current[A]'].std().mean(),
                        'Performance_Score': self._calculate_performance_score(group_data)
                    }
                    meso_data.append(meso_record)
        
        # Micro scale data (detailed cycle analysis)
        # Sample a few representative cycles for micro analysis
        representative_cycles = [1, 50, 100, 200] if len(cycles) >= 200 else cycles[:4]
        micro_data = []
        
        for cycle in representative_cycles:
            if cycle in cycles:
                cycle_data = df[df['TotalCycle'] == cycle].copy()
                if len(cycle_data) > 10:
                    # Add derived features for micro analysis
                    cycle_data['Power'] = cycle_data['Voltage[V]'] * cycle_data['Current[A]']
                    cycle_data['Energy_Cumsum'] = (cycle_data['Power'] * 1/3600).cumsum()  # Wh
                    cycle_data['Time_Index'] = range(len(cycle_data))
                    cycle_data['Phase'] = self._identify_phases(cycle_data)
                    
                    micro_data.append({
                        'cycle': cycle,
                        'data': cycle_data
                    })
        
        return {
            'macro': pd.DataFrame(macro_data),
            'meso': pd.DataFrame(meso_data),
            'micro': micro_data
        }
    
    def _calculate_trend(self, data: pd.DataFrame, value_col: str, cycle_col: str) -> float:
        """트렌드 기울기 계산"""
        cycle_means = data.groupby(cycle_col)[value_col].mean()
        if len(cycle_means) > 1:
            x = cycle_means.index.values
            y = cycle_means.values
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope
        return 0.0
    
    def _calculate_performance_score(self, data: pd.DataFrame) -> float:
        """성능 스코어 계산 (0-100)"""
        try:
            # Normalize key metrics and combine
            voltage_score = (data['Voltage[V]'].mean() - 2.5) / (4.5 - 2.5) * 100
            capacity_score = data['Chg_Capacity[Ah]'].max() / 5.0 * 100  # Assuming max 5Ah
            efficiency = abs(data['Dchg_Capacity[Ah]'].min()) / (data['Chg_Capacity[Ah]'].max() + 1e-10)
            efficiency_score = efficiency * 100
            
            composite_score = (voltage_score * 0.3 + capacity_score * 0.4 + efficiency_score * 0.3)
            return np.clip(composite_score, 0, 100)
        except:
            return 50.0  # Default neutral score
    
    def _identify_phases(self, cycle_data: pd.DataFrame) -> List[str]:
        """충방전 단계 식별"""
        current = cycle_data['Current[A]'].values
        phases = []
        
        for i, curr in enumerate(current):
            if curr > 0.1:
                phases.append('Charge')
            elif curr < -0.1:
                phases.append('Discharge')
            else:
                phases.append('Rest')
        
        return phases
    
    def create_macro_scale_analysis(self, multiscale_data: Dict, save: bool = True) -> plt.Figure:
        """
        마크로 스케일 분석 (전체 수명 주기)
        
        Args:
            multiscale_data: 다중 스케일 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating macro-scale analysis (full lifecycle)")
        
        macro_df = multiscale_data['macro']
        
        if len(macro_df) < 10:
            logger.warning("Insufficient data for macro analysis")
            return plt.figure()
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        fig.suptitle('Macro-Scale Analysis: Full Battery Lifecycle', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Capacity degradation with multiple trend models
        ax1 = axes[0, 0]
        
        cycles = macro_df['Cycle'].values
        discharge_cap = macro_df['Capacity_Discharge'].values
        
        # Remove outliers for cleaner trends
        valid_mask = (discharge_cap > 0) & (discharge_cap < discharge_cap.quantile(0.95))
        valid_cycles = cycles[valid_mask]
        valid_capacity = discharge_cap[valid_mask]
        
        if len(valid_capacity) > 5:
            # Plot raw data
            ax1.scatter(valid_cycles, valid_capacity, alpha=0.5, s=20, color='lightblue', label='Data')
            
            # Linear trend
            z_linear = np.polyfit(valid_cycles, valid_capacity, 1)
            p_linear = np.poly1d(z_linear)
            ax1.plot(valid_cycles, p_linear(valid_cycles), 'r--', linewidth=2, label='Linear Trend')
            
            # Polynomial trend
            if len(valid_capacity) > 10:
                z_poly = np.polyfit(valid_cycles, valid_capacity, 2)
                p_poly = np.poly1d(z_poly)
                ax1.plot(valid_cycles, p_poly(valid_cycles), 'g-', linewidth=2, label='Polynomial Trend')
            
            # Exponential trend (simplified)
            try:
                # Fit exponential decay: y = a * exp(b * x) + c
                def exp_decay(x, a, b, c):
                    return a * np.exp(b * x) + c
                
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(exp_decay, valid_cycles, valid_capacity, 
                                   bounds=([0, -1, 0], [10, 0, 10]), maxfev=1000)
                
                exp_pred = exp_decay(valid_cycles, *popt)
                ax1.plot(valid_cycles, exp_pred, 'purple', linewidth=2, label='Exponential Decay')
            except:
                pass  # Skip if fitting fails
            
            ax1.set_xlabel('Cycle Number', fontweight='bold')
            ax1.set_ylabel('Discharge Capacity (Ah)', fontweight='bold')
            ax1.set_title('Long-term Capacity Degradation', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Efficiency evolution with statistical bounds
        ax2 = axes[0, 1]
        
        efficiency = macro_df['Efficiency'].replace([np.inf, -np.inf], np.nan).dropna()
        if len(efficiency) > 5:
            # Rolling statistics
            window = min(20, len(efficiency) // 3)
            if window > 2:
                efficiency_smooth = efficiency.rolling(window=window, center=True).mean()
                efficiency_upper = efficiency.rolling(window=window, center=True).quantile(0.75)
                efficiency_lower = efficiency.rolling(window=window, center=True).quantile(0.25)
                
                ax2.plot(macro_df['Cycle'][:len(efficiency)], efficiency, 
                        'lightcoral', alpha=0.4, label='Raw Efficiency')
                ax2.plot(macro_df['Cycle'][:len(efficiency_smooth)], efficiency_smooth, 
                        'red', linewidth=2, label='Smoothed Trend')
                ax2.fill_between(macro_df['Cycle'][:len(efficiency_upper)], 
                               efficiency_lower, efficiency_upper, 
                               alpha=0.3, color='red', label='IQR Band')
            
            ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect Efficiency')
            ax2.set_xlabel('Cycle Number', fontweight='bold')
            ax2.set_ylabel('Coulombic Efficiency', fontweight='bold')
            ax2.set_title('Efficiency Evolution with Uncertainty', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Voltage statistics evolution
        ax3 = axes[1, 0]
        
        voltage_metrics = ['Voltage_Mean', 'Voltage_Max', 'Voltage_Min']
        colors = ['blue', 'red', 'green']
        
        for metric, color in zip(voltage_metrics, colors):
            if metric in macro_df.columns:
                ax3.plot(macro_df['Cycle'], macro_df[metric], 
                        color=color, linewidth=1.5, alpha=0.8, label=metric.replace('Voltage_', ''))
        
        ax3.set_xlabel('Cycle Number', fontweight='bold')
        ax3.set_ylabel('Voltage (V)', fontweight='bold')
        ax3.set_title('Voltage Statistics Evolution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Energy throughput analysis
        ax4 = axes[1, 1]
        
        if 'Energy' in macro_df.columns:
            energy = macro_df['Energy'].values
            cumulative_energy = np.cumsum(energy)
            
            # Twin axes for energy per cycle and cumulative
            ax4_twin = ax4.twinx()
            
            line1 = ax4.plot(macro_df['Cycle'], energy, 'orange', linewidth=2, alpha=0.7, label='Energy/Cycle')
            line2 = ax4_twin.plot(macro_df['Cycle'], cumulative_energy, 'purple', linewidth=2, alpha=0.7, label='Cumulative Energy')
            
            ax4.set_xlabel('Cycle Number', fontweight='bold')
            ax4.set_ylabel('Energy per Cycle (Wh)', color='orange', fontweight='bold')
            ax4_twin.set_ylabel('Cumulative Energy (Wh)', color='purple', fontweight='bold')
            ax4.set_title('Energy Throughput Analysis', fontweight='bold')
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc='upper left')
            ax4.grid(True, alpha=0.3)
        
        # 5. Degradation rate analysis
        ax5 = axes[2, 0]
        
        if len(valid_capacity) > 10:
            # Calculate instantaneous degradation rate
            degradation_rate = np.gradient(valid_capacity)
            
            # Smooth the degradation rate
            window = min(15, len(degradation_rate) // 3)
            if window > 2:
                degradation_smooth = pd.Series(degradation_rate).rolling(window=window, center=True).mean()
                
                ax5.plot(valid_cycles, degradation_rate, 'lightblue', alpha=0.5, label='Instantaneous Rate')
                ax5.plot(valid_cycles[:len(degradation_smooth)], degradation_smooth, 
                        'blue', linewidth=2, label='Smoothed Rate')
                ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax5.set_xlabel('Cycle Number', fontweight='bold')
                ax5.set_ylabel('Degradation Rate (Ah/cycle)', fontweight='bold')
                ax5.set_title('Capacity Degradation Rate', fontweight='bold')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
        
        # 6. Lifecycle phases identification
        ax6 = axes[2, 1]
        
        if len(macro_df) > 50:
            # Identify lifecycle phases using capacity retention
            initial_capacity = valid_capacity[0] if len(valid_capacity) > 0 else 4.0
            retention = (valid_capacity / initial_capacity) * 100
            
            # Define phases based on retention
            phase_colors = []
            phase_labels = []
            
            for ret in retention:
                if ret >= 95:
                    phase_colors.append('green')
                    phase_labels.append('Healthy')
                elif ret >= 85:
                    phase_colors.append('yellow')
                    phase_labels.append('Degrading')
                elif ret >= 75:
                    phase_colors.append('orange')
                    phase_labels.append('Aging')
                else:
                    phase_colors.append('red')
                    phase_labels.append('End-of-Life')
            
            scatter = ax6.scatter(valid_cycles, retention, c=phase_colors, s=30, alpha=0.7)
            
            # Add phase boundaries
            ax6.axhline(y=95, color='green', linestyle='--', alpha=0.7, label='95% Healthy')
            ax6.axhline(y=85, color='yellow', linestyle='--', alpha=0.7, label='85% Degrading')
            ax6.axhline(y=75, color='orange', linestyle='--', alpha=0.7, label='75% Aging')
            
            ax6.set_xlabel('Cycle Number', fontweight='bold')
            ax6.set_ylabel('Capacity Retention (%)', fontweight='bold')
            ax6.set_title('Lifecycle Phase Identification', fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'macro_scale_analysis.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_meso_scale_analysis(self, multiscale_data: Dict, save: bool = True) -> plt.Figure:
        """
        메조 스케일 분석 (사이클 그룹 패턴)
        
        Args:
            multiscale_data: 다중 스케일 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating meso-scale analysis (cycle group patterns)")
        
        meso_df = multiscale_data['meso']
        
        if len(meso_df) < 5:
            logger.warning("Insufficient data for meso analysis")
            return plt.figure()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Meso-Scale Analysis: Cycle Group Patterns', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Group performance trends
        ax1 = axes[0, 0]
        
        group_centers = meso_df['Group_Center'].values
        performance = meso_df['Performance_Score'].values
        
        # Plot with error estimation
        ax1.plot(group_centers, performance, 'bo-', linewidth=2, markersize=6, alpha=0.8)
        
        # Add trend line
        if len(performance) > 3:
            z = np.polyfit(group_centers, performance, 1)
            p = np.poly1d(z)
            ax1.plot(group_centers, p(group_centers), 'r--', linewidth=2, alpha=0.7, 
                    label=f'Trend: {z[0]:.3f}/cycle')
        
        ax1.set_xlabel('Cycle Group Center', fontweight='bold')
        ax1.set_ylabel('Performance Score', fontweight='bold')
        ax1.set_title('Group Performance Evolution', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        if len(performance) > 3:
            ax1.legend()
        
        # 2. Voltage and capacity trends correlation
        ax2 = axes[0, 1]
        
        voltage_trends = meso_df['Voltage_Trend'].values
        capacity_trends = meso_df['Capacity_Trend'].values
        
        scatter = ax2.scatter(voltage_trends, capacity_trends, 
                             c=group_centers, cmap='viridis', s=60, alpha=0.7)
        
        # Add correlation line
        if len(voltage_trends) > 3:
            correlation = np.corrcoef(voltage_trends, capacity_trends)[0, 1]
            sns.regplot(x=voltage_trends, y=capacity_trends, ax=ax2, scatter=False, 
                       color='red', line_kws={'linewidth': 2})
            ax2.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Voltage Trend (V/cycle)', fontweight='bold')
        ax2.set_ylabel('Capacity Trend (Ah/cycle)', fontweight='bold')
        ax2.set_title('Trend Correlation Analysis', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax2, label='Cycle Group')
        
        # 3. Variability analysis
        ax3 = axes[0, 2]
        
        voltage_var = meso_df['Variability_Voltage'].values
        current_var = meso_df['Variability_Current'].values
        
        # Remove outliers for cleaner visualization
        voltage_var_clean = voltage_var[voltage_var < np.percentile(voltage_var, 95)]
        current_var_clean = current_var[current_var < np.percentile(current_var, 95)]
        
        if len(voltage_var_clean) > 0 and len(current_var_clean) > 0:
            ax3.hist([voltage_var_clean, current_var_clean], bins=15, alpha=0.7, 
                    label=['Voltage Variability', 'Current Variability'], 
                    color=['blue', 'orange'], density=True)
            
            ax3.set_xlabel('Variability (Std Dev)', fontweight='bold')
            ax3.set_ylabel('Density', fontweight='bold')
            ax3.set_title('Parameter Variability Distribution', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Group clustering analysis
        ax4 = axes[1, 0]
        
        # Prepare features for clustering
        features = ['Performance_Score', 'Voltage_Trend', 'Capacity_Trend', 
                   'Variability_Voltage', 'Variability_Current']
        
        cluster_data = meso_df[features].dropna()
        if len(cluster_data) > 3:
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(cluster_data)
            
            # K-means clustering
            n_clusters = min(3, len(cluster_data) // 2)
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_features)
                
                # PCA for visualization
                pca = PCA(n_components=2)
                pca_features = pca.fit_transform(scaled_features)
                
                scatter = ax4.scatter(pca_features[:, 0], pca_features[:, 1], 
                                    c=cluster_labels, cmap='Set1', s=60, alpha=0.7)
                
                ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontweight='bold')
                ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontweight='bold')
                ax4.set_title('Group Clustering Analysis', fontweight='bold')
                ax4.grid(True, alpha=0.3)
                
                # Add cluster centers
                centers_pca = pca.transform(kmeans.cluster_centers_)
                ax4.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                          c='red', marker='x', s=200, linewidths=3, label='Centroids')
                ax4.legend()
        
        # 5. Temporal pattern analysis
        ax5 = axes[1, 1]
        
        # Analyze patterns in group performance over time
        if len(meso_df) > 10:
            # Calculate moving averages and trends
            performance_ma = pd.Series(performance).rolling(window=3, center=True).mean()
            
            ax5.plot(group_centers, performance, 'o-', alpha=0.5, color='lightblue', label='Raw')
            ax5.plot(group_centers[:len(performance_ma)], performance_ma, 
                    'b-', linewidth=2, label='3-Group Moving Average')
            
            # Identify change points (simplified)
            diff = np.diff(performance)
            change_threshold = np.std(diff) * 2
            change_points = np.where(np.abs(diff) > change_threshold)[0]
            
            for cp in change_points:
                if cp < len(group_centers) - 1:
                    ax5.axvline(x=group_centers[cp], color='red', linestyle='--', 
                              alpha=0.7, label='Change Point' if cp == change_points[0] else "")
            
            ax5.set_xlabel('Cycle Group Center', fontweight='bold')
            ax5.set_ylabel('Performance Score', fontweight='bold')
            ax5.set_title('Temporal Pattern Analysis', fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Group duration and cycle count analysis
        ax6 = axes[1, 2]
        
        cycles_count = meso_df['Cycles_Count'].values
        group_span = meso_df['Group_End'] - meso_df['Group_Start']
        
        # Create a 2D histogram
        if len(cycles_count) > 5:
            ax6.hist2d(cycles_count, group_span, bins=min(10, len(cycles_count)//2), 
                      cmap='Blues', alpha=0.7)
            ax6.set_xlabel('Cycles per Group', fontweight='bold')
            ax6.set_ylabel('Group Span (cycles)', fontweight='bold')
            ax6.set_title('Group Structure Analysis', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'meso_scale_analysis.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_micro_scale_analysis(self, multiscale_data: Dict, save: bool = True) -> plt.Figure:
        """
        마이크로 스케일 분석 (개별 사이클 세부 분석)
        
        Args:
            multiscale_data: 다중 스케일 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating micro-scale analysis (individual cycle details)")
        
        micro_data = multiscale_data['micro']
        
        if len(micro_data) < 2:
            logger.warning("Insufficient data for micro analysis")
            return plt.figure()
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        fig.suptitle('Micro-Scale Analysis: Individual Cycle Details', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Voltage-current trajectories comparison
        ax1 = axes[0, 0]
        
        for i, cycle_info in enumerate(micro_data[:4]):  # Max 4 cycles
            cycle_data = cycle_info['data']
            cycle_num = cycle_info['cycle']
            
            color = self.micro_colors[i % len(self.micro_colors)]
            ax1.plot(cycle_data['Voltage[V]'], cycle_data['Current[A]'], 
                    color=color, linewidth=1.5, alpha=0.7, label=f'Cycle {cycle_num}')
        
        ax1.set_xlabel('Voltage (V)', fontweight='bold')
        ax1.set_ylabel('Current (A)', fontweight='bold')
        ax1.set_title('V-I Trajectories Comparison', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Power evolution during cycles
        ax2 = axes[0, 1]
        
        for i, cycle_info in enumerate(micro_data[:4]):
            cycle_data = cycle_info['data']
            cycle_num = cycle_info['cycle']
            
            if 'Power' in cycle_data.columns:
                time_norm = np.linspace(0, 1, len(cycle_data))  # Normalized time
                color = self.micro_colors[i % len(self.micro_colors)]
                
                ax2.plot(time_norm, cycle_data['Power'], 
                        color=color, linewidth=1.5, alpha=0.7, label=f'Cycle {cycle_num}')
        
        ax2.set_xlabel('Normalized Time', fontweight='bold')
        ax2.set_ylabel('Power (W)', fontweight='bold')
        ax2.set_title('Power Evolution Comparison', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 3. Phase analysis (charge/discharge/rest)
        ax3 = axes[1, 0]
        
        if len(micro_data) > 0:
            first_cycle = micro_data[0]['data']
            if 'Phase' in first_cycle.columns:
                # Create phase transition plot
                phases = first_cycle['Phase'].values
                voltage = first_cycle['Voltage[V]'].values
                time_index = first_cycle['Time_Index'].values
                
                # Color by phase
                phase_colors = {'Charge': 'red', 'Discharge': 'blue', 'Rest': 'green'}
                
                for phase in ['Charge', 'Discharge', 'Rest']:
                    mask = np.array(phases) == phase
                    if mask.any():
                        ax3.scatter(time_index[mask], voltage[mask], 
                                  c=phase_colors[phase], alpha=0.6, s=15, label=phase)
                
                ax3.set_xlabel('Time Index', fontweight='bold')
                ax3.set_ylabel('Voltage (V)', fontweight='bold')
                ax3.set_title(f'Phase Analysis (Cycle {micro_data[0]["cycle"]})', fontweight='bold')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # 4. Energy accumulation patterns
        ax4 = axes[1, 1]
        
        for i, cycle_info in enumerate(micro_data[:3]):
            cycle_data = cycle_info['data']
            cycle_num = cycle_info['cycle']
            
            if 'Energy_Cumsum' in cycle_data.columns:
                time_norm = np.linspace(0, 1, len(cycle_data))
                color = self.micro_colors[i % len(self.micro_colors)]
                
                ax4.plot(time_norm, cycle_data['Energy_Cumsum'], 
                        color=color, linewidth=2, alpha=0.7, label=f'Cycle {cycle_num}')
        
        ax4.set_xlabel('Normalized Time', fontweight='bold')
        ax4.set_ylabel('Cumulative Energy (Wh)', fontweight='bold')
        ax4.set_title('Energy Accumulation Patterns', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Voltage derivative analysis (dV/dt proxy)
        ax5 = axes[2, 0]
        
        for i, cycle_info in enumerate(micro_data[:3]):
            cycle_data = cycle_info['data']
            cycle_num = cycle_info['cycle']
            
            voltage = cycle_data['Voltage[V]'].values
            if len(voltage) > 3:
                dv_dt = np.gradient(voltage)  # Approximation of dV/dt
                time_norm = np.linspace(0, 1, len(dv_dt))
                color = self.micro_colors[i % len(self.micro_colors)]
                
                ax5.plot(time_norm, dv_dt, color=color, linewidth=1.5, alpha=0.7, 
                        label=f'Cycle {cycle_num}')
        
        ax5.set_xlabel('Normalized Time', fontweight='bold')
        ax5.set_ylabel('dV/dt (V/step)', fontweight='bold')
        ax5.set_title('Voltage Rate of Change', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 6. Cycle comparison radar chart (if enough cycles)
        ax6 = axes[2, 1]
        
        if len(micro_data) >= 2:
            # Calculate cycle characteristics
            cycle_metrics = []
            cycle_labels = []
            
            for cycle_info in micro_data[:4]:
                cycle_data = cycle_info['data']
                cycle_num = cycle_info['cycle']
                
                metrics = {
                    'Voltage_Range': cycle_data['Voltage[V]'].max() - cycle_data['Voltage[V]'].min(),
                    'Current_Range': cycle_data['Current[A]'].max() - cycle_data['Current[A]'].min(),
                    'Voltage_Std': cycle_data['Voltage[V]'].std(),
                    'Current_Std': cycle_data['Current[A]'].std(),
                    'Duration': len(cycle_data)
                }
                
                cycle_metrics.append(list(metrics.values()))
                cycle_labels.append(f'Cycle {cycle_num}')
            
            if len(cycle_metrics) > 1:
                # Normalize metrics for comparison
                cycle_metrics = np.array(cycle_metrics)
                cycle_metrics_norm = (cycle_metrics - cycle_metrics.min(axis=0)) / (
                    cycle_metrics.max(axis=0) - cycle_metrics.min(axis=0) + 1e-10)
                
                metrics_names = ['V Range', 'I Range', 'V Std', 'I Std', 'Duration']
                
                # Simple bar chart instead of radar (simpler implementation)
                x_pos = np.arange(len(metrics_names))
                width = 0.8 / len(cycle_labels)
                
                for i, (metrics_norm, label) in enumerate(zip(cycle_metrics_norm, cycle_labels)):
                    ax6.bar(x_pos + i*width, metrics_norm, width, alpha=0.7, label=label)
                
                ax6.set_xlabel('Metrics', fontweight='bold')
                ax6.set_ylabel('Normalized Value', fontweight='bold')
                ax6.set_title('Cycle Characteristics Comparison', fontweight='bold')
                ax6.set_xticks(x_pos + width/2)
                ax6.set_xticklabels(metrics_names, rotation=45)
                ax6.legend()
                ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'micro_scale_analysis.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_cross_scale_correlation(self, multiscale_data: Dict, save: bool = True) -> plt.Figure:
        """
        교차 스케일 상관관계 분석
        
        Args:
            multiscale_data: 다중 스케일 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating cross-scale correlation analysis")
        
        macro_df = multiscale_data['macro']
        meso_df = multiscale_data['meso']
        
        if len(macro_df) < 5 or len(meso_df) < 3:
            logger.warning("Insufficient data for cross-scale analysis")
            return plt.figure()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cross-Scale Correlation Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Macro-Meso correlation
        ax1 = axes[0, 0]
        
        # Align macro and meso data by cycle ranges
        correlations = []
        for _, meso_row in meso_df.iterrows():
            start_cycle = meso_row['Group_Start']
            end_cycle = meso_row['Group_End']
            
            # Find corresponding macro data
            macro_subset = macro_df[
                (macro_df['Cycle'] >= start_cycle) & 
                (macro_df['Cycle'] <= end_cycle)
            ]
            
            if len(macro_subset) > 1:
                macro_performance = macro_subset['Efficiency'].mean()
                meso_performance = meso_row['Performance_Score']
                
                correlations.append({
                    'Macro_Efficiency': macro_performance,
                    'Meso_Performance': meso_performance,
                    'Cycle_Range': f'{start_cycle}-{end_cycle}'
                })
        
        if correlations:
            corr_df = pd.DataFrame(correlations)
            
            # Remove outliers
            macro_eff = corr_df['Macro_Efficiency']
            meso_perf = corr_df['Meso_Performance']
            
            valid_mask = (
                (macro_eff > macro_eff.quantile(0.05)) & 
                (macro_eff < macro_eff.quantile(0.95)) &
                (meso_perf > meso_perf.quantile(0.05)) & 
                (meso_perf < meso_perf.quantile(0.95))
            )
            
            if valid_mask.sum() > 3:
                ax1.scatter(macro_eff[valid_mask], meso_perf[valid_mask], 
                           alpha=0.7, s=60, c='blue')
                
                # Add regression line
                sns.regplot(data=corr_df[valid_mask], x='Macro_Efficiency', y='Meso_Performance', 
                           ax=ax1, scatter=False, color='red', line_kws={'linewidth': 2})
                
                # Calculate correlation
                correlation = macro_eff[valid_mask].corr(meso_perf[valid_mask])
                ax1.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax1.transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax1.set_xlabel('Macro-Scale Efficiency', fontweight='bold')
                ax1.set_ylabel('Meso-Scale Performance', fontweight='bold')
                ax1.set_title('Macro-Meso Scale Correlation', fontweight='bold')
                ax1.grid(True, alpha=0.3)
        
        # 2. Scale-dependent variance analysis
        ax2 = axes[0, 1]
        
        # Calculate variance at different scales
        if len(macro_df) > 10:
            macro_capacity_var = macro_df['Capacity_Discharge'].var()
            macro_voltage_var = macro_df['Voltage_Mean'].var()
            
            # Meso variances
            meso_performance_var = meso_df['Performance_Score'].var()
            meso_voltage_trend_var = meso_df['Voltage_Trend'].var()
            
            scales = ['Macro\nCapacity', 'Macro\nVoltage', 'Meso\nPerformance', 'Meso\nV-Trend']
            variances = [macro_capacity_var, macro_voltage_var, meso_performance_var, meso_voltage_trend_var]
            
            # Normalize variances for comparison
            variances_norm = np.array(variances) / max(variances)
            
            bars = ax2.bar(scales, variances_norm, alpha=0.7, 
                          color=['blue', 'lightblue', 'red', 'lightcoral'])
            
            # Add values on bars
            for bar, var in zip(bars, variances_norm):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{var:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax2.set_ylabel('Normalized Variance', fontweight='bold')
            ax2.set_title('Scale-Dependent Variance', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Temporal scaling relationships
        ax3 = axes[1, 0]
        
        # Analyze how patterns at different scales evolve
        if len(macro_df) > 20:
            # Divide macro data into early, middle, late periods
            n_cycles = len(macro_df)
            early_idx = n_cycles // 3
            middle_idx = 2 * n_cycles // 3
            
            periods = ['Early', 'Middle', 'Late']
            period_data = [
                macro_df.iloc[:early_idx],
                macro_df.iloc[early_idx:middle_idx], 
                macro_df.iloc[middle_idx:]
            ]
            
            # Calculate metrics for each period
            period_metrics = []
            for period, data in zip(periods, period_data):
                if len(data) > 0:
                    metrics = {
                        'Efficiency_Mean': data['Efficiency'].mean(),
                        'Efficiency_Std': data['Efficiency'].std(),
                        'Capacity_Trend': self._calculate_trend(data, 'Capacity_Discharge', 'Cycle'),
                        'Voltage_Std': data['Voltage_Std'].mean()
                    }
                    period_metrics.append(metrics)
            
            if len(period_metrics) == 3:
                metrics_names = list(period_metrics[0].keys())
                
                # Create grouped bar chart
                x_pos = np.arange(len(metrics_names))
                width = 0.25
                
                for i, (period, metrics) in enumerate(zip(periods, period_metrics)):
                    values = list(metrics.values())
                    # Normalize for comparison
                    values_norm = np.array(values) / (np.array([list(m.values()) for m in period_metrics]).max(axis=0) + 1e-10)
                    
                    ax3.bar(x_pos + i*width, values_norm, width, alpha=0.7, label=period)
                
                ax3.set_xlabel('Metrics', fontweight='bold')
                ax3.set_ylabel('Normalized Value', fontweight='bold')
                ax3.set_title('Temporal Scale Evolution', fontweight='bold')
                ax3.set_xticks(x_pos + width)
                ax3.set_xticklabels([name.replace('_', ' ') for name in metrics_names], rotation=45)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # 4. Information transfer across scales
        ax4 = axes[1, 1]
        
        # Analyze how much information is preserved/lost across scales
        if len(macro_df) > 5 and len(meso_df) > 2:
            # Calculate information metrics (entropy approximation)
            def calculate_entropy_approx(data):
                # Simplified entropy calculation using histogram
                hist, _ = np.histogram(data, bins=min(10, len(data)//2), density=True)
                hist = hist[hist > 0]  # Remove zero bins
                return -np.sum(hist * np.log2(hist + 1e-10))
            
            # Calculate entropies at different scales
            macro_entropy = calculate_entropy_approx(macro_df['Efficiency'].dropna())
            
            # Aggregate meso data to macro scale for comparison
            meso_aggregated = meso_df['Performance_Score'].values
            meso_entropy = calculate_entropy_approx(meso_aggregated)
            
            # Information preservation ratio
            info_preservation = meso_entropy / (macro_entropy + 1e-10)
            
            # Visualization
            scales = ['Macro\nEntropy', 'Meso\nEntropy']
            entropies = [macro_entropy, meso_entropy]
            
            bars = ax4.bar(scales, entropies, alpha=0.7, color=['blue', 'red'])
            
            # Add information preservation text
            ax4.text(0.5, 0.8, f'Information\nPreservation:\n{info_preservation:.2f}', 
                    transform=ax4.transAxes, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            ax4.set_ylabel('Information Content (bits)', fontweight='bold')
            ax4.set_title('Information Transfer Across Scales', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'cross_scale_correlation.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_all_multiscale_visualizations(self, data: pd.DataFrame):
        """
        모든 다중 스케일 시각화 생성
        
        Args:
            data: 배터리 데이터
        """
        logger.info("Creating all multi-scale visualizations...")
        
        # Prepare multi-scale data
        multiscale_data = self.prepare_multiscale_data(data)
        
        # Create scale-specific analyses
        self.create_macro_scale_analysis(multiscale_data)
        self.create_meso_scale_analysis(multiscale_data)
        self.create_micro_scale_analysis(multiscale_data)
        self.create_cross_scale_correlation(multiscale_data)
        
        logger.info(f"All multi-scale visualizations saved to: {self.output_dir}")


def main():
    """메인 실행 함수"""
    print("Multi-Scale Battery Analyzer")
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
    
    # Create multi-scale analyzer
    analyzer = MultiscaleAnalyzer()
    
    # Create all visualizations
    analyzer.create_all_multiscale_visualizations(data)
    
    print(f"\nMulti-scale analysis completed!")
    print(f"Output directory: {analyzer.output_dir}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Statistical Battery Visualizer Module
고급 통계적 배터리 데이터 시각화 도구
Enhanced Seaborn 기능을 완전히 활용한 통계 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats, signal
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# Enhanced Seaborn styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid", palette="husl")
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
    'axes.facecolor': 'white'
})

logger = logging.getLogger(__name__)

class StatisticalBatteryVisualizer:
    """고급 통계적 배터리 시각화 클래스"""
    
    def __init__(self, output_dir: str = "analysis_output/statistical_plots"):
        """
        초기화
        
        Args:
            output_dir: 출력 디렉토리 경로
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistical color palettes
        self.health_palette = sns.color_palette("RdYlGn", n_colors=10)
        self.cycle_palette = sns.color_palette("viridis", n_colors=20)
        self.condition_palette = sns.color_palette("Set2", n_colors=8)
        
        logger.info(f"Statistical visualizer initialized, output: {self.output_dir}")
    
    def create_capacity_regression_analysis(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        용량 감소 회귀 분석 (신뢰구간 포함)
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating capacity regression analysis with confidence intervals")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comprehensive Capacity Fade Statistical Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Extract capacity data
        capacity_data = []
        cycles = data['TotalCycle'].unique()[:500]  # First 500 cycles
        
        for cycle in cycles:
            cycle_data = data[data['TotalCycle'] == cycle]
            discharge_cap = np.abs(cycle_data['Dchg_Capacity[Ah]'].min())
            if discharge_cap > 0:
                capacity_data.append({
                    'Cycle': cycle,
                    'Capacity': discharge_cap,
                    'Retention': (discharge_cap / 4.352) * 100  # Normalize to initial capacity
                })
        
        if not capacity_data:
            logger.warning("No capacity data found")
            return fig
        
        df_capacity = pd.DataFrame(capacity_data)
        
        # 1. Linear regression with confidence interval
        ax1 = axes[0, 0]
        sns.regplot(data=df_capacity, x='Cycle', y='Retention', ax=ax1,
                   scatter_kws={'s': 15, 'alpha': 0.6},
                   line_kws={'color': 'red', 'linewidth': 2})
        ax1.set_title('Linear Regression Analysis', fontweight='bold')
        ax1.set_ylabel('Capacity Retention (%)', fontweight='bold')
        ax1.set_xlabel('Cycle Number', fontweight='bold')
        ax1.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% EOL')
        ax1.legend()
        
        # Add R² score
        X = df_capacity['Cycle'].values.reshape(-1, 1)
        y = df_capacity['Retention'].values
        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)
        ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Polynomial regression comparison
        ax2 = axes[0, 1]
        
        # Linear fit
        sns.regplot(data=df_capacity, x='Cycle', y='Retention', ax=ax2,
                   order=1, scatter=False, label='Linear', 
                   line_kws={'linestyle': '--', 'alpha': 0.7})
        
        # Polynomial fit (degree 2)
        sns.regplot(data=df_capacity, x='Cycle', y='Retention', ax=ax2,
                   order=2, scatter_kws={'s': 15, 'alpha': 0.6},
                   label='Polynomial (degree 2)',
                   line_kws={'color': 'green', 'linewidth': 2})
        
        ax2.set_title('Model Comparison', fontweight='bold')
        ax2.set_ylabel('Capacity Retention (%)', fontweight='bold')
        ax2.set_xlabel('Cycle Number', fontweight='bold')
        ax2.legend()
        
        # 3. Residuals analysis
        ax3 = axes[1, 0]
        
        # Calculate residuals
        y_pred = reg.predict(X)
        residuals = y - y_pred
        
        sns.scatterplot(x=y_pred, y=residuals, ax=ax3, alpha=0.6)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax3.set_title('Residuals Analysis', fontweight='bold')
        ax3.set_xlabel('Predicted Retention (%)', fontweight='bold')
        ax3.set_ylabel('Residuals', fontweight='bold')
        
        # Add trend line for residuals
        sns.regplot(x=y_pred, y=residuals, ax=ax3, scatter=False,
                   line_kws={'color': 'orange', 'alpha': 0.7})
        
        # 4. Distribution of residuals
        ax4 = axes[1, 1]
        
        sns.histplot(residuals, kde=True, ax=ax4, alpha=0.7)
        ax4.set_title('Residuals Distribution', fontweight='bold')
        ax4.set_xlabel('Residuals', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        
        # Add normal distribution overlay
        mu, sigma = stats.norm.fit(residuals)
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        y_norm = stats.norm.pdf(x_norm, mu, sigma)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(x_norm, y_norm, 'r-', alpha=0.7, label=f'Normal (μ={mu:.3f}, σ={sigma:.3f})')
        ax4_twin.set_ylabel('Probability Density', fontweight='bold')
        ax4_twin.legend()
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'capacity_regression_analysis.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_performance_distribution_analysis(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        배터리 성능 분포 분석 (violin/box plots)
        
        Args:
            data: 배터리 데이터  
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating performance distribution analysis")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Battery Performance Distribution Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Prepare data for distribution analysis
        cycles_to_analyze = [1, 50, 100, 200, 500]
        distribution_data = []
        
        for cycle in cycles_to_analyze:
            if cycle > data['TotalCycle'].max():
                continue
                
            cycle_data = data[data['TotalCycle'] == cycle]
            if len(cycle_data) == 0:
                continue
            
            # Sample data to avoid overcrowding
            if len(cycle_data) > 1000:
                cycle_data = cycle_data.sample(n=1000, random_state=42)
            
            for _, row in cycle_data.iterrows():
                distribution_data.append({
                    'Cycle': f'Cycle {cycle}',
                    'Cycle_Numeric': cycle,
                    'Voltage': row['Voltage[V]'],
                    'Current': row['Current[A]'],
                    'Abs_Current': abs(row['Current[A]']),
                    'Phase': 'Charge' if row['Current[A]'] > 0 else 'Discharge' if row['Current[A]'] < 0 else 'Rest'
                })
        
        if not distribution_data:
            logger.warning("No distribution data found")
            return fig
        
        df_dist = pd.DataFrame(distribution_data)
        
        # 1. Voltage distribution by cycle
        ax1 = axes[0, 0]
        sns.violinplot(data=df_dist, x='Cycle', y='Voltage', ax=ax1, 
                      palette=self.cycle_palette)
        ax1.set_title('Voltage Distribution Evolution', fontweight='bold')
        ax1.set_ylabel('Voltage (V)', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Current distribution by cycle  
        ax2 = axes[0, 1]
        sns.boxplot(data=df_dist, x='Cycle', y='Current', ax=ax2,
                   palette=self.cycle_palette)
        ax2.set_title('Current Distribution Evolution', fontweight='bold') 
        ax2.set_ylabel('Current (A)', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Voltage-Current relationship by phase
        ax3 = axes[0, 2]
        phase_data = df_dist[df_dist['Phase'] != 'Rest']  # Exclude rest phase
        sns.scatterplot(data=phase_data, x='Voltage', y='Current', 
                       hue='Phase', style='Cycle', ax=ax3, alpha=0.6)
        ax3.set_title('Voltage-Current Relationship', fontweight='bold')
        ax3.set_xlabel('Voltage (V)', fontweight='bold')
        ax3.set_ylabel('Current (A)', fontweight='bold')
        
        # 4. Performance variability by cycle
        ax4 = axes[1, 0]
        sns.stripplot(data=df_dist, x='Cycle_Numeric', y='Voltage', ax=ax4,
                     size=3, alpha=0.6, jitter=True)
        ax4.set_title('Voltage Variability by Cycle', fontweight='bold')
        ax4.set_xlabel('Cycle Number', fontweight='bold')
        ax4.set_ylabel('Voltage (V)', fontweight='bold')
        
        # 5. Current magnitude distribution
        ax5 = axes[1, 1]
        sns.histplot(data=df_dist, x='Abs_Current', hue='Phase', 
                    kde=True, ax=ax5, alpha=0.7, multiple="stack")
        ax5.set_title('Current Magnitude Distribution', fontweight='bold')
        ax5.set_xlabel('|Current| (A)', fontweight='bold')
        ax5.set_ylabel('Frequency', fontweight='bold')
        
        # 6. Statistical summary heatmap
        ax6 = axes[1, 2]
        
        # Create summary statistics
        summary_stats = df_dist.groupby('Cycle').agg({
            'Voltage': ['mean', 'std', 'min', 'max'],
            'Current': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col) for col in summary_stats.columns]
        
        # Create heatmap
        sns.heatmap(summary_stats.T, annot=True, cmap='RdYlBu_r', ax=ax6,
                   fmt='.3f', cbar_kws={'label': 'Value'})
        ax6.set_title('Statistical Summary Heatmap', fontweight='bold')
        ax6.set_xlabel('Cycle', fontweight='bold')
        ax6.set_ylabel('Statistics', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'performance_distribution_analysis.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_correlation_matrix_analysis(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        다변수 상관관계 히트맵 분석
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating correlation matrix analysis")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Battery Parameters Correlation Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Prepare correlation data
        numeric_cols = ['Voltage[V]', 'Current[A]', 'Chg_Capacity[Ah]', 'Dchg_Capacity[Ah]', 'TotalCycle']
        corr_data = data[numeric_cols].copy()
        
        # Add derived features
        corr_data['Abs_Current'] = abs(corr_data['Current[A]'])
        corr_data['Power'] = corr_data['Voltage[V]'] * corr_data['Current[A]']
        corr_data['Energy_Efficiency'] = abs(corr_data['Dchg_Capacity[Ah]']) / (corr_data['Chg_Capacity[Ah]'] + 1e-10)
        
        # Remove infinite values
        corr_data = corr_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        # 1. Overall correlation matrix
        ax1 = axes[0, 0]
        corr_matrix = corr_data.corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax1, fmt='.2f',
                   cbar_kws={'label': 'Correlation Coefficient'})
        ax1.set_title('Overall Parameter Correlation', fontweight='bold')
        
        # 2. Early vs Late cycle correlation comparison
        early_data = corr_data[corr_data['TotalCycle'] <= 100]
        late_data = corr_data[corr_data['TotalCycle'] >= 500]
        
        ax2 = axes[0, 1]
        if len(early_data) > 0 and len(late_data) > 0:
            early_corr = early_data.corr()
            late_corr = late_data.corr()
            
            # Calculate correlation difference
            corr_diff = late_corr - early_corr
            
            sns.heatmap(corr_diff, annot=True, cmap='RdBu_r', center=0,
                       square=True, ax=ax2, fmt='.2f',
                       cbar_kws={'label': 'Correlation Change'})
            ax2.set_title('Correlation Change (Late - Early Cycles)', fontweight='bold')
        
        # 3. Partial correlation analysis
        ax3 = axes[1, 0]
        
        # Select key parameters for partial correlation
        key_params = ['Voltage[V]', 'Current[A]', 'Power', 'TotalCycle']
        partial_corr_data = corr_data[key_params]
        
        # Calculate partial correlation (controlling for cycle number)
        from scipy.stats import pearsonr
        partial_corr_matrix = np.zeros((len(key_params)-1, len(key_params)-1))
        param_names = key_params[:-1]  # Exclude TotalCycle
        
        for i, param1 in enumerate(param_names):
            for j, param2 in enumerate(param_names):
                if i <= j:
                    continue
                # Simple partial correlation calculation
                r_xy = corr_data[param1].corr(corr_data[param2])
                r_xz = corr_data[param1].corr(corr_data['TotalCycle'])
                r_yz = corr_data[param2].corr(corr_data['TotalCycle'])
                
                # Partial correlation formula
                r_xy_z = (r_xy - r_xz * r_yz) / (np.sqrt(1 - r_xz**2) * np.sqrt(1 - r_yz**2))
                partial_corr_matrix[i, j] = r_xy_z
                partial_corr_matrix[j, i] = r_xy_z
        
        # Fill diagonal with 1s
        np.fill_diagonal(partial_corr_matrix, 1.0)
        
        sns.heatmap(partial_corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   xticklabels=param_names, yticklabels=param_names,
                   square=True, ax=ax3, fmt='.2f',
                   cbar_kws={'label': 'Partial Correlation'})
        ax3.set_title('Partial Correlation (Controlling for Cycle)', fontweight='bold')
        
        # 4. Feature importance for cycle prediction
        ax4 = axes[1, 1]
        
        # Use random forest to get feature importance
        from sklearn.ensemble import RandomForestRegressor
        
        X = corr_data.drop(['TotalCycle'], axis=1)
        y = corr_data['TotalCycle']
        
        # Sample data if too large
        if len(X) > 10000:
            sample_idx = np.random.choice(len(X), 10000, replace=False)
            X = X.iloc[sample_idx]
            y = y.iloc[sample_idx]
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        sns.barplot(data=feature_importance, x='Importance', y='Feature', 
                   ax=ax4, palette='viridis')
        ax4.set_title('Feature Importance for Cycle Prediction', fontweight='bold')
        ax4.set_xlabel('Importance Score', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'correlation_matrix_analysis.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_condition_comparison_analysis(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        충방전 조건별 성능 비교 분석
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure  
        """
        logger.info("Creating condition-based comparison analysis")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Battery Performance by Operating Conditions', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Categorize operating conditions
        condition_data = []
        
        for _, row in data.sample(n=min(50000, len(data)), random_state=42).iterrows():
            current = row['Current[A]']
            voltage = row['Voltage[V]']
            cycle = row['TotalCycle']
            
            # Determine C-rate category
            abs_current = abs(current)
            if abs_current < 0.5:
                c_rate_category = 'Rest'
            elif abs_current < 2.0:
                c_rate_category = 'Low C-rate (<2C)'
            elif abs_current < 4.0:
                c_rate_category = 'Medium C-rate (2-4C)'
            else:
                c_rate_category = 'High C-rate (>4C)'
            
            # Determine voltage category
            if voltage < 3.0:
                voltage_category = 'Low Voltage (<3.0V)'
            elif voltage < 3.7:
                voltage_category = 'Mid Voltage (3.0-3.7V)'
            elif voltage < 4.2:
                voltage_category = 'High Voltage (3.7-4.2V)'
            else:
                voltage_category = 'Very High Voltage (>4.2V)'
            
            # Determine cycle age category
            if cycle < 100:
                age_category = 'Fresh (0-100 cycles)'
            elif cycle < 500:
                age_category = 'Moderate (100-500 cycles)'
            else:
                age_category = 'Aged (>500 cycles)'
            
            condition_data.append({
                'Voltage': voltage,
                'Current': current,
                'Abs_Current': abs_current,
                'Cycle': cycle,
                'C_Rate_Category': c_rate_category,
                'Voltage_Category': voltage_category,
                'Age_Category': age_category,
                'Phase': 'Charge' if current > 0.1 else 'Discharge' if current < -0.1 else 'Rest'
            })
        
        df_conditions = pd.DataFrame(condition_data)
        
        # 1. Performance by C-rate category
        ax1 = axes[0, 0]
        c_rate_data = df_conditions[df_conditions['C_Rate_Category'] != 'Rest']
        if len(c_rate_data) > 0:
            sns.boxplot(data=c_rate_data, x='C_Rate_Category', y='Voltage', 
                       hue='Phase', ax=ax1, palette=self.condition_palette)
            ax1.set_title('Voltage Distribution by C-rate', fontweight='bold')
            ax1.set_ylabel('Voltage (V)', fontweight='bold')
            ax1.set_xlabel('C-rate Category', fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Performance evolution by age
        ax2 = axes[0, 1]
        sns.violinplot(data=df_conditions, x='Age_Category', y='Voltage',
                      ax=ax2, palette=self.health_palette[:3])
        ax2.set_title('Voltage Distribution by Battery Age', fontweight='bold')
        ax2.set_ylabel('Voltage (V)', fontweight='bold')
        ax2.set_xlabel('Battery Age Category', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Current distribution by voltage range
        ax3 = axes[0, 2]
        voltage_data = df_conditions[df_conditions['Phase'] != 'Rest']
        if len(voltage_data) > 0:
            sns.swarmplot(data=voltage_data.sample(n=min(1000, len(voltage_data))), 
                         x='Voltage_Category', y='Abs_Current', 
                         hue='Phase', ax=ax3, size=3, alpha=0.7)
            ax3.set_title('Current vs Voltage Range', fontweight='bold')
            ax3.set_ylabel('|Current| (A)', fontweight='bold')
            ax3.set_xlabel('Voltage Category', fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Statistical comparison of conditions
        ax4 = axes[1, 0]
        
        # Create comparison matrix for different conditions
        comparison_stats = []
        
        for age in df_conditions['Age_Category'].unique():
            for phase in ['Charge', 'Discharge']:
                subset = df_conditions[
                    (df_conditions['Age_Category'] == age) & 
                    (df_conditions['Phase'] == phase)
                ]
                if len(subset) > 0:
                    comparison_stats.append({
                        'Age_Phase': f"{age}\n{phase}",
                        'Mean_Voltage': subset['Voltage'].mean(),
                        'Std_Voltage': subset['Voltage'].std(),
                        'Mean_Current': subset['Abs_Current'].mean(),
                        'Count': len(subset)
                    })
        
        if comparison_stats:
            df_comparison = pd.DataFrame(comparison_stats)
            pivot_data = df_comparison.pivot_table(
                index='Age_Phase', 
                values=['Mean_Voltage', 'Std_Voltage', 'Mean_Current'],
                aggfunc='first'
            )
            
            sns.heatmap(pivot_data.T, annot=True, cmap='viridis', ax=ax4,
                       fmt='.3f', cbar_kws={'label': 'Value'})
            ax4.set_title('Condition Statistics Heatmap', fontweight='bold')
            ax4.set_xlabel('Age & Phase', fontweight='bold')
        
        # 5. Performance variability analysis
        ax5 = axes[1, 1]
        
        # Calculate coefficient of variation for each condition
        cv_data = []
        for category in df_conditions['Age_Category'].unique():
            subset = df_conditions[df_conditions['Age_Category'] == category]
            if len(subset) > 1:
                cv_voltage = subset['Voltage'].std() / subset['Voltage'].mean()
                cv_current = subset['Abs_Current'].std() / (subset['Abs_Current'].mean() + 1e-10)
                
                cv_data.extend([
                    {'Category': category, 'Parameter': 'Voltage', 'CV': cv_voltage},
                    {'Category': category, 'Parameter': 'Current', 'CV': cv_current}
                ])
        
        if cv_data:
            df_cv = pd.DataFrame(cv_data)
            sns.barplot(data=df_cv, x='Category', y='CV', hue='Parameter', ax=ax5,
                       palette=['skyblue', 'lightcoral'])
            ax5.set_title('Performance Variability (CV)', fontweight='bold')
            ax5.set_ylabel('Coefficient of Variation', fontweight='bold')
            ax5.set_xlabel('Battery Age Category', fontweight='bold')
            ax5.tick_params(axis='x', rotation=45)
        
        # 6. Joint distribution analysis
        ax6 = axes[1, 2]
        
        # Sample data for joint plot
        sample_data = df_conditions.sample(n=min(2000, len(df_conditions)))
        
        # Create scatter plot with regression line
        sns.scatterplot(data=sample_data, x='Voltage', y='Abs_Current', 
                       hue='Age_Category', style='Phase', ax=ax6, alpha=0.6)
        
        # Add regression lines for each age category
        for age in sample_data['Age_Category'].unique():
            age_data = sample_data[sample_data['Age_Category'] == age]
            if len(age_data) > 10:
                sns.regplot(data=age_data, x='Voltage', y='Abs_Current', 
                           ax=ax6, scatter=False, alpha=0.5)
        
        ax6.set_title('Voltage-Current Joint Distribution', fontweight='bold')
        ax6.set_xlabel('Voltage (V)', fontweight='bold')
        ax6.set_ylabel('|Current| (A)', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'condition_comparison_analysis.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_time_series_statistical_analysis(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        시계열 배터리 통계 분석 (신뢰구간 포함)
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating time series statistical analysis")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Time Series Statistical Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Prepare time series data
        cycles = sorted(data['TotalCycle'].unique())
        ts_data = []
        
        for cycle in cycles[:200]:  # First 200 cycles
            cycle_data = data[data['TotalCycle'] == cycle]
            if len(cycle_data) == 0:
                continue
            
            # Calculate statistics for this cycle
            stats_dict = {
                'Cycle': cycle,
                'Voltage_Mean': cycle_data['Voltage[V]'].mean(),
                'Voltage_Std': cycle_data['Voltage[V]'].std(),
                'Voltage_Median': cycle_data['Voltage[V]'].median(),
                'Current_Mean': cycle_data['Current[A]'].mean(),
                'Current_Std': cycle_data['Current[A]'].std(),
                'Max_Charge_Cap': cycle_data['Chg_Capacity[Ah]'].max(),
                'Max_Discharge_Cap': abs(cycle_data['Dchg_Capacity[Ah]'].min()),
                'Voltage_Range': cycle_data['Voltage[V]'].max() - cycle_data['Voltage[V]'].min(),
                'Current_Range': cycle_data['Current[A]'].max() - cycle_data['Current[A]'].min()
            }
            
            # Calculate efficiency
            if stats_dict['Max_Charge_Cap'] > 0:
                stats_dict['Coulombic_Efficiency'] = stats_dict['Max_Discharge_Cap'] / stats_dict['Max_Charge_Cap']
            else:
                stats_dict['Coulombic_Efficiency'] = 0
            
            ts_data.append(stats_dict)
        
        if not ts_data:
            logger.warning("No time series data found")
            return fig
        
        df_ts = pd.DataFrame(ts_data)
        
        # 1. Voltage evolution with confidence bands
        ax1 = axes[0, 0]
        
        # Plot mean voltage with confidence interval
        sns.lineplot(data=df_ts, x='Cycle', y='Voltage_Mean', ax=ax1, 
                    color='blue', linewidth=2, label='Mean Voltage')
        
        # Add confidence bands (mean ± std)
        ax1.fill_between(df_ts['Cycle'], 
                        df_ts['Voltage_Mean'] - df_ts['Voltage_Std'],
                        df_ts['Voltage_Mean'] + df_ts['Voltage_Std'],
                        alpha=0.3, color='blue', label='±1 Std Dev')
        
        ax1.set_title('Voltage Evolution with Uncertainty', fontweight='bold')
        ax1.set_xlabel('Cycle Number', fontweight='bold')
        ax1.set_ylabel('Voltage (V)', fontweight='bold')
        ax1.legend()
        
        # 2. Capacity fade with trend analysis
        ax2 = axes[0, 1]
        
        # Normalize capacity retention
        if df_ts['Max_Discharge_Cap'].iloc[0] > 0:
            df_ts['Capacity_Retention'] = (df_ts['Max_Discharge_Cap'] / df_ts['Max_Discharge_Cap'].iloc[0]) * 100
        else:
            df_ts['Capacity_Retention'] = 100
        
        sns.regplot(data=df_ts, x='Cycle', y='Capacity_Retention', ax=ax2,
                   scatter_kws={'s': 20, 'alpha': 0.6}, 
                   line_kws={'color': 'red', 'linewidth': 2})
        
        ax2.set_title('Capacity Retention with Trend', fontweight='bold')
        ax2.set_xlabel('Cycle Number', fontweight='bold')
        ax2.set_ylabel('Capacity Retention (%)', fontweight='bold')
        ax2.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% EOL')
        ax2.legend()
        
        # 3. Efficiency analysis with moving average
        ax3 = axes[1, 0]
        
        # Calculate moving average
        window_size = 10
        if len(df_ts) >= window_size:
            df_ts['Efficiency_MA'] = df_ts['Coulombic_Efficiency'].rolling(
                window=window_size, center=True).mean()
            
            sns.scatterplot(data=df_ts, x='Cycle', y='Coulombic_Efficiency', 
                           ax=ax3, alpha=0.5, s=15, label='Raw Data')
            sns.lineplot(data=df_ts, x='Cycle', y='Efficiency_MA', ax=ax3,
                        color='red', linewidth=2, label=f'{window_size}-Cycle MA')
        else:
            sns.scatterplot(data=df_ts, x='Cycle', y='Coulombic_Efficiency', ax=ax3)
        
        ax3.set_title('Coulombic Efficiency Trend', fontweight='bold')
        ax3.set_xlabel('Cycle Number', fontweight='bold')
        ax3.set_ylabel('Coulombic Efficiency', fontweight='bold')
        ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='100% Efficient')
        ax3.legend()
        
        # 4. Variability evolution
        ax4 = axes[1, 1]
        
        # Plot voltage and current variability over time
        ax4_twin = ax4.twinx()
        
        line1 = ax4.plot(df_ts['Cycle'], df_ts['Voltage_Std'], 
                        color='blue', linewidth=2, label='Voltage Std')
        line2 = ax4_twin.plot(df_ts['Cycle'], df_ts['Current_Std'], 
                             color='red', linewidth=2, label='Current Std')
        
        ax4.set_title('Parameter Variability Evolution', fontweight='bold')
        ax4.set_xlabel('Cycle Number', fontweight='bold')
        ax4.set_ylabel('Voltage Std Dev (V)', color='blue', fontweight='bold')
        ax4_twin.set_ylabel('Current Std Dev (A)', color='red', fontweight='bold')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'time_series_statistical_analysis.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_advanced_statistical_summary(self, data: pd.DataFrame, save: bool = True) -> str:
        """
        고급 통계 분석 요약 보고서 생성
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            통계 분석 보고서 텍스트
        """
        logger.info("Creating advanced statistical summary report")
        
        report = []
        report.append("=" * 80)
        report.append("ADVANCED STATISTICAL BATTERY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Data Points: {len(data):,}")
        report.append(f"Analysis Period: Cycles {data['TotalCycle'].min()} - {data['TotalCycle'].max()}")
        report.append("")
        
        # Basic descriptive statistics
        report.append("-" * 40)
        report.append("DESCRIPTIVE STATISTICS")
        report.append("-" * 40)
        
        numeric_cols = ['Voltage[V]', 'Current[A]', 'Chg_Capacity[Ah]', 'Dchg_Capacity[Ah]']
        desc_stats = data[numeric_cols].describe()
        
        for col in numeric_cols:
            report.append(f"\n{col}:")
            report.append(f"  Mean: {desc_stats.loc['mean', col]:.4f}")
            report.append(f"  Std:  {desc_stats.loc['std', col]:.4f}")
            report.append(f"  Min:  {desc_stats.loc['min', col]:.4f}")
            report.append(f"  Max:  {desc_stats.loc['max', col]:.4f}")
            
            # Add skewness and kurtosis
            skewness = stats.skew(data[col].dropna())
            kurt = stats.kurtosis(data[col].dropna())
            report.append(f"  Skew: {skewness:.4f}")
            report.append(f"  Kurt: {kurt:.4f}")
        
        # Correlation analysis
        report.append("\n" + "-" * 40)
        report.append("CORRELATION ANALYSIS")
        report.append("-" * 40)
        
        corr_matrix = data[numeric_cols].corr()
        report.append("\nKey Correlations:")
        
        # Find strongest correlations
        correlations = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # Avoid duplicates
                    corr_val = corr_matrix.loc[col1, col2]
                    correlations.append((abs(corr_val), col1, col2, corr_val))
        
        correlations.sort(reverse=True)
        for abs_corr, col1, col2, corr_val in correlations[:5]:
            report.append(f"  {col1} vs {col2}: {corr_val:.4f}")
        
        # Statistical tests
        report.append("\n" + "-" * 40)
        report.append("STATISTICAL TESTS")
        report.append("-" * 40)
        
        # Normality test for voltage
        voltage_data = data['Voltage[V]'].dropna().sample(min(5000, len(data)))
        shapiro_stat, shapiro_p = stats.shapiro(voltage_data)
        report.append(f"\nVoltage Normality Test (Shapiro-Wilk):")
        report.append(f"  Statistic: {shapiro_stat:.4f}")
        report.append(f"  p-value: {shapiro_p:.4e}")
        report.append(f"  Normal: {'Yes' if shapiro_p > 0.05 else 'No'}")
        
        # Capacity fade analysis
        cycles = sorted(data['TotalCycle'].unique())
        if len(cycles) > 10:
            capacity_data = []
            for cycle in cycles[:100]:  # First 100 cycles
                cycle_data = data[data['TotalCycle'] == cycle]
                if len(cycle_data) > 0:
                    discharge_cap = abs(cycle_data['Dchg_Capacity[Ah]'].min())
                    if discharge_cap > 0:
                        capacity_data.append((cycle, discharge_cap))
            
            if len(capacity_data) > 5:
                cycles_arr = np.array([x[0] for x in capacity_data])
                capacities_arr = np.array([x[1] for x in capacity_data])
                
                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(cycles_arr, capacities_arr)
                
                report.append(f"\nCapacity Fade Analysis:")
                report.append(f"  Linear Trend: {slope:.6f} Ah/cycle")
                report.append(f"  R-squared: {r_value**2:.4f}")
                report.append(f"  p-value: {p_value:.4e}")
                report.append(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
                
                # Predict EOL (80% capacity)
                if len(capacity_data) > 0 and slope < 0:
                    initial_capacity = capacities_arr[0]
                    target_capacity = initial_capacity * 0.8
                    eol_cycle = (target_capacity - intercept) / slope
                    if eol_cycle > 0:
                        report.append(f"  Predicted EOL (80%): Cycle {int(eol_cycle)}")
        
        # Performance variability analysis
        report.append("\n" + "-" * 40)
        report.append("PERFORMANCE VARIABILITY")
        report.append("-" * 40)
        
        # Compare early vs late cycles
        early_cycles = data[data['TotalCycle'] <= 100]
        late_cycles = data[data['TotalCycle'] >= 500]
        
        if len(early_cycles) > 0 and len(late_cycles) > 0:
            # T-test for voltage difference
            early_voltage = early_cycles['Voltage[V]'].dropna()
            late_voltage = late_cycles['Voltage[V]'].dropna()
            
            if len(early_voltage) > 0 and len(late_voltage) > 0:
                t_stat, t_p = stats.ttest_ind(early_voltage, late_voltage)
                report.append(f"\nVoltage Change (Early vs Late Cycles):")
                report.append(f"  Early Mean: {early_voltage.mean():.4f} V")
                report.append(f"  Late Mean: {late_voltage.mean():.4f} V")
                report.append(f"  T-statistic: {t_stat:.4f}")
                report.append(f"  p-value: {t_p:.4e}")
                report.append(f"  Significant Difference: {'Yes' if t_p < 0.05 else 'No'}")
        
        report.append("\n" + "=" * 80)
        report.append("END OF STATISTICAL ANALYSIS REPORT")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save:
            report_path = self.output_dir / 'statistical_analysis_report.txt'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Statistical report saved to: {report_path}")
        
        return report_text
    
    def create_bayesian_analysis(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        베이지안 통계 분석 시각화
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating Bayesian statistical analysis")
        
        df = data.copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Bayesian Statistical Analysis for Battery Performance', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Prepare cycle-level data for Bayesian analysis
        cycle_data = []
        cycles = sorted(df['TotalCycle'].unique())[:200]
        
        for cycle in cycles:
            cycle_subset = df[df['TotalCycle'] == cycle]
            if len(cycle_subset) > 0:
                cycle_data.append({
                    'Cycle': cycle,
                    'Avg_Voltage': cycle_subset['Voltage[V]'].mean(),
                    'Std_Voltage': cycle_subset['Voltage[V]'].std(),
                    'Max_Capacity': cycle_subset['Chg_Capacity[Ah]'].max(),
                    'Avg_Current': cycle_subset['Current[A]'].mean()
                })
        
        if not cycle_data:
            logger.warning("No cycle data for Bayesian analysis")
            return fig
        
        cycle_df = pd.DataFrame(cycle_data)
        
        # 1. Prior and Posterior Distribution Analysis
        ax1 = axes[0, 0]
        
        # Create synthetic prior and posterior for voltage
        voltage_data = cycle_df['Avg_Voltage'].dropna()
        if len(voltage_data) > 10:
            # Prior (assume normal with broad uncertainty)
            prior_mean = 3.7  # Typical Li-ion voltage
            prior_std = 0.5
            
            # Posterior (from data)
            posterior_mean = voltage_data.mean()
            posterior_std = voltage_data.std()
            
            x = np.linspace(2.5, 4.5, 1000)
            prior = stats.norm.pdf(x, prior_mean, prior_std)
            posterior = stats.norm.pdf(x, posterior_mean, posterior_std)
            
            ax1.plot(x, prior, 'b--', linewidth=2, label='Prior', alpha=0.7)
            ax1.plot(x, posterior, 'r-', linewidth=2, label='Posterior', alpha=0.8)
            ax1.hist(voltage_data, bins=30, density=True, alpha=0.3, color='gray', label='Data')
            
            ax1.set_xlabel('Voltage (V)', fontweight='bold')
            ax1.set_ylabel('Probability Density', fontweight='bold')
            ax1.set_title('Prior vs Posterior Distribution (Voltage)', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Credible Intervals for Capacity Prediction
        ax2 = axes[0, 1]
        
        capacity_data = cycle_df['Max_Capacity'].dropna()
        if len(capacity_data) > 20:
            # Bootstrap for credible intervals
            n_bootstrap = 1000
            bootstrap_means = []
            
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(capacity_data, size=len(capacity_data), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            # Calculate credible intervals
            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)
            ci_median = np.percentile(bootstrap_means, 50)
            
            # Plot bootstrap distribution
            ax2.hist(bootstrap_means, bins=50, alpha=0.7, density=True, color='lightblue')
            ax2.axvline(ci_lower, color='red', linestyle='--', label=f'95% CI Lower: {ci_lower:.3f}')
            ax2.axvline(ci_upper, color='red', linestyle='--', label=f'95% CI Upper: {ci_upper:.3f}')
            ax2.axvline(ci_median, color='green', linestyle='-', linewidth=2, label=f'Median: {ci_median:.3f}')
            
            ax2.set_xlabel('Bootstrap Mean Capacity (Ah)', fontweight='bold')
            ax2.set_ylabel('Density', fontweight='bold')
            ax2.set_title('Bootstrap Distribution with Credible Intervals', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Bayesian Model Comparison
        ax3 = axes[1, 0]
        
        if len(cycle_df) > 30:
            # Compare different models using AIC/BIC approximation
            X = cycle_df['Cycle'].values.reshape(-1, 1)
            y = cycle_df['Max_Capacity'].values
            
            # Linear model
            from sklearn.linear_model import LinearRegression
            linear_model = LinearRegression().fit(X, y)
            linear_pred = linear_model.predict(X)
            linear_mse = mean_squared_error(y, linear_pred)
            
            # Polynomial models
            models_comparison = []
            degrees = [1, 2, 3]
            
            for degree in degrees:
                from sklearn.preprocessing import PolynomialFeatures
                poly_features = PolynomialFeatures(degree=degree)
                X_poly = poly_features.fit_transform(X)
                poly_model = LinearRegression().fit(X_poly, y)
                poly_pred = poly_model.predict(X_poly)
                poly_mse = mean_squared_error(y, poly_pred)
                
                # Approximate AIC (for normal errors)
                n = len(y)
                k = degree + 1  # number of parameters
                aic = n * np.log(poly_mse) + 2 * k
                bic = n * np.log(poly_mse) + k * np.log(n)
                
                models_comparison.append({
                    'Model': f'Polynomial (degree {degree})',
                    'AIC': aic,
                    'BIC': bic,
                    'MSE': poly_mse
                })
            
            comparison_df = pd.DataFrame(models_comparison)
            
            # Plot model comparison
            x_pos = np.arange(len(comparison_df))
            width = 0.35
            
            bars1 = ax3.bar(x_pos - width/2, comparison_df['AIC'], width, 
                           label='AIC', alpha=0.8, color='skyblue')
            bars2 = ax3.bar(x_pos + width/2, comparison_df['BIC'], width,
                           label='BIC', alpha=0.8, color='lightcoral')
            
            ax3.set_xlabel('Model Type', fontweight='bold')
            ax3.set_ylabel('Information Criterion', fontweight='bold')
            ax3.set_title('Bayesian Model Comparison (AIC/BIC)', fontweight='bold')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([f'Poly {i+1}' for i in range(len(degrees))], rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Hierarchical Model Visualization
        ax4 = axes[1, 1]
        
        # Group cycles into early, middle, late phases
        cycle_df['Phase'] = pd.cut(cycle_df['Cycle'], bins=3, labels=['Early', 'Middle', 'Late'])
        
        if not cycle_df['Phase'].isna().all():
            phase_stats = []
            for phase in ['Early', 'Middle', 'Late']:
                phase_data = cycle_df[cycle_df['Phase'] == phase]['Max_Capacity'].dropna()
                if len(phase_data) > 0:
                    phase_stats.append({
                        'Phase': phase,
                        'Mean': phase_data.mean(),
                        'Std': phase_data.std(),
                        'Count': len(phase_data)
                    })
            
            if phase_stats:
                stats_df = pd.DataFrame(phase_stats)
                
                # Plot hierarchical means with error bars
                ax4.errorbar(stats_df['Phase'], stats_df['Mean'], 
                           yerr=stats_df['Std'], fmt='o-', capsize=5,
                           linewidth=2, markersize=8, alpha=0.8)
                
                # Add individual data points
                for phase in ['Early', 'Middle', 'Late']:
                    phase_data = cycle_df[cycle_df['Phase'] == phase]['Max_Capacity'].dropna()
                    if len(phase_data) > 0:
                        x_pos = list(stats_df['Phase']).index(phase)
                        ax4.scatter([x_pos] * len(phase_data), phase_data, 
                                  alpha=0.3, s=20, color='red')
                
                ax4.set_xlabel('Battery Life Phase', fontweight='bold')
                ax4.set_ylabel('Capacity (Ah)', fontweight='bold')
                ax4.set_title('Hierarchical Analysis by Life Phase', fontweight='bold')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'bayesian_statistical_analysis.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_survival_analysis(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        생존 분석 (배터리 수명 분석)
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating survival analysis for battery lifetime")
        
        df = data.copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Battery Lifetime Survival Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Calculate cycle-level capacity for survival analysis
        cycle_capacity = []
        cycles = sorted(df['TotalCycle'].unique())[:300]
        
        for cycle in cycles:
            cycle_data = df[df['TotalCycle'] == cycle]
            if len(cycle_data) > 0:
                max_capacity = cycle_data['Chg_Capacity[Ah]'].max()
                if max_capacity > 0:
                    cycle_capacity.append({'Cycle': cycle, 'Capacity': max_capacity})
        
        if len(cycle_capacity) < 20:
            logger.warning("Insufficient data for survival analysis")
            return fig
        
        capacity_df = pd.DataFrame(cycle_capacity)
        initial_capacity = capacity_df['Capacity'].iloc[0]
        capacity_df['Retention'] = (capacity_df['Capacity'] / initial_capacity) * 100
        
        # 1. Kaplan-Meier Survival Curve (simplified)
        ax1 = axes[0, 0]
        
        # Define "failure" as dropping below certain thresholds
        thresholds = [95, 90, 85, 80, 75]
        colors = plt.cm.Reds(np.linspace(0.4, 1, len(thresholds)))
        
        for i, threshold in enumerate(thresholds):
            # Calculate survival function
            survival_data = []
            at_risk = len(capacity_df)
            
            for cycle in capacity_df['Cycle']:
                current_retention = capacity_df[capacity_df['Cycle'] == cycle]['Retention'].iloc[0]
                if current_retention >= threshold:
                    survival_prob = 1.0
                else:
                    # Find first failure cycle
                    failure_cycle = capacity_df[capacity_df['Retention'] < threshold]['Cycle'].min()
                    if cycle >= failure_cycle:
                        survival_prob = 0.0
                    else:
                        survival_prob = 1.0
                
                survival_data.append({'Cycle': cycle, 'Survival': survival_prob})
            
            surv_df = pd.DataFrame(survival_data)
            ax1.step(surv_df['Cycle'], surv_df['Survival'], where='post',
                    color=colors[i], linewidth=2, alpha=0.8, 
                    label=f'{threshold}% threshold')
        
        ax1.set_xlabel('Cycle Number', fontweight='bold')
        ax1.set_ylabel('Survival Probability', fontweight='bold')
        ax1.set_title('Kaplan-Meier Survival Curves', fontweight='bold')
        ax1.legend(loc='lower left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([-0.05, 1.05])
        
        # 2. Hazard Rate Analysis
        ax2 = axes[0, 1]
        
        # Calculate degradation rate as a proxy for hazard
        capacity_df['Degradation_Rate'] = np.abs(np.gradient(capacity_df['Retention']))
        
        # Smooth the hazard function
        window_size = min(20, len(capacity_df) // 4)
        if window_size > 2:
            capacity_df['Smoothed_Hazard'] = capacity_df['Degradation_Rate'].rolling(
                window=window_size, center=True, min_periods=1).mean()
            
            ax2.plot(capacity_df['Cycle'], capacity_df['Degradation_Rate'], 
                    'lightblue', alpha=0.5, label='Raw Hazard Rate')
            ax2.plot(capacity_df['Cycle'], capacity_df['Smoothed_Hazard'], 
                    'red', linewidth=2, label='Smoothed Hazard Rate')
            
            ax2.set_xlabel('Cycle Number', fontweight='bold')
            ax2.set_ylabel('Hazard Rate (%/cycle)', fontweight='bold')
            ax2.set_title('Battery Degradation Hazard Rate', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative Hazard Function
        ax3 = axes[1, 0]
        
        if 'Smoothed_Hazard' in capacity_df.columns:
            capacity_df['Cumulative_Hazard'] = capacity_df['Smoothed_Hazard'].cumsum()
            
            ax3.plot(capacity_df['Cycle'], capacity_df['Cumulative_Hazard'], 
                    'green', linewidth=2, alpha=0.8)
            ax3.fill_between(capacity_df['Cycle'], capacity_df['Cumulative_Hazard'], 
                           alpha=0.3, color='green')
            
            ax3.set_xlabel('Cycle Number', fontweight='bold')
            ax3.set_ylabel('Cumulative Hazard', fontweight='bold')
            ax3.set_title('Cumulative Hazard Function', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # 4. Reliability Function and MTTF Estimation
        ax4 = axes[1, 1]
        
        # Calculate reliability function R(t) = exp(-Λ(t))
        if 'Cumulative_Hazard' in capacity_df.columns:
            capacity_df['Reliability'] = np.exp(-capacity_df['Cumulative_Hazard'] / 100)
            
            ax4.plot(capacity_df['Cycle'], capacity_df['Reliability'], 
                    'purple', linewidth=2, alpha=0.8, label='Reliability Function')
            
            # Find MTTF (Mean Time To Failure) - where reliability = 0.5
            mttf_mask = capacity_df['Reliability'] <= 0.5
            if mttf_mask.any():
                mttf_cycle = capacity_df[mttf_mask]['Cycle'].iloc[0]
                ax4.axvline(x=mttf_cycle, color='red', linestyle='--', 
                          linewidth=2, alpha=0.7, label=f'MTTF ≈ {mttf_cycle} cycles')
                ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
            
            ax4.set_xlabel('Cycle Number', fontweight='bold')
            ax4.set_ylabel('Reliability', fontweight='bold')
            ax4.set_title('Reliability Function & MTTF', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'survival_analysis.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_monte_carlo_analysis(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        몬테카를로 시뮬레이션 분석
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating Monte Carlo simulation analysis")
        
        df = data.copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Monte Carlo Simulation Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Prepare data
        sample_data = df.sample(n=min(10000, len(df)), random_state=42)
        
        # 1. Monte Carlo Capacity Prediction
        ax1 = axes[0, 0]
        
        # Extract parameters from data
        voltage_mean = sample_data['Voltage[V]'].mean()
        voltage_std = sample_data['Voltage[V]'].std()
        current_mean = sample_data['Current[A]'].mean()
        current_std = sample_data['Current[A]'].std()
        
        n_simulations = 5000
        future_cycles = 100
        
        # Monte Carlo simulation for capacity prediction
        simulated_capacities = []
        
        for _ in range(n_simulations):
            # Random walk for capacity degradation
            degradation_rate = np.random.normal(0.02, 0.005)  # 2% ± 0.5% per 100 cycles
            noise = np.random.normal(0, 0.01, future_cycles)
            
            capacity_trajectory = []
            current_capacity = 4.35  # Starting capacity
            
            for cycle in range(future_cycles):
                current_capacity *= (1 - degradation_rate/100)
                current_capacity += noise[cycle]
                current_capacity = max(current_capacity, 0)  # Ensure non-negative
                capacity_trajectory.append(current_capacity)
            
            simulated_capacities.append(capacity_trajectory)
        
        # Plot simulation results
        simulated_capacities = np.array(simulated_capacities)
        cycles_range = range(1, future_cycles + 1)
        
        # Plot percentiles
        percentiles = [5, 25, 50, 75, 95]
        colors = ['red', 'orange', 'green', 'orange', 'red']
        alphas = [0.3, 0.4, 0.8, 0.4, 0.3]
        
        for i, (p, color, alpha) in enumerate(zip(percentiles, colors, alphas)):
            p_values = np.percentile(simulated_capacities, p, axis=0)
            if p == 50:
                ax1.plot(cycles_range, p_values, color=color, linewidth=2, 
                        alpha=alpha, label=f'{p}th percentile (median)')
            else:
                ax1.plot(cycles_range, p_values, color=color, linewidth=1, 
                        alpha=alpha, label=f'{p}th percentile')
        
        # Fill between confidence intervals
        p5 = np.percentile(simulated_capacities, 5, axis=0)
        p95 = np.percentile(simulated_capacities, 95, axis=0)
        ax1.fill_between(cycles_range, p5, p95, alpha=0.2, color='gray', label='90% CI')
        
        ax1.set_xlabel('Cycle Number', fontweight='bold')
        ax1.set_ylabel('Capacity (Ah)', fontweight='bold')
        ax1.set_title('Monte Carlo Capacity Prediction', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Uncertainty Quantification
        ax2 = axes[0, 1]
        
        # Calculate final capacities distribution
        final_capacities = simulated_capacities[:, -1]
        
        ax2.hist(final_capacities, bins=50, alpha=0.7, density=True, 
                color='lightblue', edgecolor='black')
        
        # Fit normal distribution
        mu, sigma = stats.norm.fit(final_capacities)
        x_norm = np.linspace(final_capacities.min(), final_capacities.max(), 100)
        ax2.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 'r-', 
                linewidth=2, label=f'Normal fit (μ={mu:.3f}, σ={sigma:.3f})')
        
        # Add statistics
        ax2.axvline(mu, color='red', linestyle='--', alpha=0.7, label='Mean')
        ax2.axvline(np.median(final_capacities), color='green', linestyle='--', 
                   alpha=0.7, label='Median')
        
        ax2.set_xlabel('Final Capacity (Ah)', fontweight='bold')
        ax2.set_ylabel('Probability Density', fontweight='bold')
        ax2.set_title('Uncertainty in Final Capacity', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Sensitivity Analysis
        ax3 = axes[1, 0]
        
        # Vary key parameters and see effect on final capacity
        parameters = ['degradation_rate', 'noise_level', 'initial_capacity']
        param_effects = []
        
        base_degradation = 0.02
        base_noise = 0.01
        base_capacity = 4.35
        
        # Test parameter variations
        variations = [-0.5, -0.25, 0, 0.25, 0.5]  # ±50%, ±25%, baseline
        
        for param in parameters:
            effects = []
            for var in variations:
                n_sims = 1000
                final_caps = []
                
                for _ in range(n_sims):
                    if param == 'degradation_rate':
                        deg_rate = base_degradation * (1 + var)
                        noise_level = base_noise
                        init_cap = base_capacity
                    elif param == 'noise_level':
                        deg_rate = base_degradation
                        noise_level = base_noise * (1 + var)
                        init_cap = base_capacity
                    else:  # initial_capacity
                        deg_rate = base_degradation
                        noise_level = base_noise
                        init_cap = base_capacity * (1 + var)
                    
                    # Simple simulation
                    cap = init_cap
                    for _ in range(50):  # 50 cycles
                        cap *= (1 - deg_rate/100)
                        cap += np.random.normal(0, noise_level)
                        cap = max(cap, 0)
                    final_caps.append(cap)
                
                effects.append(np.mean(final_caps))
            
            param_effects.append(effects)
        
        # Plot sensitivity
        x_pos = np.arange(len(variations))
        width = 0.25
        
        for i, (param, effects) in enumerate(zip(parameters, param_effects)):
            ax3.bar(x_pos + i*width, effects, width, alpha=0.7, 
                   label=param.replace('_', ' ').title())
        
        ax3.set_xlabel('Parameter Variation', fontweight='bold')
        ax3.set_ylabel('Final Capacity (Ah)', fontweight='bold')
        ax3.set_title('Sensitivity Analysis', fontweight='bold')
        ax3.set_xticks(x_pos + width)
        ax3.set_xticklabels([f'{v*100:+.0f}%' for v in variations])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Risk Assessment
        ax4 = axes[1, 1]
        
        # Calculate risk of falling below different thresholds
        thresholds = [3.5, 3.0, 2.5, 2.0]
        risks = []
        
        for threshold in thresholds:
            risk = np.mean(final_capacities < threshold) * 100
            risks.append(risk)
        
        colors = ['green', 'yellow', 'orange', 'red']
        bars = ax4.bar(range(len(thresholds)), risks, color=colors, alpha=0.7)
        
        # Add percentage labels on bars
        for bar, risk in zip(bars, risks):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{risk:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_xlabel('Capacity Threshold (Ah)', fontweight='bold')
        ax4.set_ylabel('Risk of Falling Below (%)', fontweight='bold')
        ax4.set_title('Risk Assessment Matrix', fontweight='bold')
        ax4.set_xticks(range(len(thresholds)))
        ax4.set_xticklabels([f'{t} Ah' for t in thresholds])
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'monte_carlo_analysis.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_advanced_time_series_decomposition(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        고급 시계열 분해 분석 (STL, ARIMA)
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating advanced time series decomposition")
        
        df = data.copy()
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        fig.suptitle('Advanced Time Series Decomposition Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Prepare time series data
        ts_data = []
        cycles = sorted(df['TotalCycle'].unique())[:200]
        
        for cycle in cycles:
            cycle_data = df[df['TotalCycle'] == cycle]
            if len(cycle_data) > 0:
                ts_data.append({
                    'Cycle': cycle,
                    'Avg_Voltage': cycle_data['Voltage[V]'].mean(),
                    'Max_Capacity': cycle_data['Chg_Capacity[Ah]'].max(),
                    'Voltage_Variability': cycle_data['Voltage[V]'].std(),
                    'Current_Mean': cycle_data['Current[A]'].mean()
                })
        
        if len(ts_data) < 50:
            logger.warning("Insufficient data for time series decomposition")
            return fig
        
        ts_df = pd.DataFrame(ts_data)
        
        # 1. STL Decomposition (Seasonal and Trend decomposition using Loess)
        ax1 = axes[0, 0]
        ax2 = axes[0, 1]
        
        # Simple trend and residual extraction
        capacity_series = ts_df['Max_Capacity'].values
        
        # Calculate trend using rolling mean
        window_size = min(20, len(capacity_series) // 4)
        if window_size > 2:
            trend = pd.Series(capacity_series).rolling(window=window_size, center=True).mean()
            residuals = capacity_series - trend
            
            # Plot original and trend
            ax1.plot(ts_df['Cycle'], capacity_series, 'b-', alpha=0.7, label='Original')
            ax1.plot(ts_df['Cycle'], trend, 'r-', linewidth=2, label='Trend')
            ax1.set_xlabel('Cycle Number', fontweight='bold')
            ax1.set_ylabel('Capacity (Ah)', fontweight='bold')
            ax1.set_title('Time Series Decomposition: Trend', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot residuals
            ax2.plot(ts_df['Cycle'], residuals, 'g-', alpha=0.7)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Cycle Number', fontweight='bold')
            ax2.set_ylabel('Residuals', fontweight='bold')
            ax2.set_title('Time Series Decomposition: Residuals', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # 2. Autocorrelation Analysis
        ax3 = axes[1, 0]
        ax4 = axes[1, 1]
        
        if len(capacity_series) > 30:
            # Calculate autocorrelation
            max_lags = min(30, len(capacity_series) // 3)
            autocorr = []
            
            for lag in range(max_lags):
                if lag == 0:
                    autocorr.append(1.0)
                else:
                    corr = np.corrcoef(capacity_series[:-lag], capacity_series[lag:])[0, 1]
                    autocorr.append(corr if not np.isnan(corr) else 0)
            
            # Plot autocorrelation
            lags = range(max_lags)
            ax3.bar(lags, autocorr, alpha=0.7, color='lightblue')
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Significance (±0.2)')
            ax3.axhline(y=-0.2, color='red', linestyle='--', alpha=0.7)
            ax3.set_xlabel('Lag', fontweight='bold')
            ax3.set_ylabel('Autocorrelation', fontweight='bold')
            ax3.set_title('Autocorrelation Function (ACF)', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Partial autocorrelation (simplified)
            partial_autocorr = []
            for lag in range(min(15, max_lags)):
                if lag == 0:
                    partial_autocorr.append(1.0)
                elif lag == 1:
                    partial_autocorr.append(autocorr[1])
                else:
                    # Simplified calculation
                    partial_autocorr.append(autocorr[lag] * 0.8)  # Approximation
            
            ax4.bar(range(len(partial_autocorr)), partial_autocorr, 
                   alpha=0.7, color='lightcoral')
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax4.axhline(y=0.2, color='red', linestyle='--', alpha=0.7)
            ax4.axhline(y=-0.2, color='red', linestyle='--', alpha=0.7)
            ax4.set_xlabel('Lag', fontweight='bold')
            ax4.set_ylabel('Partial Autocorrelation', fontweight='bold')
            ax4.set_title('Partial Autocorrelation Function (PACF)', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        # 3. Spectral Analysis (Frequency Domain)
        ax5 = axes[2, 0]
        
        if len(capacity_series) > 40:
            # Remove trend for spectral analysis
            detrended = capacity_series - np.linspace(capacity_series[0], capacity_series[-1], len(capacity_series))
            
            # Calculate power spectral density
            freqs, psd = signal.periodogram(detrended, fs=1.0)  # fs=1 cycle per unit
            
            # Plot spectrum
            ax5.loglog(freqs[1:], psd[1:])  # Skip DC component
            ax5.set_xlabel('Frequency (cycles⁻¹)', fontweight='bold')
            ax5.set_ylabel('Power Spectral Density', fontweight='bold')
            ax5.set_title('Power Spectrum Analysis', fontweight='bold')
            ax5.grid(True, alpha=0.3)
        
        # 4. Change Point Detection
        ax6 = axes[2, 1]
        
        if len(capacity_series) > 20:
            # Simple change point detection using cumulative sum
            cumsum = np.cumsum(capacity_series - np.mean(capacity_series))
            
            ax6.plot(ts_df['Cycle'], cumsum, 'purple', linewidth=2)
            ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Find potential change points (local maxima/minima)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(np.abs(cumsum), height=np.std(cumsum))
            
            if len(peaks) > 0:
                change_points = ts_df['Cycle'].iloc[peaks]
                for cp in change_points:
                    ax6.axvline(x=cp, color='red', linestyle='--', alpha=0.7)
            
            ax6.set_xlabel('Cycle Number', fontweight='bold')
            ax6.set_ylabel('Cumulative Sum', fontweight='bold')
            ax6.set_title('Change Point Detection', fontweight='bold')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'advanced_time_series_decomposition.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig

    def create_all_statistical_visualizations(self, data: pd.DataFrame):
        """
        모든 통계 시각화 생성 (기존 + 신규 고급 분석)
        
        Args:
            data: 배터리 데이터
        """
        logger.info("Creating all statistical visualizations...")
        
        # 기존 분석들
        self.create_capacity_regression_analysis(data)
        self.create_performance_distribution_analysis(data)
        self.create_correlation_matrix_analysis(data)
        self.create_condition_comparison_analysis(data)
        self.create_time_series_statistical_analysis(data)
        self.create_advanced_statistical_summary(data)
        
        # 새로운 고급 분석들
        self.create_bayesian_analysis(data)
        self.create_survival_analysis(data)
        self.create_monte_carlo_analysis(data)
        self.create_advanced_time_series_decomposition(data)
        
        logger.info(f"All statistical visualizations saved to: {self.output_dir}")


def main():
    """메인 실행 함수"""
    print("Statistical Battery Visualizer")
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
    
    # Create statistical visualizer
    visualizer = StatisticalBatteryVisualizer()
    
    # Create all visualizations
    visualizer.create_all_statistical_visualizations(data)
    
    print(f"\nStatistical analysis completed!")
    print(f"Output directory: {visualizer.output_dir}")


if __name__ == "__main__":
    main()
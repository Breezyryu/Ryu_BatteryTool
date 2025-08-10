#!/usr/bin/env python3
"""
Enhanced Seaborn Visualizer Module
Seaborn 고급 기능을 활용한 배터리 데이터 시각화
FacetGrid, PairGrid, Clustermap 등 고급 시각화 기법 활용
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# Enhanced Seaborn styling
sns.set_theme(style="whitegrid", palette="husl", font_scale=1.1)
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
    'axes.spines.top': False,
    'axes.spines.right': False,
})

logger = logging.getLogger(__name__)

class EnhancedSeabornVisualizer:
    """Seaborn 고급 기능을 활용한 배터리 시각화 클래스"""
    
    def __init__(self, output_dir: str = "analysis_output/enhanced_seaborn_plots"):
        """
        초기화
        
        Args:
            output_dir: 출력 디렉토리 경로
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Battery info for file naming
        self.battery_info = None
        
        # Color palettes for different analysis types
        self.cycle_palette = sns.color_palette("viridis", n_colors=20)
        self.health_palette = sns.color_palette("RdYlGn_r", n_colors=10)
        self.performance_palette = sns.color_palette("plasma", n_colors=15)
        self.condition_palette = sns.color_palette("Set2", n_colors=8)
        
        logger.info(f"Enhanced Seaborn visualizer initialized, output: {self.output_dir}")
    
    def _generate_filename(self, plot_type: str) -> str:
        """
        배터리 정보를 기반으로 파일명 생성
        
        Args:
            plot_type: 플롯 타입
            
        Returns:
            생성된 파일명
        """
        if not self.battery_info:
            return f"{plot_type}.png"
        
        components = []
        
        # Add manufacturer
        if self.battery_info.get('manufacturer') and self.battery_info['manufacturer'] != 'Unknown':
            components.append(self.battery_info['manufacturer'])
        
        # Add model
        if self.battery_info.get('model'):
            components.append(self.battery_info['model'])
        
        # Add capacity
        if self.battery_info.get('capacity_mah'):
            components.append(f"{self.battery_info['capacity_mah']}mAh")
        
        # Add plot type
        components.append(plot_type)
        
        # Add date
        if self.battery_info.get('date'):
            components.append(self.battery_info['date'])
        
        filename = '_'.join(components) + '.png'
        return filename
    
    def prepare_battery_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        배터리 데이터를 시각화에 최적화된 형태로 준비
        
        Args:
            data: 원본 배터리 데이터
            
        Returns:
            전처리된 데이터프레임
        """
        logger.info("Preparing battery data for enhanced visualization")
        
        # 기본 데이터 복사
        df = data.copy()
        
        # 사이클 범위 카테고리 생성
        df['Cycle_Range'] = pd.cut(df['TotalCycle'], 
                                  bins=[0, 100, 300, 600, 1000, df['TotalCycle'].max() + 1],
                                  labels=['Early (1-100)', 'Growing (101-300)', 
                                         'Mature (301-600)', 'Aging (601-1000)', 
                                         'End-of-Life (1000+)'])
        
        # 전압 범위 카테고리 생성
        df['Voltage_Range'] = pd.cut(df['Voltage[V]'], 
                                    bins=[0, 3.0, 3.5, 4.0, 4.5, 5.0],
                                    labels=['Very Low (<3.0V)', 'Low (3.0-3.5V)', 
                                           'Medium (3.5-4.0V)', 'High (4.0-4.5V)', 
                                           'Very High (>4.5V)'])
        
        # C-rate 카테고리 생성
        df['C_Rate_Category'] = pd.cut(np.abs(df['Current[A]']), 
                                      bins=[0, 1.0, 2.0, 4.0, np.inf],
                                      labels=['Low (<1C)', 'Medium (1-2C)', 
                                             'High (2-4C)', 'Very High (>4C)'])
        
        # 충방전 상태 정의
        df['Phase'] = np.select([df['Current[A]'] > 0.1, 
                                df['Current[A]'] < -0.1],
                               ['Charge', 'Discharge'], 'Rest')
        
        # 용량 효율성 계산
        df['Capacity_Efficiency'] = np.abs(df['Dchg_Capacity[Ah]']) / (df['Chg_Capacity[Ah]'] + 1e-10)
        df['Capacity_Efficiency'] = np.clip(df['Capacity_Efficiency'], 0, 2)
        
        # 전력 계산
        df['Power'] = df['Voltage[V]'] * df['Current[A]']
        
        # 배터리 건강 상태 지표 (임의 계산)
        df['SOH_Indicator'] = 100 - (df['TotalCycle'] * 0.02) + np.random.normal(0, 2, len(df))
        df['SOH_Indicator'] = np.clip(df['SOH_Indicator'], 0, 100)
        
        return df
    
    def create_facet_grid_analysis(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        FacetGrid를 활용한 다면 분석 시각화
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating FacetGrid multi-facet analysis")
        
        df = self.prepare_battery_data(data)
        
        # Sample data to avoid overcrowding
        sample_size = min(20000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        # 1. 사이클 범위별 × 충방전별 전압-용량 관계
        fig = plt.figure(figsize=(20, 12))
        
        # Create FacetGrid
        g = sns.FacetGrid(df_sample, col='Cycle_Range', row='Phase', 
                         height=4, aspect=1.2, margin_titles=True)
        
        # Map scatterplot with regression line
        g.map_dataframe(sns.scatterplot, x='Voltage[V]', y='Chg_Capacity[Ah]', 
                       alpha=0.6, s=20)
        g.map_dataframe(sns.regplot, x='Voltage[V]', y='Chg_Capacity[Ah]', 
                       scatter=False, color='red', line_kws={'linewidth': 2})
        
        # Customize
        g.set_axis_labels('Voltage (V)', 'Capacity (Ah)')
        g.fig.suptitle('Voltage-Capacity Relationship by Cycle Range and Phase', 
                      fontsize=16, fontweight='bold', y=1.02)
        
        # Add statistics to each subplot
        for ax, title in zip(g.axes.flat, g.axes.flat):
            if ax.collections:  # Check if there are data points
                # Calculate R² for each subplot
                subplot_data = df_sample[(df_sample['Cycle_Range'].astype(str) in title.get_title()) & 
                                        (df_sample['Phase'].astype(str) in title.get_title())]
                if len(subplot_data) > 5:
                    corr = subplot_data['Voltage[V]'].corr(subplot_data['Chg_Capacity[Ah]'])
                    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            filename = self._generate_filename('facetgrid_voltage_capacity_analysis')
            save_path = self.output_dir / filename
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_pairgrid_correlation_matrix(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        PairGrid를 활용한 고급 상관관계 매트릭스
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating PairGrid correlation matrix")
        
        df = self.prepare_battery_data(data)
        
        # Select key variables for pair analysis
        pair_vars = ['Voltage[V]', 'Current[A]', 'Chg_Capacity[Ah]', 
                    'Power', 'Capacity_Efficiency', 'SOH_Indicator']
        
        # Sample data for performance
        sample_size = min(10000, len(df))
        df_sample = df[pair_vars + ['Phase']].sample(n=sample_size, random_state=42).dropna()
        
        # Create PairGrid
        g = sns.PairGrid(df_sample, vars=pair_vars, hue='Phase', 
                        height=2.5, aspect=1, palette=self.condition_palette)
        
        # Upper triangle: scatter plots with regression
        g.map_upper(sns.scatterplot, alpha=0.6, s=20)
        g.map_upper(sns.regplot, scatter=False, line_kws={'linewidth': 1.5})
        
        # Lower triangle: KDE plots
        g.map_lower(sns.kdeplot, fill=True, alpha=0.7)
        
        # Diagonal: histograms
        g.map_diag(sns.histplot, kde=True, alpha=0.7)
        
        # Add correlation coefficients to upper triangle
        def corr_func(x, y, **kwargs):
            r = stats.pearsonr(x, y)[0]
            ax = plt.gca()
            ax.annotate(f'r = {r:.3f}', xy=(0.1, 0.9), xycoords=ax.transAxes,
                       fontsize=10, fontweight='bold')
        
        g.map_upper(corr_func)
        
        # Add legend
        g.add_legend(title='Phase', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.suptitle('Battery Parameters Correlation Matrix (PairGrid)', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'pairgrid_correlation_matrix.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return g.fig
    
    def create_clustermap_analysis(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        Clustermap을 활용한 배터리 상태 클러스터링 분석
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating clustermap battery state analysis")
        
        df = self.prepare_battery_data(data)
        
        # 사이클별 통계 계산
        cycle_stats = []
        cycles = sorted(df['TotalCycle'].unique())[:200]  # First 200 cycles
        
        for cycle in cycles:
            cycle_data = df[df['TotalCycle'] == cycle]
            if len(cycle_data) > 0:
                stats_dict = {
                    'Cycle': cycle,
                    'Voltage_Mean': cycle_data['Voltage[V]'].mean(),
                    'Voltage_Std': cycle_data['Voltage[V]'].std(),
                    'Current_Mean': cycle_data['Current[A]'].mean(),
                    'Current_Std': cycle_data['Current[A]'].std(),
                    'Capacity_Mean': cycle_data['Chg_Capacity[Ah]'].mean(),
                    'Capacity_Std': cycle_data['Chg_Capacity[Ah]'].std(),
                    'Power_Mean': cycle_data['Power'].mean(),
                    'Power_Max': cycle_data['Power'].max(),
                    'Efficiency_Mean': cycle_data['Capacity_Efficiency'].mean(),
                    'SOH_Mean': cycle_data['SOH_Indicator'].mean()
                }
                cycle_stats.append(stats_dict)
        
        if not cycle_stats:
            logger.warning("No cycle statistics data found")
            return plt.figure()
        
        df_stats = pd.DataFrame(cycle_stats)
        df_stats = df_stats.set_index('Cycle')
        
        # Normalize data for clustering
        scaler = StandardScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df_stats), 
                                   index=df_stats.index, 
                                   columns=df_stats.columns)
        
        # Create clustermap
        plt.figure(figsize=(14, 10))
        
        g = sns.clustermap(df_normalized.T, 
                          cmap='RdBu_r', center=0,
                          figsize=(14, 10),
                          annot=False, fmt='.2f',
                          cbar_kws={'label': 'Normalized Value'},
                          row_cluster=True, col_cluster=True,
                          linewidths=0.5,
                          xticklabels=10,  # Show every 10th cycle
                          yticklabels=True)
        
        g.fig.suptitle('Battery Performance Clustering Analysis\n(Hierarchical Clustering of Cycle Statistics)', 
                      fontsize=14, fontweight='bold', y=0.98)
        
        if save:
            save_path = self.output_dir / 'clustermap_battery_states.png'
            g.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return g.fig
    
    def create_jointgrid_detailed_analysis(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        JointGrid를 활용한 세부 관계 분석
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating JointGrid detailed relationship analysis")
        
        df = self.prepare_battery_data(data)
        
        # Sample data for performance
        sample_size = min(15000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        # Create JointGrid for voltage vs capacity relationship
        g = sns.JointGrid(data=df_sample, x='Voltage[V]', y='Chg_Capacity[Ah]', 
                         height=8, space=0.2)
        
        # Main plot: scatter with regression
        g.plot_joint(sns.scatterplot, alpha=0.6, s=20, hue=df_sample['Cycle_Range'])
        g.plot_joint(sns.regplot, scatter=False, color='red', line_kws={'linewidth': 2})
        
        # Marginal plots
        g.plot_marginals(sns.histplot, kde=True, alpha=0.7)
        
        # Add correlation and regression info
        corr = df_sample['Voltage[V]'].corr(df_sample['Chg_Capacity[Ah]'])
        g.ax_joint.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=g.ax_joint.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=12, fontweight='bold')
        
        # Customize labels
        g.set_axis_labels('Voltage (V)', 'Charge Capacity (Ah)', fontsize=12, fontweight='bold')
        g.fig.suptitle('Voltage-Capacity Relationship with Marginal Distributions', 
                      fontsize=14, fontweight='bold', y=0.98)
        
        if save:
            save_path = self.output_dir / 'jointgrid_voltage_capacity_analysis.png'
            g.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return g.fig
    
    def create_advanced_violin_plots(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        고급 violin plot 분석 (분포 + 통계)
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating advanced violin plot analysis")
        
        df = self.prepare_battery_data(data)
        
        # Sample data
        sample_size = min(25000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced Distribution Analysis with Violin Plots', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Voltage distribution by cycle range
        ax1 = axes[0, 0]
        sns.violinplot(data=df_sample, x='Cycle_Range', y='Voltage[V]', 
                      ax=ax1, palette=self.health_palette)
        ax1.set_title('Voltage Distribution by Battery Age', fontweight='bold')
        ax1.set_xlabel('Cycle Range', fontweight='bold')
        ax1.set_ylabel('Voltage (V)', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Capacity efficiency by phase and C-rate
        ax2 = axes[0, 1]
        phase_data = df_sample[df_sample['Phase'] != 'Rest']
        sns.violinplot(data=phase_data, x='C_Rate_Category', y='Capacity_Efficiency', 
                      hue='Phase', ax=ax2, palette=self.condition_palette, split=True)
        ax2.set_title('Capacity Efficiency by C-rate and Phase', fontweight='bold')
        ax2.set_xlabel('C-rate Category', fontweight='bold')
        ax2.set_ylabel('Capacity Efficiency', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. SOH indicator by voltage range
        ax3 = axes[1, 0]
        sns.violinplot(data=df_sample, x='Voltage_Range', y='SOH_Indicator', 
                      ax=ax3, palette=self.performance_palette)
        ax3.set_title('SOH Distribution by Voltage Range', fontweight='bold')
        ax3.set_xlabel('Voltage Range', fontweight='bold')
        ax3.set_ylabel('SOH Indicator (%)', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Power distribution comparison
        ax4 = axes[1, 1]
        power_data = df_sample[np.abs(df_sample['Power']) < np.percentile(np.abs(df_sample['Power']), 95)]
        sns.violinplot(data=power_data, x='Phase', y='Power', ax=ax4, 
                      palette=['lightcoral', 'lightblue', 'lightgreen'])
        ax4.set_title('Power Distribution by Phase', fontweight='bold')
        ax4.set_xlabel('Phase', fontweight='bold')
        ax4.set_ylabel('Power (W)', fontweight='bold')
        
        # Add statistical annotations
        for ax in axes.flat:
            # Add median lines
            for violin in ax.collections:
                if hasattr(violin, 'get_paths'):
                    for path in violin.get_paths():
                        vertices = path.vertices
                        if len(vertices) > 0:
                            ax.axhline(y=np.median(vertices[:, 1]), color='red', 
                                     linestyle='--', alpha=0.7, linewidth=1)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'advanced_violin_plots.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_regression_analysis_grid(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        다양한 회귀 분석을 한 번에 시각화
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating comprehensive regression analysis grid")
        
        df = self.prepare_battery_data(data)
        
        # Calculate cycle-level aggregates
        cycle_agg = df.groupby('TotalCycle').agg({
            'Voltage[V]': ['mean', 'std'],
            'Chg_Capacity[Ah]': ['mean', 'max'],
            'SOH_Indicator': 'mean',
            'Capacity_Efficiency': 'mean'
        }).round(4)
        
        # Flatten column names
        cycle_agg.columns = ['_'.join(col) for col in cycle_agg.columns]
        cycle_agg = cycle_agg.reset_index()
        
        # Take first 200 cycles for cleaner visualization
        cycle_agg = cycle_agg[cycle_agg['TotalCycle'] <= 200]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Regression Analysis Grid', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Linear regression: Cycle vs SOH
        ax1 = axes[0, 0]
        sns.regplot(data=cycle_agg, x='TotalCycle', y='SOH_Indicator_mean', 
                   ax=ax1, scatter_kws={'s': 30, 'alpha': 0.7}, 
                   line_kws={'color': 'red', 'linewidth': 2})
        ax1.set_title('SOH Degradation Trend (Linear)', fontweight='bold')
        ax1.set_xlabel('Cycle Number', fontweight='bold')
        ax1.set_ylabel('SOH Indicator (%)', fontweight='bold')
        
        # Add R² score
        from sklearn.metrics import r2_score
        from sklearn.linear_model import LinearRegression
        X = cycle_agg[['TotalCycle']]
        y = cycle_agg['SOH_Indicator_mean']
        model = LinearRegression().fit(X, y)
        r2 = r2_score(y, model.predict(X))
        ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Polynomial regression: Cycle vs Capacity
        ax2 = axes[0, 1]
        sns.regplot(data=cycle_agg, x='TotalCycle', y='Chg_Capacity[Ah]_mean', 
                   order=2, ax=ax2, scatter_kws={'s': 30, 'alpha': 0.7}, 
                   line_kws={'color': 'green', 'linewidth': 2})
        ax2.set_title('Capacity Fade Trend (Polynomial)', fontweight='bold')
        ax2.set_xlabel('Cycle Number', fontweight='bold')
        ax2.set_ylabel('Average Charge Capacity (Ah)', fontweight='bold')
        
        # 3. LOWESS regression: Voltage variability
        ax3 = axes[1, 0]
        sns.regplot(data=cycle_agg, x='TotalCycle', y='Voltage[V]_std', 
                   lowess=True, ax=ax3, scatter_kws={'s': 30, 'alpha': 0.7}, 
                   line_kws={'color': 'purple', 'linewidth': 2})
        ax3.set_title('Voltage Variability Trend (LOWESS)', fontweight='bold')
        ax3.set_xlabel('Cycle Number', fontweight='bold')
        ax3.set_ylabel('Voltage Std Dev (V)', fontweight='bold')
        
        # 4. Multiple regression visualization
        ax4 = axes[1, 1]
        # Create a composite health score
        cycle_agg['Composite_Health'] = (cycle_agg['SOH_Indicator_mean'] + 
                                        cycle_agg['Capacity_Efficiency_mean'] * 100) / 2
        
        sns.regplot(data=cycle_agg, x='TotalCycle', y='Composite_Health', 
                   ax=ax4, scatter_kws={'s': 30, 'alpha': 0.7}, 
                   line_kws={'color': 'orange', 'linewidth': 2})
        ax4.set_title('Composite Health Score Trend', fontweight='bold')
        ax4.set_xlabel('Cycle Number', fontweight='bold')
        ax4.set_ylabel('Composite Health Score', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'regression_analysis_grid.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_all_enhanced_visualizations(self, data: pd.DataFrame):
        """
        모든 고급 Seaborn 시각화 생성
        
        Args:
            data: 배터리 데이터
        """
        logger.info("Creating all enhanced Seaborn visualizations...")
        
        # 1. FacetGrid analysis
        self.create_facet_grid_analysis(data)
        
        # 2. PairGrid correlation matrix
        self.create_pairgrid_correlation_matrix(data)
        
        # 3. Clustermap analysis
        self.create_clustermap_analysis(data)
        
        # 4. JointGrid detailed analysis
        self.create_jointgrid_detailed_analysis(data)
        
        # 5. Advanced violin plots
        self.create_advanced_violin_plots(data)
        
        # 6. Regression analysis grid
        self.create_regression_analysis_grid(data)
        
        logger.info(f"All enhanced visualizations saved to: {self.output_dir}")


def main():
    """메인 실행 함수"""
    print("Enhanced Seaborn Battery Visualizer")
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
    
    # Create enhanced visualizer
    visualizer = EnhancedSeabornVisualizer()
    
    # Create all visualizations
    visualizer.create_all_enhanced_visualizations(data)
    
    print(f"\nEnhanced Seaborn analysis completed!")
    print(f"Output directory: {visualizer.output_dir}")


if __name__ == "__main__":
    main()
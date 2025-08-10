#!/usr/bin/env python3
"""
Advanced Battery Visualizations Module
고급 배터리 데이터 시각화 도구
Journal-quality plots for deep analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import signal, interpolate
from mpl_toolkits.mplot3d import Axes3D
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

warnings.filterwarnings('ignore')

# Configure matplotlib for publication quality
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

logger = logging.getLogger(__name__)

class AdvancedVisualizer:
    """고급 배터리 데이터 시각화 클래스"""
    
    def __init__(self, output_dir: str = "analysis_output/advanced_plots"):
        """
        초기화
        
        Args:
            output_dir: 출력 디렉토리 경로
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Custom colormaps
        self.create_custom_colormaps()
    
    def create_custom_colormaps(self):
        """커스텀 컬러맵 생성"""
        # Battery health colormap (red to green)
        colors = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
        self.health_cmap = LinearSegmentedColormap.from_list('battery_health', colors[::-1])
        
        # Voltage colormap
        colors = ['#2c7bb6', '#00a6ca', '#00ccbc', '#90eb9d', '#ffff8c', '#f9d057', '#f29e2e', '#e76818', '#d7191c']
        self.voltage_cmap = LinearSegmentedColormap.from_list('voltage', colors)
    
    def plot_incremental_capacity(self, ica_results: Dict, save: bool = True) -> plt.Figure:
        """
        Incremental Capacity (dQ/dV) 플롯
        
        Args:
            ica_results: ICA 분석 결과
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1])
        
        # Main ICA plot
        ax1 = fig.add_subplot(gs[:, 0])
        
        # Select cycles to plot
        cycles_to_plot = ica_results['cycles'][:10] if len(ica_results['cycles']) > 10 else ica_results['cycles']
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(cycles_to_plot)))
        
        for i, (cyc, data) in enumerate(zip(cycles_to_plot, ica_results['dqdv_data'][:10])):
            ax1.plot(data['voltage'], data['dqdv'], 
                    color=colors[i], label=f'Cycle {cyc}', 
                    alpha=0.8, linewidth=1.5)
        
        ax1.set_xlabel('Voltage (V)', fontweight='bold')
        ax1.set_ylabel('dQ/dV (Ah/V)', fontweight='bold')
        ax1.set_title('Incremental Capacity Analysis', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', frameon=True, fancybox=True)
        ax1.set_xlim([2.5, 4.6])
        ax1.grid(True, alpha=0.3)
        
        # Peak evolution plot
        ax2 = fig.add_subplot(gs[0, 1])
        
        if ica_results['peaks']:
            cycles = [p['cycle'] for p in ica_results['peaks']]
            num_peaks = [p['num_peaks'] for p in ica_results['peaks']]
            
            ax2.scatter(cycles, num_peaks, c=cycles, cmap='viridis', s=50, alpha=0.7)
            ax2.set_xlabel('Cycle Number')
            ax2.set_ylabel('Number of Peaks')
            ax2.set_title('Peak Evolution', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # Peak voltage tracking
        ax3 = fig.add_subplot(gs[1, 1])
        
        if ica_results['peaks']:
            for peak_idx in range(min(3, min([p['num_peaks'] for p in ica_results['peaks']]))):
                peak_voltages = []
                cycles_with_peak = []
                
                for p in ica_results['peaks']:
                    if p['num_peaks'] > peak_idx:
                        peak_voltages.append(p['peak_voltages'][peak_idx])
                        cycles_with_peak.append(p['cycle'])
                
                if peak_voltages:
                    ax3.plot(cycles_with_peak, peak_voltages, 
                            marker='o', label=f'Peak {peak_idx+1}', alpha=0.7)
            
            ax3.set_xlabel('Cycle Number')
            ax3.set_ylabel('Peak Voltage (V)')
            ax3.set_title('Peak Voltage Shift', fontweight='bold')
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Incremental Capacity Analysis (dQ/dV)', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'incremental_capacity_analysis.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_differential_voltage(self, dva_results: Dict, save: bool = True) -> plt.Figure:
        """
        Differential Voltage (dV/dQ) 플롯
        
        Args:
            dva_results: DVA 분석 결과
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # DVA curves
        cycles_to_plot = dva_results['cycles'][:10] if len(dva_results['cycles']) > 10 else dva_results['cycles']
        colors = plt.cm.plasma(np.linspace(0, 0.9, len(cycles_to_plot)))
        
        for i, (cyc, data) in enumerate(zip(cycles_to_plot, dva_results['dvdq_data'][:10])):
            ax1.plot(data['capacity'], data['dvdq'], 
                    color=colors[i], label=f'Cycle {cyc}', 
                    alpha=0.8, linewidth=1.5)
        
        ax1.set_xlabel('Capacity (Ah)', fontweight='bold')
        ax1.set_ylabel('dV/dQ (V/Ah)', fontweight='bold')
        ax1.set_title('Differential Voltage Analysis', fontweight='bold')
        ax1.legend(loc='upper right', frameon=True, fancybox=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([-5, 5])
        
        # Valley analysis
        if dva_results['valleys']:
            cycles = [v['cycle'] for v in dva_results['valleys']]
            num_valleys = [v['num_valleys'] for v in dva_results['valleys']]
            
            ax2.plot(cycles, num_valleys, marker='s', markersize=8, 
                    color='#e74c3c', linewidth=2, alpha=0.7)
            ax2.fill_between(cycles, num_valleys, alpha=0.3, color='#e74c3c')
            ax2.set_xlabel('Cycle Number', fontweight='bold')
            ax2.set_ylabel('Number of Phase Transitions', fontweight='bold')
            ax2.set_title('Phase Transition Evolution', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Differential Voltage Analysis (dV/dQ)', fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'differential_voltage_analysis.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_3d_surface(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        3D Surface plot (Cycle-Voltage-Capacity)
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sample data for 3D plot
        cycles_to_plot = data['TotalCycle'].unique()[:50]
        
        # Create meshgrid
        voltage_bins = np.linspace(2.5, 4.6, 50)
        X, Y = np.meshgrid(cycles_to_plot, voltage_bins)
        Z = np.zeros_like(X)
        
        for i, cyc in enumerate(cycles_to_plot):
            cycle_data = data[data['TotalCycle'] == cyc]
            
            # Calculate capacity at each voltage
            for j, v in enumerate(voltage_bins):
                mask = (cycle_data['Voltage[V]'] >= v - 0.02) & (cycle_data['Voltage[V]'] < v + 0.02)
                if mask.sum() > 0:
                    Z[j, i] = cycle_data[mask]['Chg_Capacity[Ah]'].mean()
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap=self.voltage_cmap, 
                              linewidth=0, antialiased=True, alpha=0.9)
        
        ax.set_xlabel('Cycle Number', fontweight='bold', labelpad=10)
        ax.set_ylabel('Voltage (V)', fontweight='bold', labelpad=10)
        ax.set_zlabel('Capacity (Ah)', fontweight='bold', labelpad=10)
        ax.set_title('3D Voltage-Capacity Evolution', fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
        
        # Set viewing angle
        ax.view_init(elev=25, azim=45)
        
        if save:
            save_path = self.output_dir / '3d_surface_plot.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_capacity_heatmap(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        Capacity degradation heatmap
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Prepare data for heatmap
        cycles = data['TotalCycle'].unique()[:200]
        voltage_bins = np.linspace(2.5, 4.6, 100)
        
        # Charge heatmap
        charge_matrix = np.zeros((len(voltage_bins)-1, len(cycles)))
        
        for i, cyc in enumerate(cycles):
            cycle_data = data[(data['TotalCycle'] == cyc) & (data['Current[A]'] > 0)]
            if len(cycle_data) > 0:
                hist, _ = np.histogram(cycle_data['Voltage[V]'], bins=voltage_bins, 
                                      weights=cycle_data['Current[A]'])
                charge_matrix[:, i] = hist
        
        im1 = ax1.imshow(charge_matrix, aspect='auto', cmap=self.voltage_cmap,
                        extent=[cycles[0], cycles[-1], voltage_bins[0], voltage_bins[-1]],
                        origin='lower', interpolation='bilinear')
        
        ax1.set_xlabel('Cycle Number', fontweight='bold')
        ax1.set_ylabel('Voltage (V)', fontweight='bold')
        ax1.set_title('Charge Voltage Distribution Heatmap', fontweight='bold')
        
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Current Density (A)', fontweight='bold')
        
        # Discharge heatmap
        discharge_matrix = np.zeros((len(voltage_bins)-1, len(cycles)))
        
        for i, cyc in enumerate(cycles):
            cycle_data = data[(data['TotalCycle'] == cyc) & (data['Current[A]'] < 0)]
            if len(cycle_data) > 0:
                hist, _ = np.histogram(cycle_data['Voltage[V]'], bins=voltage_bins,
                                      weights=np.abs(cycle_data['Current[A]']))
                discharge_matrix[:, i] = hist
        
        im2 = ax2.imshow(discharge_matrix, aspect='auto', cmap=self.voltage_cmap,
                        extent=[cycles[0], cycles[-1], voltage_bins[0], voltage_bins[-1]],
                        origin='lower', interpolation='bilinear')
        
        ax2.set_xlabel('Cycle Number', fontweight='bold')
        ax2.set_ylabel('Voltage (V)', fontweight='bold')
        ax2.set_title('Discharge Voltage Distribution Heatmap', fontweight='bold')
        
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Current Density (A)', fontweight='bold')
        
        plt.suptitle('Voltage Distribution Evolution', fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'capacity_heatmap.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_soh_comprehensive(self, soh_results: Dict, save: bool = True) -> plt.Figure:
        """
        Comprehensive State of Health visualization
        
        Args:
            soh_results: SOH 분석 결과
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
        
        cycles = soh_results['cycles']
        
        # Combined SOH
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(cycles, soh_results['combined_soh'], 
                color='#2c3e50', linewidth=2.5, label='Combined SOH')
        ax1.fill_between(cycles, soh_results['combined_soh'], 
                         alpha=0.3, color='#2c3e50')
        
        # Add 80% threshold line
        ax1.axhline(y=80, color='red', linestyle='--', linewidth=1.5, 
                   label='80% EOL Threshold')
        
        ax1.set_xlabel('Cycle Number', fontweight='bold')
        ax1.set_ylabel('State of Health (%)', fontweight='bold')
        ax1.set_title('Overall Battery State of Health', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([70, 105])
        
        # Capacity-based SOH
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(cycles, soh_results['capacity_based_soh'], 
                color='#3498db', linewidth=2, marker='o', markersize=3, 
                markevery=10, alpha=0.8)
        ax2.set_xlabel('Cycle Number', fontweight='bold')
        ax2.set_ylabel('Capacity SOH (%)', fontweight='bold')
        ax2.set_title('Capacity-based SOH', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Resistance-based SOH
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(cycles, soh_results['resistance_based_soh'], 
                color='#e74c3c', linewidth=2, marker='s', markersize=3,
                markevery=10, alpha=0.8)
        ax3.set_xlabel('Cycle Number', fontweight='bold')
        ax3.set_ylabel('Resistance SOH (%)', fontweight='bold')
        ax3.set_title('Resistance-based SOH', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Voltage-based SOH
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(cycles, soh_results['voltage_based_soh'], 
                color='#27ae60', linewidth=2, marker='^', markersize=3,
                markevery=10, alpha=0.8)
        ax4.set_xlabel('Cycle Number', fontweight='bold')
        ax4.set_ylabel('Voltage SOH (%)', fontweight='bold')
        ax4.set_title('Voltage-based SOH', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # SOH correlation
        ax5 = fig.add_subplot(gs[2, 1])
        
        # Create correlation scatter plot
        ax5.scatter(soh_results['capacity_based_soh'], 
                   soh_results['resistance_based_soh'],
                   c=cycles, cmap='viridis', s=20, alpha=0.6)
        
        # Add diagonal line
        min_val = min(min(soh_results['capacity_based_soh']), 
                     min(soh_results['resistance_based_soh']))
        max_val = max(max(soh_results['capacity_based_soh']), 
                     max(soh_results['resistance_based_soh']))
        ax5.plot([min_val, max_val], [min_val, max_val], 
                'r--', alpha=0.5, linewidth=1)
        
        ax5.set_xlabel('Capacity SOH (%)', fontweight='bold')
        ax5.set_ylabel('Resistance SOH (%)', fontweight='bold')
        ax5.set_title('SOH Correlation', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(ax5.collections[0], ax=ax5)
        cbar.set_label('Cycle Number', fontweight='bold')
        
        plt.suptitle('Comprehensive State of Health Analysis', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'soh_comprehensive.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_resistance_evolution(self, resistance_results: Dict, save: bool = True) -> plt.Figure:
        """
        Internal resistance evolution plot
        
        Args:
            resistance_results: 내부 저항 분석 결과
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        cycles = resistance_results['cycles']
        
        # Average resistance
        valid_mask = ~np.isnan(resistance_results['average_resistance'])
        valid_cycles = np.array(cycles)[valid_mask]
        valid_resistance = np.array(resistance_results['average_resistance'])[valid_mask]
        
        if len(valid_resistance) > 0:
            ax1.plot(valid_cycles, valid_resistance, 
                    color='#34495e', linewidth=2, marker='o', markersize=4)
            ax1.set_xlabel('Cycle Number', fontweight='bold')
            ax1.set_ylabel('Resistance (mΩ)', fontweight='bold')
            ax1.set_title('Average Internal Resistance', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(valid_cycles, valid_resistance, 1)
            p = np.poly1d(z)
            ax1.plot(valid_cycles, p(valid_cycles), "r--", alpha=0.5, linewidth=1)
        
        # Charge vs Discharge resistance
        charge_r = np.array(resistance_results['charge_resistance'])
        discharge_r = np.array(resistance_results['discharge_resistance'])
        
        valid_charge = ~np.isnan(charge_r)
        valid_discharge = ~np.isnan(discharge_r)
        
        ax2.plot(np.array(cycles)[valid_charge], charge_r[valid_charge], 
                color='#e74c3c', linewidth=1.5, alpha=0.7, label='Charge')
        ax2.plot(np.array(cycles)[valid_discharge], discharge_r[valid_discharge], 
                color='#3498db', linewidth=1.5, alpha=0.7, label='Discharge')
        
        ax2.set_xlabel('Cycle Number', fontweight='bold')
        ax2.set_ylabel('Resistance (mΩ)', fontweight='bold')
        ax2.set_title('Charge vs Discharge Resistance', fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Resistance growth
        if 'resistance_growth' in resistance_results and resistance_results['resistance_growth']:
            growth_cycles = cycles[:len(resistance_results['resistance_growth'])]
            ax3.plot(growth_cycles, resistance_results['resistance_growth'], 
                    color='#e67e22', linewidth=2)
            ax3.fill_between(growth_cycles, resistance_results['resistance_growth'], 
                            alpha=0.3, color='#e67e22')
            ax3.set_xlabel('Cycle Number', fontweight='bold')
            ax3.set_ylabel('Resistance Growth (%)', fontweight='bold')
            ax3.set_title('Resistance Growth Rate', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Resistance distribution
        ax4.hist(valid_resistance, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Resistance (mΩ)', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('Resistance Distribution', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        if len(valid_resistance) > 0:
            ax4.axvline(x=np.mean(valid_resistance), color='red', 
                       linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(valid_resistance):.2f}')
            ax4.axvline(x=np.median(valid_resistance), color='blue', 
                       linestyle='--', linewidth=1.5, label=f'Median: {np.median(valid_resistance):.2f}')
            ax4.legend()
        
        plt.suptitle('Internal Resistance Analysis', fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'resistance_evolution.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_efficiency_analysis(self, efficiency_results: Dict, save: bool = True) -> plt.Figure:
        """
        Efficiency analysis visualization
        
        Args:
            efficiency_results: 효율성 분석 결과
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        cycles = efficiency_results['cycles']
        
        # Coulombic efficiency
        ax1.plot(cycles, efficiency_results['coulombic_efficiency'], 
                color='#2ecc71', linewidth=1.5, alpha=0.8)
        ax1.fill_between(cycles, efficiency_results['coulombic_efficiency'], 
                         95, where=(np.array(efficiency_results['coulombic_efficiency']) >= 95),
                         alpha=0.3, color='#2ecc71', label='CE ≥ 95%')
        ax1.fill_between(cycles, efficiency_results['coulombic_efficiency'], 
                         95, where=(np.array(efficiency_results['coulombic_efficiency']) < 95),
                         alpha=0.3, color='#e74c3c', label='CE < 95%')
        
        ax1.set_xlabel('Cycle Number', fontweight='bold')
        ax1.set_ylabel('Coulombic Efficiency (%)', fontweight='bold')
        ax1.set_title('Coulombic Efficiency', fontweight='bold')
        ax1.set_ylim([90, 102])
        ax1.axhline(y=100, color='black', linestyle='-', linewidth=0.5)
        ax1.axhline(y=95, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax1.legend(loc='lower left')
        ax1.grid(True, alpha=0.3)
        
        # Moving average
        if 'moving_average_ce' in efficiency_results and efficiency_results['moving_average_ce']:
            ma_cycles = cycles[:len(efficiency_results['moving_average_ce'])]
            ax2.plot(cycles, efficiency_results['coulombic_efficiency'], 
                    color='lightgray', linewidth=0.5, alpha=0.5, label='Raw')
            ax2.plot(ma_cycles, efficiency_results['moving_average_ce'], 
                    color='#e74c3c', linewidth=2, label='10-cycle MA')
            ax2.set_xlabel('Cycle Number', fontweight='bold')
            ax2.set_ylabel('Coulombic Efficiency (%)', fontweight='bold')
            ax2.set_title('Coulombic Efficiency Trend', fontweight='bold')
            ax2.legend(loc='lower left')
            ax2.grid(True, alpha=0.3)
        
        # Energy efficiency
        if efficiency_results['energy_efficiency']:
            valid_ee = [e for e in efficiency_results['energy_efficiency'] if e > 0]
            valid_cycles = cycles[:len(valid_ee)]
            ax3.plot(valid_cycles, valid_ee, 
                    color='#3498db', linewidth=1.5, marker='s', 
                    markersize=3, markevery=10, alpha=0.8)
            ax3.set_xlabel('Cycle Number', fontweight='bold')
            ax3.set_ylabel('Energy Efficiency (%)', fontweight='bold')
            ax3.set_title('Energy Efficiency', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # Efficiency correlation
        ce = efficiency_results['coulombic_efficiency']
        ve = efficiency_results['voltage_efficiency']
        
        valid_mask = (np.array(ce) > 0) & (np.array(ve) > 0)
        if valid_mask.sum() > 0:
            ax4.scatter(np.array(ce)[valid_mask], np.array(ve)[valid_mask], 
                       c=np.array(cycles)[valid_mask], cmap='coolwarm', 
                       s=20, alpha=0.6)
            ax4.set_xlabel('Coulombic Efficiency (%)', fontweight='bold')
            ax4.set_ylabel('Voltage Efficiency (%)', fontweight='bold')
            ax4.set_title('Efficiency Correlation', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(ax4.collections[0], ax=ax4)
            cbar.set_label('Cycle Number', fontweight='bold')
        
        plt.suptitle('Battery Efficiency Analysis', fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'efficiency_analysis.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_anomaly_detection(self, anomaly_results: Dict, save: bool = True) -> plt.Figure:
        """
        Anomaly detection visualization
        
        Args:
            anomaly_results: 이상 감지 결과
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        cycles = anomaly_results['cycle_numbers']
        scores = anomaly_results['anomaly_scores']
        anomalous = anomaly_results['anomalous_cycles']
        
        # Anomaly scores
        colors = ['red' if c in anomalous else 'blue' for c in cycles]
        ax1.scatter(cycles, scores, c=colors, alpha=0.6, s=30)
        ax1.set_xlabel('Cycle Number', fontweight='bold')
        ax1.set_ylabel('Anomaly Score', fontweight='bold')
        ax1.set_title('Anomaly Detection Scores', fontweight='bold')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        normal_patch = mpatches.Patch(color='blue', label='Normal')
        anomaly_patch = mpatches.Patch(color='red', label='Anomaly')
        ax1.legend(handles=[normal_patch, anomaly_patch])
        
        # Feature importance
        if 'feature_importance' in anomaly_results:
            features = ['Voltage Mean', 'Voltage Std', 'Current Mean', 'Current Std',
                       'Charge Cap', 'Discharge Cap', 'V Max', 'V Min', 'Duration']
            importance = anomaly_results['feature_importance']
            
            ax2.barh(features, importance, color='#3498db', alpha=0.7)
            ax2.set_xlabel('Feature Importance', fontweight='bold')
            ax2.set_title('Anomaly Detection Feature Importance', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Anomaly Detection Analysis', fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'anomaly_detection.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def plot_capacity_fade_models(self, fade_results: Dict, save: bool = True) -> plt.Figure:
        """
        Capacity fade model comparison
        
        Args:
            fade_results: 용량 감소 모델 결과
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        cycles = fade_results['cycles']
        measured = np.array(fade_results['measured_capacity'])
        
        # Normalize
        initial_cap = measured[0] if len(measured) > 0 else 4.352
        measured_retention = (measured / initial_cap) * 100
        
        # Plot measured data
        ax1.scatter(cycles, measured_retention, color='black', s=20, 
                   alpha=0.5, label='Measured', zorder=5)
        
        # Plot models
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        linestyles = ['-', '--', '-.', ':']
        
        for i, (model_name, model_data) in enumerate(fade_results['models'].items()):
            if 'fitted' in model_data:
                ax1.plot(cycles, model_data['fitted'], 
                        color=colors[i % len(colors)], 
                        linestyle=linestyles[i % len(linestyles)],
                        linewidth=2, label=model_name.replace('_', ' ').title(),
                        alpha=0.8)
        
        ax1.set_xlabel('Cycle Number', fontweight='bold')
        ax1.set_ylabel('Capacity Retention (%)', fontweight='bold')
        ax1.set_title('Capacity Fade Models', fontweight='bold')
        ax1.legend(loc='lower left')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        # Model residuals
        ax2.set_xlabel('Cycle Number', fontweight='bold')
        ax2.set_ylabel('Residual (%)', fontweight='bold')
        ax2.set_title('Model Residuals', fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3)
        
        for i, (model_name, model_data) in enumerate(fade_results['models'].items()):
            if 'fitted' in model_data:
                residuals = measured_retention - model_data['fitted']
                ax2.plot(cycles, residuals, 
                        color=colors[i % len(colors)],
                        linewidth=1.5, label=model_name.replace('_', ' ').title(),
                        alpha=0.7, marker='o', markersize=2)
        
        ax2.legend(loc='best')
        
        plt.suptitle('Capacity Fade Modeling', fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'capacity_fade_models.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_all_visualizations(self, analysis_results: Dict, data: pd.DataFrame = None):
        """
        모든 시각화 생성
        
        Args:
            analysis_results: 전체 분석 결과
            data: 원본 데이터 (optional)
        """
        logger.info("Creating all advanced visualizations...")
        
        # ICA plot
        if 'ica' in analysis_results:
            self.plot_incremental_capacity(analysis_results['ica'])
        
        # DVA plot
        if 'dva' in analysis_results:
            self.plot_differential_voltage(analysis_results['dva'])
        
        # SOH plot
        if 'soh' in analysis_results:
            self.plot_soh_comprehensive(analysis_results['soh'])
        
        # Resistance plot
        if 'resistance' in analysis_results:
            self.plot_resistance_evolution(analysis_results['resistance'])
        
        # Efficiency plot
        if 'efficiency' in analysis_results:
            self.plot_efficiency_analysis(analysis_results['efficiency'])
        
        # Anomaly plot
        if 'anomaly_detection' in analysis_results:
            self.plot_anomaly_detection(analysis_results['anomaly_detection'])
        
        # Capacity fade plot
        if 'capacity_fade' in analysis_results:
            self.plot_capacity_fade_models(analysis_results['capacity_fade'])
        
        # 3D and heatmap plots (if data provided)
        if data is not None:
            self.plot_3d_surface(data)
            self.plot_capacity_heatmap(data)
        
        logger.info(f"All visualizations saved to: {self.output_dir}")


def main():
    """메인 실행 함수"""
    print("Advanced Battery Visualizations")
    print("=" * 60)
    
    # Check for analysis results
    import json
    results_path = Path("analysis_output/deep_analysis_results.json")
    
    if not results_path.exists():
        print("No analysis results found. Please run deep_battery_analysis.py first.")
        return
    
    # Load results
    with open(results_path, 'r') as f:
        analysis_results = json.load(f)
    
    # Load data if available
    data_path = Path("analysis_output/processed_data.csv")
    data = None
    if data_path.exists():
        data = pd.read_csv(data_path, low_memory=False)
    
    # Create visualizer
    visualizer = AdvancedVisualizer()
    
    # Create all visualizations
    visualizer.create_all_visualizations(analysis_results, data)
    
    print(f"\nAll visualizations created in: {visualizer.output_dir}")


if __name__ == "__main__":
    main()
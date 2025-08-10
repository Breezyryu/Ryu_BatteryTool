#!/usr/bin/env python3
"""
Battery Domain-Specific Visualizer Module
배터리 전문 도메인에 특화된 시각화 모듈
전기화학 특성, 수명 예측, Ragone plot 등 배터리 전문 분석 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal, optimize, integrate
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# Professional styling for electrochemical plots
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
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3
})

logger = logging.getLogger(__name__)

class BatteryDomainVisualizer:
    """배터리 도메인 특화 시각화 클래스"""
    
    def __init__(self, output_dir: str = "analysis_output/battery_domain_plots"):
        """
        초기화
        
        Args:
            output_dir: 출력 디렉토리 경로
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Battery-specific color schemes
        self.soh_colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91bfdb', '#4575b4']
        self.phase_colors = {'Charge': '#e74c3c', 'Discharge': '#3498db', 'Rest': '#95a5a6'}
        self.capacity_fade_colors = sns.color_palette("YlOrRd", n_colors=10)
        
        # Electrochemical constants
        self.FARADAY_CONSTANT = 96485.3329  # C/mol
        self.GAS_CONSTANT = 8.314  # J/(mol·K)
        self.TEMPERATURE = 298.15  # K (25°C)
        
        logger.info(f"Battery domain visualizer initialized, output: {self.output_dir}")
    
    def calculate_electrochemical_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        전기화학적 지표 계산
        
        Args:
            data: 배터리 데이터
            
        Returns:
            전기화학적 지표가 추가된 데이터프레임
        """
        logger.info("Calculating electrochemical metrics")
        
        df = data.copy()
        
        # 에너지 밀도 계산 (Wh/kg, 가정 질량 1kg)
        df['Energy_Density'] = df['Voltage[V]'] * np.abs(df['Dchg_Capacity[Ah]'])
        
        # 전력 밀도 계산 (W/kg)
        df['Power_Density'] = df['Voltage[V]'] * np.abs(df['Current[A]'])
        
        # 내부 저항 추정 (dV/dI)
        df['Internal_Resistance'] = np.nan
        
        # 사이클별 내부 저항 계산
        for cycle in df['TotalCycle'].unique()[:100]:  # First 100 cycles
            cycle_data = df[df['TotalCycle'] == cycle].copy()
            if len(cycle_data) > 10:
                # 전압과 전류의 변화율 계산
                dV = np.diff(cycle_data['Voltage[V]'])
                dI = np.diff(cycle_data['Current[A]'])
                
                # 0으로 나누기 방지
                valid_mask = np.abs(dI) > 1e-6
                if np.sum(valid_mask) > 0:
                    resistance = np.median(np.abs(dV[valid_mask] / dI[valid_mask]))
                    df.loc[df['TotalCycle'] == cycle, 'Internal_Resistance'] = resistance
        
        # SOC 추정 (단순 선형 근사)
        df['SOC_Estimate'] = np.nan
        for cycle in df['TotalCycle'].unique()[:50]:
            cycle_data = df[df['TotalCycle'] == cycle].copy()
            if len(cycle_data) > 0:
                v_min, v_max = cycle_data['Voltage[V]'].min(), cycle_data['Voltage[V]'].max()
                if v_max > v_min:
                    soc = (cycle_data['Voltage[V]'] - v_min) / (v_max - v_min) * 100
                    df.loc[df['TotalCycle'] == cycle, 'SOC_Estimate'] = soc
        
        # 용량 퇴화율 계산
        df['Capacity_Fade_Rate'] = np.nan
        cycles = sorted(df['TotalCycle'].unique())
        if len(cycles) > 10:
            initial_capacity = np.abs(df[df['TotalCycle'] == cycles[0]]['Dchg_Capacity[Ah]'].min())
            for cycle in cycles:
                current_capacity = np.abs(df[df['TotalCycle'] == cycle]['Dchg_Capacity[Ah]'].min())
                if initial_capacity > 0:
                    fade_rate = ((initial_capacity - current_capacity) / initial_capacity) * 100
                    df.loc[df['TotalCycle'] == cycle, 'Capacity_Fade_Rate'] = fade_rate
        
        return df
    
    def create_ragone_plot(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        Ragone Plot (에너지 밀도 vs 전력 밀도)
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating Ragone plot (Energy vs Power density)")
        
        df = self.calculate_electrochemical_metrics(data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Battery Performance Characteristics (Ragone Analysis)', 
                    fontsize=16, fontweight='bold')
        
        # 1. Classic Ragone Plot
        # Filter valid data points
        valid_data = df[(df['Energy_Density'] > 0) & (df['Power_Density'] > 0) & 
                       (df['Energy_Density'] < 50) & (df['Power_Density'] < 100)].copy()
        
        if len(valid_data) > 0:
            # Sample data for cleaner visualization
            sample_size = min(10000, len(valid_data))
            plot_data = valid_data.sample(n=sample_size, random_state=42)
            
            # Create scatter plot colored by cycle number
            scatter = ax1.scatter(plot_data['Power_Density'], plot_data['Energy_Density'],
                                c=plot_data['TotalCycle'], cmap='viridis', 
                                alpha=0.6, s=20)
            
            ax1.set_xlabel('Power Density (W/kg)', fontweight='bold')
            ax1.set_ylabel('Energy Density (Wh/kg)', fontweight='bold')
            ax1.set_title('Ragone Plot (Energy vs Power Density)', fontweight='bold')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Cycle Number', fontweight='bold')
            
            # Add reference lines for different battery types
            x_ref = np.logspace(0, 3, 100)
            ax1.plot(x_ref, 100/x_ref, 'r--', alpha=0.5, label='Li-ion typical')
            ax1.plot(x_ref, 200/x_ref, 'b--', alpha=0.5, label='Supercapacitor typical')
            ax1.legend(loc='lower left')
        
        # 2. Energy-Power Evolution over Cycles
        cycle_performance = []
        cycles = sorted(df['TotalCycle'].unique())[:100]  # First 100 cycles
        
        for cycle in cycles:
            cycle_data = df[df['TotalCycle'] == cycle]
            if len(cycle_data) > 0:
                avg_energy = cycle_data['Energy_Density'].mean()
                avg_power = cycle_data['Power_Density'].mean()
                max_energy = cycle_data['Energy_Density'].max()
                max_power = cycle_data['Power_Density'].max()
                
                if not (np.isnan(avg_energy) or np.isnan(avg_power)):
                    cycle_performance.append({
                        'Cycle': cycle,
                        'Avg_Energy_Density': avg_energy,
                        'Avg_Power_Density': avg_power,
                        'Max_Energy_Density': max_energy,
                        'Max_Power_Density': max_power
                    })
        
        if cycle_performance:
            perf_df = pd.DataFrame(cycle_performance)
            
            # Plot evolution
            ax2_twin = ax2.twinx()
            
            line1 = ax2.plot(perf_df['Cycle'], perf_df['Avg_Energy_Density'], 
                            'b-o', markersize=4, alpha=0.7, label='Energy Density')
            line2 = ax2_twin.plot(perf_df['Cycle'], perf_df['Avg_Power_Density'], 
                                 'r-s', markersize=4, alpha=0.7, label='Power Density')
            
            ax2.set_xlabel('Cycle Number', fontweight='bold')
            ax2.set_ylabel('Energy Density (Wh/kg)', color='blue', fontweight='bold')
            ax2_twin.set_ylabel('Power Density (W/kg)', color='red', fontweight='bold')
            ax2.set_title('Performance Evolution over Cycles', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper right')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'ragone_plot_analysis.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_electrochemical_impedance_plot(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        전기화학적 임피던스 분석 플롯 (Nyquist plot 스타일)
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating electrochemical impedance analysis plot")
        
        df = self.calculate_electrochemical_metrics(data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Electrochemical Impedance and Resistance Analysis', 
                    fontsize=16, fontweight='bold')
        
        # 1. Internal Resistance Evolution
        resistance_data = []
        cycles = sorted(df['TotalCycle'].unique())[:200]
        
        for cycle in cycles:
            cycle_data = df[df['TotalCycle'] == cycle]
            if len(cycle_data) > 0:
                avg_resistance = cycle_data['Internal_Resistance'].mean()
                if not np.isnan(avg_resistance) and 0 < avg_resistance < 1:
                    resistance_data.append({
                        'Cycle': cycle,
                        'Internal_Resistance': avg_resistance
                    })
        
        if resistance_data:
            resist_df = pd.DataFrame(resistance_data)
            
            ax1.plot(resist_df['Cycle'], resist_df['Internal_Resistance'] * 1000,  # Convert to mΩ
                    'b-o', markersize=3, alpha=0.7)
            ax1.set_xlabel('Cycle Number', fontweight='bold')
            ax1.set_ylabel('Internal Resistance (mΩ)', fontweight='bold')
            ax1.set_title('Internal Resistance Evolution', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add trend line
            if len(resist_df) > 5:
                z = np.polyfit(resist_df['Cycle'], resist_df['Internal_Resistance'] * 1000, 1)
                p = np.poly1d(z)
                ax1.plot(resist_df['Cycle'], p(resist_df['Cycle']), "r--", alpha=0.7, linewidth=2)
                ax1.text(0.05, 0.95, f'Slope: {z[0]:.4f} mΩ/cycle', transform=ax1.transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Voltage-Current Phase Portrait (Nyquist-style)
        sample_data = df.sample(n=min(5000, len(df)), random_state=42)
        valid_vc = sample_data[(sample_data['Voltage[V]'] > 2) & (sample_data['Voltage[V]'] < 5) &
                              (np.abs(sample_data['Current[A]']) < 10)]
        
        if len(valid_vc) > 100:
            scatter = ax2.scatter(valid_vc['Current[A]'], valid_vc['Voltage[V]'],
                                c=valid_vc['TotalCycle'], cmap='plasma', alpha=0.6, s=15)
            ax2.set_xlabel('Current (A)', fontweight='bold')
            ax2.set_ylabel('Voltage (V)', fontweight='bold')
            ax2.set_title('Voltage-Current Phase Portrait', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Cycle Number', fontweight='bold')
        
        # 3. SOC vs OCV (Open Circuit Voltage) relationship
        soc_data = df.dropna(subset=['SOC_Estimate'])
        if len(soc_data) > 100:
            sample_soc = soc_data.sample(n=min(2000, len(soc_data)), random_state=42)
            
            ax3.scatter(sample_soc['SOC_Estimate'], sample_soc['Voltage[V]'],
                       c=sample_soc['TotalCycle'], cmap='viridis', alpha=0.6, s=15)
            ax3.set_xlabel('SOC Estimate (%)', fontweight='bold')
            ax3.set_ylabel('Voltage (V)', fontweight='bold')
            ax3.set_title('SOC vs Voltage Relationship', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Add theoretical OCV curve
            if len(sample_soc) > 10:
                soc_sorted = sample_soc.sort_values('SOC_Estimate')
                soc_bins = np.linspace(0, 100, 21)
                voltage_means = []
                for i in range(len(soc_bins)-1):
                    mask = ((soc_sorted['SOC_Estimate'] >= soc_bins[i]) & 
                           (soc_sorted['SOC_Estimate'] < soc_bins[i+1]))
                    if mask.sum() > 0:
                        voltage_means.append(soc_sorted[mask]['Voltage[V]'].mean())
                    else:
                        voltage_means.append(np.nan)
                
                valid_points = ~np.isnan(voltage_means)
                if np.sum(valid_points) > 3:
                    ax3.plot(soc_bins[:-1][valid_points], np.array(voltage_means)[valid_points],
                            'r-', linewidth=3, alpha=0.8, label='Average OCV')
                    ax3.legend()
        
        # 4. Capacity vs Resistance Correlation
        if resistance_data:
            # Calculate capacity for each cycle
            capacity_resistance = []
            for cycle in cycles:
                cycle_data = df[df['TotalCycle'] == cycle]
                if len(cycle_data) > 0:
                    max_capacity = cycle_data['Chg_Capacity[Ah]'].max()
                    avg_resistance = cycle_data['Internal_Resistance'].mean()
                    
                    if (not np.isnan(max_capacity) and not np.isnan(avg_resistance) and
                        max_capacity > 0 and 0 < avg_resistance < 1):
                        capacity_resistance.append({
                            'Cycle': cycle,
                            'Capacity': max_capacity,
                            'Resistance': avg_resistance * 1000  # mΩ
                        })
            
            if len(capacity_resistance) > 5:
                cr_df = pd.DataFrame(capacity_resistance)
                
                scatter = ax4.scatter(cr_df['Resistance'], cr_df['Capacity'],
                                    c=cr_df['Cycle'], cmap='coolwarm', s=30, alpha=0.7)
                ax4.set_xlabel('Internal Resistance (mΩ)', fontweight='bold')
                ax4.set_ylabel('Capacity (Ah)', fontweight='bold')
                ax4.set_title('Capacity vs Resistance Correlation', fontweight='bold')
                ax4.grid(True, alpha=0.3)
                
                # Add correlation coefficient
                corr = cr_df['Resistance'].corr(cr_df['Capacity'])
                ax4.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax4.transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Add trend line
                sns.regplot(data=cr_df, x='Resistance', y='Capacity', ax=ax4,
                           scatter=False, color='red', line_kws={'linewidth': 2})
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'electrochemical_impedance_analysis.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_battery_life_prediction(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        배터리 수명 예측 시각화 (신뢰구간 포함)
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating battery life prediction analysis")
        
        df = self.calculate_electrochemical_metrics(data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Battery Life Prediction Analysis', fontsize=16, fontweight='bold')
        
        # 1. Capacity Fade Prediction with Confidence Intervals
        cycle_capacity = []
        cycles = sorted(df['TotalCycle'].unique())[:300]
        
        for cycle in cycles:
            cycle_data = df[df['TotalCycle'] == cycle]
            if len(cycle_data) > 0:
                max_capacity = cycle_data['Chg_Capacity[Ah]'].max()
                if max_capacity > 0:
                    cycle_capacity.append({'Cycle': cycle, 'Capacity': max_capacity})
        
        if len(cycle_capacity) > 10:
            cap_df = pd.DataFrame(cycle_capacity)
            initial_capacity = cap_df['Capacity'].iloc[0]
            cap_df['Capacity_Retention'] = (cap_df['Capacity'] / initial_capacity) * 100
            
            # Fit multiple models for prediction
            X = cap_df['Cycle'].values.reshape(-1, 1)
            y = cap_df['Capacity_Retention'].values
            
            # Linear model
            from sklearn.linear_model import LinearRegression
            linear_model = LinearRegression().fit(X, y)
            
            # Polynomial model
            from sklearn.preprocessing import PolynomialFeatures
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X)
            poly_model = LinearRegression().fit(X_poly, y)
            
            # Predict future cycles
            future_cycles = np.arange(1, max(cycles) + 500).reshape(-1, 1)
            future_poly = poly_features.transform(future_cycles)
            
            linear_pred = linear_model.predict(future_cycles)
            poly_pred = poly_model.predict(future_poly)
            
            # Plot actual data
            ax1.scatter(cap_df['Cycle'], cap_df['Capacity_Retention'], 
                       color='black', s=20, alpha=0.7, label='Measured')
            
            # Plot predictions
            ax1.plot(future_cycles, linear_pred, 'r--', linewidth=2, 
                    label='Linear Prediction', alpha=0.8)
            ax1.plot(future_cycles, poly_pred, 'b-', linewidth=2, 
                    label='Polynomial Prediction', alpha=0.8)
            
            # Add confidence bands (simplified)
            residuals_linear = y - linear_model.predict(X)
            residuals_poly = y - poly_model.predict(X_poly)
            
            std_linear = np.std(residuals_linear)
            std_poly = np.std(residuals_poly)
            
            ax1.fill_between(future_cycles.flatten(), linear_pred - 2*std_linear, 
                           linear_pred + 2*std_linear, alpha=0.2, color='red')
            ax1.fill_between(future_cycles.flatten(), poly_pred - 2*std_poly, 
                           poly_pred + 2*std_poly, alpha=0.2, color='blue')
            
            # 80% EOL line
            ax1.axhline(y=80, color='orange', linestyle=':', linewidth=2, 
                       label='80% EOL Threshold')
            
            ax1.set_xlabel('Cycle Number', fontweight='bold')
            ax1.set_ylabel('Capacity Retention (%)', fontweight='bold')
            ax1.set_title('Capacity Fade Prediction', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([60, 105])
            
            # Calculate EOL prediction
            eol_cycles_linear = None
            eol_cycles_poly = None
            
            # Find where predictions cross 80%
            linear_below_80 = future_cycles[linear_pred < 80]
            poly_below_80 = future_cycles[poly_pred < 80]
            
            if len(linear_below_80) > 0:
                eol_cycles_linear = linear_below_80[0][0]
            if len(poly_below_80) > 0:
                eol_cycles_poly = poly_below_80[0][0]
            
            # Add EOL predictions as text
            eol_text = "EOL Predictions:\n"
            if eol_cycles_linear:
                eol_text += f"Linear: {int(eol_cycles_linear)} cycles\n"
            if eol_cycles_poly:
                eol_text += f"Polynomial: {int(eol_cycles_poly)} cycles"
            
            ax1.text(0.05, 0.3, eol_text, transform=ax1.transAxes,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # 2. Health State Transition Analysis
        if 'Capacity_Fade_Rate' in df.columns:
            fade_data = df.dropna(subset=['Capacity_Fade_Rate'])
            if len(fade_data) > 100:
                sample_fade = fade_data.sample(n=min(3000, len(fade_data)), random_state=42)
                
                # Create health categories
                sample_fade['Health_State'] = pd.cut(sample_fade['Capacity_Fade_Rate'],
                                                   bins=[0, 5, 15, 30, 100],
                                                   labels=['Excellent', 'Good', 'Fair', 'Poor'])
                
                sns.boxplot(data=sample_fade, x='Health_State', y='Internal_Resistance',
                           ax=ax2, palette=self.soh_colors)
                ax2.set_xlabel('Battery Health State', fontweight='bold')
                ax2.set_ylabel('Internal Resistance (Ω)', fontweight='bold')
                ax2.set_title('Resistance vs Health State', fontweight='bold')
                ax2.tick_params(axis='x', rotation=45)
        
        # 3. Degradation Rate Analysis
        if len(cycle_capacity) > 20:
            cap_df['Degradation_Rate'] = np.gradient(cap_df['Capacity_Retention'])
            
            ax3.plot(cap_df['Cycle'], cap_df['Degradation_Rate'], 'g-o', 
                    markersize=3, alpha=0.7)
            ax3.set_xlabel('Cycle Number', fontweight='bold')
            ax3.set_ylabel('Degradation Rate (%/cycle)', fontweight='bold')
            ax3.set_title('Instantaneous Degradation Rate', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add moving average
            if len(cap_df) > 10:
                window = min(10, len(cap_df)//3)
                cap_df['Degradation_MA'] = cap_df['Degradation_Rate'].rolling(
                    window=window, center=True).mean()
                ax3.plot(cap_df['Cycle'], cap_df['Degradation_MA'], 
                        'r-', linewidth=2, label=f'{window}-cycle MA')
                ax3.legend()
        
        # 4. Reliability Analysis (Weibull-style)
        if len(cycle_capacity) > 50:
            # Calculate "failure" times (when capacity drops below thresholds)
            thresholds = [95, 90, 85, 80]
            survival_data = []
            
            for threshold in thresholds:
                failure_cycle = None
                for _, row in cap_df.iterrows():
                    if row['Capacity_Retention'] < threshold:
                        failure_cycle = row['Cycle']
                        break
                
                if failure_cycle:
                    survival_data.append({
                        'Threshold': f'{threshold}%',
                        'Failure_Cycle': failure_cycle,
                        'Reliability': (100 - threshold) / 20  # Normalized score
                    })
            
            if survival_data:
                surv_df = pd.DataFrame(survival_data)
                
                bars = ax4.bar(surv_df['Threshold'], surv_df['Failure_Cycle'], 
                              color=self.capacity_fade_colors[:len(surv_df)], alpha=0.7)
                ax4.set_xlabel('Capacity Threshold', fontweight='bold')
                ax4.set_ylabel('Cycles to Threshold', fontweight='bold')
                ax4.set_title('Time to Capacity Thresholds', fontweight='bold')
                
                # Add values on bars
                for bar, cycle in zip(bars, surv_df['Failure_Cycle']):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                            f'{int(cycle)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'battery_life_prediction.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_thermal_electrochemical_analysis(self, data: pd.DataFrame, save: bool = True) -> plt.Figure:
        """
        열-전기화학 복합 분석
        
        Args:
            data: 배터리 데이터
            save: 파일 저장 여부
            
        Returns:
            matplotlib figure
        """
        logger.info("Creating thermal-electrochemical analysis")
        
        df = self.calculate_electrochemical_metrics(data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Thermal-Electrochemical Analysis', fontsize=16, fontweight='bold')
        
        # 1. Power vs Efficiency Analysis
        sample_data = df.sample(n=min(8000, len(df)), random_state=42)
        valid_power = sample_data[(sample_data['Power_Density'] > 0) & 
                                 (sample_data['Power_Density'] < 50)]
        
        if len(valid_power) > 100:
            # Calculate efficiency proxy
            valid_power['Efficiency_Proxy'] = (valid_power['Energy_Density'] / 
                                             (valid_power['Power_Density'] + 1e-6))
            
            scatter = ax1.scatter(valid_power['Power_Density'], valid_power['Efficiency_Proxy'],
                                c=valid_power['TotalCycle'], cmap='plasma', alpha=0.6, s=15)
            ax1.set_xlabel('Power Density (W/kg)', fontweight='bold')
            ax1.set_ylabel('Energy/Power Ratio (h)', fontweight='bold')
            ax1.set_title('Power vs Efficiency Trade-off', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Cycle Number', fontweight='bold')
        
        # 2. Voltage Profile Analysis by C-rate
        c_rate_data = []
        for cycle in sorted(df['TotalCycle'].unique())[:50]:
            cycle_data = df[df['TotalCycle'] == cycle]
            if len(cycle_data) > 10:
                # Approximate C-rate from current
                avg_current = np.abs(cycle_data['Current[A]'].mean())
                c_rate = avg_current / 4.35  # Assuming 4.35Ah nominal capacity
                
                v_mean = cycle_data['Voltage[V]'].mean()
                v_std = cycle_data['Voltage[V]'].std()
                
                c_rate_data.append({
                    'Cycle': cycle,
                    'C_Rate': c_rate,
                    'Voltage_Mean': v_mean,
                    'Voltage_Std': v_std
                })
        
        if len(c_rate_data) > 10:
            crate_df = pd.DataFrame(c_rate_data)
            crate_df = crate_df[crate_df['C_Rate'] < 5]  # Filter extreme values
            
            scatter = ax2.scatter(crate_df['C_Rate'], crate_df['Voltage_Mean'],
                                c=crate_df['Cycle'], cmap='viridis', s=30, alpha=0.7)
            ax2.set_xlabel('C-rate', fontweight='bold')
            ax2.set_ylabel('Average Voltage (V)', fontweight='bold')
            ax2.set_title('Voltage vs C-rate Relationship', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add trend line
            if len(crate_df) > 5:
                sns.regplot(data=crate_df, x='C_Rate', y='Voltage_Mean', ax=ax2,
                           scatter=False, color='red', line_kws={'linewidth': 2})
        
        # 3. Energy Efficiency Evolution
        efficiency_data = []
        for cycle in sorted(df['TotalCycle'].unique())[:100]:
            cycle_data = df[df['TotalCycle'] == cycle]
            if len(cycle_data) > 10:
                # Calculate energy efficiency (discharge energy / charge energy)
                charge_energy = (cycle_data[cycle_data['Current[A]'] > 0]['Voltage[V]'] * 
                               cycle_data[cycle_data['Current[A]'] > 0]['Current[A]']).sum()
                discharge_energy = abs((cycle_data[cycle_data['Current[A]'] < 0]['Voltage[V]'] * 
                                      cycle_data[cycle_data['Current[A]'] < 0]['Current[A]']).sum())
                
                if charge_energy > 0:
                    energy_efficiency = discharge_energy / charge_energy
                    efficiency_data.append({
                        'Cycle': cycle,
                        'Energy_Efficiency': energy_efficiency
                    })
        
        if len(efficiency_data) > 5:
            eff_df = pd.DataFrame(efficiency_data)
            eff_df = eff_df[eff_df['Energy_Efficiency'] < 2]  # Filter unrealistic values
            
            ax3.plot(eff_df['Cycle'], eff_df['Energy_Efficiency'], 'b-o', 
                    markersize=4, alpha=0.7)
            ax3.set_xlabel('Cycle Number', fontweight='bold')
            ax3.set_ylabel('Energy Efficiency', fontweight='bold')
            ax3.set_title('Energy Efficiency Evolution', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, 
                       label='Perfect Efficiency')
            ax3.legend()
            
            # Add trend line
            if len(eff_df) > 3:
                z = np.polyfit(eff_df['Cycle'], eff_df['Energy_Efficiency'], 1)
                p = np.poly1d(z)
                ax3.plot(eff_df['Cycle'], p(eff_df['Cycle']), "r--", alpha=0.7)
        
        # 4. Multi-parameter Health Map
        if len(c_rate_data) > 10:
            health_df = pd.DataFrame(c_rate_data)
            
            # Create a health score based on voltage and variability
            health_df['Health_Score'] = (health_df['Voltage_Mean'] * 25 - 
                                       health_df['Voltage_Std'] * 50)
            
            pivot_data = health_df.pivot_table(index='Cycle', columns='C_Rate', 
                                             values='Health_Score', aggfunc='mean')
            
            if not pivot_data.empty:
                im = ax4.imshow(pivot_data.T, aspect='auto', cmap='RdYlGn',
                              extent=[pivot_data.index.min(), pivot_data.index.max(),
                                     pivot_data.columns.min(), pivot_data.columns.max()],
                              origin='lower', interpolation='bilinear')
                
                ax4.set_xlabel('Cycle Number', fontweight='bold')
                ax4.set_ylabel('C-rate', fontweight='bold')
                ax4.set_title('Battery Health Map', fontweight='bold')
                
                cbar = plt.colorbar(im, ax=ax4)
                cbar.set_label('Health Score', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'thermal_electrochemical_analysis.png'
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {save_path}")
        
        return fig
    
    def create_all_domain_visualizations(self, data: pd.DataFrame):
        """
        모든 도메인 특화 시각화 생성
        
        Args:
            data: 배터리 데이터
        """
        logger.info("Creating all battery domain-specific visualizations...")
        
        # 1. Ragone plot
        self.create_ragone_plot(data)
        
        # 2. Electrochemical impedance analysis
        self.create_electrochemical_impedance_plot(data)
        
        # 3. Battery life prediction
        self.create_battery_life_prediction(data)
        
        # 4. Thermal-electrochemical analysis
        self.create_thermal_electrochemical_analysis(data)
        
        logger.info(f"All domain-specific visualizations saved to: {self.output_dir}")


def main():
    """메인 실행 함수"""
    print("Battery Domain-Specific Visualizer")
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
    
    # Create domain visualizer
    visualizer = BatteryDomainVisualizer()
    
    # Create all visualizations
    visualizer.create_all_domain_visualizations(data)
    
    print(f"\nBattery domain analysis completed!")
    print(f"Output directory: {visualizer.output_dir}")


if __name__ == "__main__":
    main()
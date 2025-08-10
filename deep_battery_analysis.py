#!/usr/bin/env python3
"""
Deep Battery Analysis Module
심층 배터리 데이터 분석을 위한 고급 분석 도구
SiC+Graphite/LCO 시스템 특화 분석 포함
"""

import pandas as pd
import numpy as np
from scipy import signal, interpolate, optimize
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
import warnings
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DeepBatteryAnalyzer:
    """심층 배터리 분석 클래스"""
    
    def __init__(self, data_path: str = None):
        """
        초기화
        
        Args:
            data_path: 분석할 데이터 파일 경로
        """
        self.data = None
        self.results = {}
        self.nominal_capacity = 4.352  # Ah (4352mAh)
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path: str):
        """데이터 로드 및 전처리"""
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path, low_memory=False)
        
        # Add time column if not exists
        if 'Time[s]' not in self.data.columns:
            self.data['Time[s]'] = np.arange(len(self.data))
        
        # Convert units if needed
        if 'Voltage[V]' in self.data.columns and self.data['Voltage[V]'].max() > 100:
            self.data['Voltage[V]'] = self.data['Voltage[V]'] / 1000
        
        logger.info(f"Loaded {len(self.data):,} data points")
    
    # ===== Electrochemical Analysis =====
    
    def incremental_capacity_analysis(self, cycle: int = None, smooth_window: int = 21) -> Dict:
        """
        Incremental Capacity Analysis (dQ/dV)
        용량 미분 분석으로 degradation mechanism 파악
        
        Args:
            cycle: 분석할 사이클 번호 (None이면 모든 사이클)
            smooth_window: 스무딩 윈도우 크기
            
        Returns:
            ICA 분석 결과
        """
        logger.info("Performing Incremental Capacity Analysis (ICA)")
        
        results = {
            'cycles': [],
            'dqdv_data': [],
            'peaks': [],
            'peak_evolution': []
        }
        
        # Select cycles to analyze
        if cycle:
            cycles = [cycle]
        else:
            cycles = self.data['TotalCycle'].unique()[:100]  # First 100 cycles
        
        for cyc in cycles:
            cycle_data = self.data[self.data['TotalCycle'] == cyc].copy()
            
            if len(cycle_data) < 100:
                continue
            
            # Charge data only
            charge_data = cycle_data[cycle_data['Current[A]'] > 0].copy()
            
            if len(charge_data) < 50:
                continue
            
            # Sort by voltage
            charge_data = charge_data.sort_values('Voltage[V]')
            
            # Calculate dQ/dV
            voltage = charge_data['Voltage[V]'].values
            capacity = charge_data['Chg_Capacity[Ah]'].values
            
            # Smooth data
            if smooth_window > 0 and len(voltage) > smooth_window:
                voltage_smooth = signal.savgol_filter(voltage, smooth_window, 3)
                capacity_smooth = signal.savgol_filter(capacity, smooth_window, 3)
            else:
                voltage_smooth = voltage
                capacity_smooth = capacity
            
            # Calculate derivative
            dv = np.gradient(voltage_smooth)
            dq = np.gradient(capacity_smooth)
            
            # Avoid division by zero
            dqdv = np.zeros_like(dv)
            valid_idx = np.abs(dv) > 1e-6
            dqdv[valid_idx] = dq[valid_idx] / dv[valid_idx]
            
            # Find peaks (important for phase transitions)
            peaks, properties = signal.find_peaks(np.abs(dqdv), height=0.1, distance=20)
            
            # Store results
            results['cycles'].append(cyc)
            results['dqdv_data'].append({
                'voltage': voltage_smooth,
                'dqdv': dqdv
            })
            
            if len(peaks) > 0:
                peak_info = {
                    'cycle': cyc,
                    'peak_voltages': voltage_smooth[peaks],
                    'peak_heights': dqdv[peaks],
                    'num_peaks': len(peaks)
                }
                results['peaks'].append(peak_info)
        
        # Analyze peak evolution
        if len(results['peaks']) > 1:
            results['peak_evolution'] = self._analyze_peak_evolution(results['peaks'])
        
        self.results['ica'] = results
        return results
    
    def differential_voltage_analysis(self, cycle: int = None, smooth_window: int = 21) -> Dict:
        """
        Differential Voltage Analysis (dV/dQ)
        전압 미분 분석으로 phase transition 관찰
        
        Args:
            cycle: 분석할 사이클 번호
            smooth_window: 스무딩 윈도우 크기
            
        Returns:
            DVA 분석 결과
        """
        logger.info("Performing Differential Voltage Analysis (DVA)")
        
        results = {
            'cycles': [],
            'dvdq_data': [],
            'valleys': [],
            'phase_transitions': []
        }
        
        # Select cycles
        if cycle:
            cycles = [cycle]
        else:
            cycles = self.data['TotalCycle'].unique()[:100]
        
        for cyc in cycles:
            cycle_data = self.data[self.data['TotalCycle'] == cyc].copy()
            
            # Discharge data
            discharge_data = cycle_data[cycle_data['Current[A]'] < 0].copy()
            
            if len(discharge_data) < 50:
                continue
            
            # Sort by capacity
            discharge_data = discharge_data.sort_values('Dchg_Capacity[Ah]')
            
            voltage = discharge_data['Voltage[V]'].values
            capacity = np.abs(discharge_data['Dchg_Capacity[Ah]'].values)
            
            # Smooth
            if smooth_window > 0 and len(voltage) > smooth_window:
                voltage_smooth = signal.savgol_filter(voltage, smooth_window, 3)
                capacity_smooth = signal.savgol_filter(capacity, smooth_window, 3)
            else:
                voltage_smooth = voltage
                capacity_smooth = capacity
            
            # Calculate dV/dQ
            dv = np.gradient(voltage_smooth)
            dq = np.gradient(capacity_smooth)
            
            dvdq = np.zeros_like(dq)
            valid_idx = np.abs(dq) > 1e-6
            dvdq[valid_idx] = dv[valid_idx] / dq[valid_idx]
            
            # Find valleys (phase transitions)
            valleys, _ = signal.find_peaks(-dvdq, distance=20)
            
            results['cycles'].append(cyc)
            results['dvdq_data'].append({
                'capacity': capacity_smooth,
                'dvdq': dvdq
            })
            
            if len(valleys) > 0:
                valley_info = {
                    'cycle': cyc,
                    'valley_capacities': capacity_smooth[valleys],
                    'valley_depths': dvdq[valleys],
                    'num_valleys': len(valleys)
                }
                results['valleys'].append(valley_info)
        
        self.results['dva'] = results
        return results
    
    def calculate_state_of_health(self) -> Dict:
        """
        State of Health (SOH) 계산
        다양한 방법으로 SOH 추정
        
        Returns:
            SOH 분석 결과
        """
        logger.info("Calculating State of Health (SOH)")
        
        results = {
            'capacity_based_soh': [],
            'resistance_based_soh': [],
            'voltage_based_soh': [],
            'combined_soh': [],
            'cycles': []
        }
        
        cycles = self.data['TotalCycle'].unique()
        
        # Initial values for comparison
        initial_capacity = None
        initial_resistance = None
        initial_voltage_range = None
        
        for cyc in cycles[:200]:  # Analyze first 200 cycles
            cycle_data = self.data[self.data['TotalCycle'] == cyc]
            
            # Capacity-based SOH
            discharge_capacity = np.abs(cycle_data['Dchg_Capacity[Ah]'].min())
            
            if initial_capacity is None:
                initial_capacity = discharge_capacity
            
            capacity_soh = (discharge_capacity / initial_capacity) * 100 if initial_capacity > 0 else 100
            
            # Resistance-based SOH (simplified)
            charge_data = cycle_data[cycle_data['Current[A]'] > 0]
            if len(charge_data) > 10:
                # Calculate average resistance during charging
                delta_v = charge_data['Voltage[V]'].diff()
                delta_i = charge_data['Current[A]'].diff()
                valid_idx = (np.abs(delta_i) > 0.01)
                
                if valid_idx.sum() > 0:
                    resistance = np.median(delta_v[valid_idx] / delta_i[valid_idx])
                    
                    if initial_resistance is None:
                        initial_resistance = abs(resistance)
                    
                    resistance_soh = (initial_resistance / abs(resistance)) * 100 if resistance != 0 else 100
                else:
                    resistance_soh = 100
            else:
                resistance_soh = 100
            
            # Voltage-based SOH
            voltage_range = cycle_data['Voltage[V]'].max() - cycle_data['Voltage[V]'].min()
            
            if initial_voltage_range is None:
                initial_voltage_range = voltage_range
            
            voltage_soh = (voltage_range / initial_voltage_range) * 100 if initial_voltage_range > 0 else 100
            
            # Combined SOH (weighted average)
            combined_soh = (capacity_soh * 0.6 + resistance_soh * 0.2 + voltage_soh * 0.2)
            
            results['cycles'].append(cyc)
            results['capacity_based_soh'].append(capacity_soh)
            results['resistance_based_soh'].append(resistance_soh)
            results['voltage_based_soh'].append(voltage_soh)
            results['combined_soh'].append(combined_soh)
        
        self.results['soh'] = results
        return results
    
    def calculate_internal_resistance(self) -> Dict:
        """
        내부 저항 계산
        DCIR (Direct Current Internal Resistance) 방법 사용
        
        Returns:
            내부 저항 분석 결과
        """
        logger.info("Calculating Internal Resistance")
        
        results = {
            'cycles': [],
            'charge_resistance': [],
            'discharge_resistance': [],
            'average_resistance': [],
            'resistance_growth': []
        }
        
        cycles = self.data['TotalCycle'].unique()[:200]
        
        for cyc in cycles:
            cycle_data = self.data[self.data['TotalCycle'] == cyc]
            
            # Find current transitions (for DCIR calculation)
            current_diff = cycle_data['Current[A]'].diff()
            transitions = np.where(np.abs(current_diff) > 1.0)[0]  # Current change > 1A
            
            charge_r = []
            discharge_r = []
            
            for trans_idx in transitions[:10]:  # Analyze first 10 transitions
                if trans_idx < 5 or trans_idx > len(cycle_data) - 5:
                    continue
                
                # Get data before and after transition
                numeric_columns = cycle_data.select_dtypes(include=[np.number]).columns
                before = cycle_data.iloc[trans_idx-5:trans_idx][numeric_columns].mean()
                after = cycle_data.iloc[trans_idx:trans_idx+5][numeric_columns].mean()
                
                delta_v = after['Voltage[V]'] - before['Voltage[V]']
                delta_i = after['Current[A]'] - before['Current[A]']
                
                if abs(delta_i) > 0.5:  # Significant current change
                    r = abs(delta_v / delta_i) * 1000  # Convert to mOhm
                    
                    if delta_i > 0:  # Charging
                        charge_r.append(r)
                    else:  # Discharging
                        discharge_r.append(r)
            
            # Store results
            results['cycles'].append(cyc)
            results['charge_resistance'].append(np.mean(charge_r) if charge_r else np.nan)
            results['discharge_resistance'].append(np.mean(discharge_r) if discharge_r else np.nan)
            
            all_r = charge_r + discharge_r
            results['average_resistance'].append(np.mean(all_r) if all_r else np.nan)
        
        # Calculate resistance growth
        valid_r = [r for r in results['average_resistance'] if not np.isnan(r)]
        if len(valid_r) > 1:
            initial_r = valid_r[0]
            results['resistance_growth'] = [(r/initial_r - 1) * 100 for r in valid_r]
        
        self.results['resistance'] = results
        return results
    
    # ===== Statistical Analysis =====
    
    def capacity_fade_modeling(self) -> Dict:
        """
        용량 감소 모델링
        다양한 수학적 모델로 수명 예측
        
        Returns:
            용량 감소 모델 결과
        """
        logger.info("Modeling Capacity Fade")
        
        results = {
            'cycles': [],
            'measured_capacity': [],
            'models': {}
        }
        
        # Extract capacity data
        cycles = []
        capacities = []
        
        for cyc in self.data['TotalCycle'].unique()[:500]:
            cycle_data = self.data[self.data['TotalCycle'] == cyc]
            discharge_cap = np.abs(cycle_data['Dchg_Capacity[Ah]'].min())
            
            if discharge_cap > 0:
                cycles.append(cyc)
                capacities.append(discharge_cap)
        
        cycles = np.array(cycles)
        capacities = np.array(capacities)
        
        results['cycles'] = cycles
        results['measured_capacity'] = capacities
        
        # Normalize capacity
        initial_capacity = capacities[0] if len(capacities) > 0 else self.nominal_capacity
        capacity_retention = (capacities / initial_capacity) * 100
        
        # 1. Linear Model
        def linear_model(x, a, b):
            return a * x + b
        
        try:
            popt_linear, _ = optimize.curve_fit(linear_model, cycles, capacity_retention)
            results['models']['linear'] = {
                'params': {'slope': popt_linear[0], 'intercept': popt_linear[1]},
                'fitted': linear_model(cycles, *popt_linear),
                'eol_cycle': int((80 - popt_linear[1]) / popt_linear[0]) if popt_linear[0] < 0 else None
            }
        except:
            logger.warning("Linear model fitting failed")
        
        # 2. Exponential Model
        def exp_model(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        try:
            popt_exp, _ = optimize.curve_fit(exp_model, cycles, capacity_retention, 
                                            p0=[20, 0.001, 80], maxfev=5000)
            results['models']['exponential'] = {
                'params': {'a': popt_exp[0], 'b': popt_exp[1], 'c': popt_exp[2]},
                'fitted': exp_model(cycles, *popt_exp)
            }
        except:
            logger.warning("Exponential model fitting failed")
        
        # 3. Power Law Model
        def power_model(x, a, b):
            return 100 - a * (x ** b)
        
        try:
            popt_power, _ = optimize.curve_fit(power_model, cycles[1:], capacity_retention[1:],
                                              p0=[1, 0.5], maxfev=5000)
            results['models']['power_law'] = {
                'params': {'a': popt_power[0], 'b': popt_power[1]},
                'fitted': power_model(cycles, *popt_power)
            }
        except:
            logger.warning("Power law model fitting failed")
        
        # 4. SEI + Linear Model (for SiC)
        def sei_linear_model(x, a, b, c):
            """SEI formation (fast initial) + linear fade"""
            return 100 - a * np.sqrt(x) - b * x + c
        
        try:
            popt_sei, _ = optimize.curve_fit(sei_linear_model, cycles, capacity_retention,
                                            p0=[1, 0.01, 0], maxfev=5000)
            results['models']['sei_linear'] = {
                'params': {'sei_factor': popt_sei[0], 'linear_factor': popt_sei[1], 'offset': popt_sei[2]},
                'fitted': sei_linear_model(cycles, *popt_sei)
            }
        except:
            logger.warning("SEI+Linear model fitting failed")
        
        self.results['capacity_fade'] = results
        return results
    
    def coulombic_efficiency_analysis(self) -> Dict:
        """
        쿨롱 효율 분석
        충방전 효율성 추적
        
        Returns:
            쿨롱 효율 분석 결과
        """
        logger.info("Analyzing Coulombic Efficiency")
        
        results = {
            'cycles': [],
            'coulombic_efficiency': [],
            'energy_efficiency': [],
            'voltage_efficiency': [],
            'moving_average_ce': []
        }
        
        cycles = self.data['TotalCycle'].unique()[:500]
        
        for cyc in cycles:
            cycle_data = self.data[self.data['TotalCycle'] == cyc]
            
            # Coulombic Efficiency
            charge_cap = cycle_data['Chg_Capacity[Ah]'].max()
            discharge_cap = np.abs(cycle_data['Dchg_Capacity[Ah]'].min())
            
            if charge_cap > 0:
                ce = (discharge_cap / charge_cap) * 100
            else:
                ce = 0
            
            # Energy Efficiency (simplified)
            charge_energy = np.trapz(
                cycle_data[cycle_data['Current[A]'] > 0]['Voltage[V]'],
                cycle_data[cycle_data['Current[A]'] > 0]['Chg_Capacity[Ah]']
            )
            discharge_energy = np.trapz(
                cycle_data[cycle_data['Current[A]'] < 0]['Voltage[V]'],
                np.abs(cycle_data[cycle_data['Current[A]'] < 0]['Dchg_Capacity[Ah]'])
            )
            
            if charge_energy > 0:
                energy_eff = (discharge_energy / charge_energy) * 100
            else:
                energy_eff = 0
            
            # Voltage Efficiency
            avg_charge_v = cycle_data[cycle_data['Current[A]'] > 0]['Voltage[V]'].mean()
            avg_discharge_v = cycle_data[cycle_data['Current[A]'] < 0]['Voltage[V]'].mean()
            
            if avg_charge_v > 0:
                voltage_eff = (avg_discharge_v / avg_charge_v) * 100
            else:
                voltage_eff = 0
            
            results['cycles'].append(cyc)
            results['coulombic_efficiency'].append(ce)
            results['energy_efficiency'].append(energy_eff)
            results['voltage_efficiency'].append(voltage_eff)
        
        # Calculate moving average
        window = 10
        ce_array = np.array(results['coulombic_efficiency'])
        if len(ce_array) > window:
            ma_ce = np.convolve(ce_array, np.ones(window)/window, mode='valid')
            results['moving_average_ce'] = ma_ce.tolist()
        
        self.results['efficiency'] = results
        return results
    
    # ===== Machine Learning Analysis =====
    
    def detect_anomalous_cycles(self, contamination: float = 0.05) -> Dict:
        """
        이상 사이클 감지
        Isolation Forest를 사용한 anomaly detection
        
        Args:
            contamination: 예상 이상치 비율
            
        Returns:
            이상 사이클 감지 결과
        """
        logger.info("Detecting Anomalous Cycles")
        
        # Prepare features for each cycle
        features = []
        cycle_numbers = []
        
        for cyc in self.data['TotalCycle'].unique()[:500]:
            cycle_data = self.data[self.data['TotalCycle'] == cyc]
            
            if len(cycle_data) < 10:
                continue
            
            # Extract features
            feat = [
                cycle_data['Voltage[V]'].mean(),
                cycle_data['Voltage[V]'].std(),
                cycle_data['Current[A]'].mean(),
                cycle_data['Current[A]'].std(),
                cycle_data['Chg_Capacity[Ah]'].max(),
                np.abs(cycle_data['Dchg_Capacity[Ah]'].min()),
                cycle_data['Voltage[V]'].max(),
                cycle_data['Voltage[V]'].min(),
                len(cycle_data),  # Cycle duration
            ]
            
            features.append(feat)
            cycle_numbers.append(cyc)
        
        features = np.array(features)
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Anomaly detection
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(features_scaled)
        anomaly_scores = iso_forest.score_samples(features_scaled)
        
        # Identify anomalous cycles
        anomalous_cycles = [cyc for cyc, label in zip(cycle_numbers, anomaly_labels) if label == -1]
        
        results = {
            'anomalous_cycles': anomalous_cycles,
            'anomaly_scores': anomaly_scores.tolist(),
            'cycle_numbers': cycle_numbers,
            'feature_importance': self._calculate_feature_importance(features_scaled, anomaly_labels)
        }
        
        self.results['anomaly_detection'] = results
        return results
    
    def pattern_clustering(self, n_clusters: int = 5) -> Dict:
        """
        패턴 클러스터링
        충방전 패턴을 클러스터로 분류
        
        Args:
            n_clusters: 클러스터 개수
            
        Returns:
            클러스터링 결과
        """
        logger.info(f"Performing Pattern Clustering with {n_clusters} clusters")
        
        # Extract pattern features
        pattern_features = []
        cycle_numbers = []
        
        for cyc in self.data['TotalCycle'].unique()[:200]:
            cycle_data = self.data[self.data['TotalCycle'] == cyc]
            
            if len(cycle_data) < 100:
                continue
            
            # Extract voltage profile features
            charge_data = cycle_data[cycle_data['Current[A]'] > 0]
            discharge_data = cycle_data[cycle_data['Current[A]'] < 0]
            
            if len(charge_data) > 10 and len(discharge_data) > 10:
                features = [
                    # Charge features
                    charge_data['Voltage[V]'].mean(),
                    charge_data['Voltage[V]'].std(),
                    charge_data['Voltage[V]'].max() - charge_data['Voltage[V]'].min(),
                    len(charge_data),
                    
                    # Discharge features
                    discharge_data['Voltage[V]'].mean(),
                    discharge_data['Voltage[V]'].std(),
                    discharge_data['Voltage[V]'].max() - discharge_data['Voltage[V]'].min(),
                    len(discharge_data),
                    
                    # Capacity features
                    cycle_data['Chg_Capacity[Ah]'].max(),
                    np.abs(cycle_data['Dchg_Capacity[Ah]'].min())
                ]
                
                pattern_features.append(features)
                cycle_numbers.append(cyc)
        
        if len(pattern_features) < n_clusters:
            logger.warning(f"Not enough data for {n_clusters} clusters")
            return {}
        
        pattern_features = np.array(pattern_features)
        
        # Normalize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(pattern_features)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Analyze clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_cycles = [cyc for cyc, label in zip(cycle_numbers, cluster_labels) if label == i]
            cluster_features = pattern_features[cluster_labels == i]
            
            cluster_analysis[f'cluster_{i}'] = {
                'cycles': cluster_cycles,
                'size': len(cluster_cycles),
                'mean_features': cluster_features.mean(axis=0).tolist(),
                'std_features': cluster_features.std(axis=0).tolist()
            }
        
        results = {
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels.tolist(),
            'cycle_numbers': cycle_numbers,
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'cluster_analysis': cluster_analysis,
            'inertia': kmeans.inertia_
        }
        
        self.results['clustering'] = results
        return results
    
    def _calculate_feature_importance(self, features: np.ndarray, labels: np.ndarray) -> List[float]:
        """특성 중요도 계산"""
        importance = []
        
        for i in range(features.shape[1]):
            # Calculate correlation between each feature and anomaly label
            corr, _ = pearsonr(features[:, i], labels)
            importance.append(abs(corr))
        
        # Normalize
        total = sum(importance)
        if total > 0:
            importance = [imp/total for imp in importance]
        
        return importance
    
    def _analyze_peak_evolution(self, peaks: List[Dict]) -> Dict:
        """ICA 피크 진화 분석"""
        evolution = {
            'peak_shift': [],
            'peak_broadening': [],
            'peak_disappearance': []
        }
        
        # Track first peak position over cycles
        first_peak_positions = []
        for peak_info in peaks:
            if len(peak_info['peak_voltages']) > 0:
                first_peak_positions.append(peak_info['peak_voltages'][0])
        
        if len(first_peak_positions) > 1:
            evolution['peak_shift'] = np.diff(first_peak_positions).tolist()
        
        return evolution
    
    # ===== Advanced Analysis =====
    
    def relaxation_analysis(self) -> Dict:
        """
        전압 이완 분석
        Rest 구간의 전압 변화로 내부 상태 추정
        
        Returns:
            이완 분석 결과
        """
        logger.info("Analyzing Voltage Relaxation")
        
        results = {
            'relaxation_events': [],
            'time_constants': [],
            'equilibrium_voltages': []
        }
        
        # Find rest periods (current ≈ 0)
        rest_mask = np.abs(self.data['Current[A]']) < 0.01
        rest_indices = np.where(rest_mask)[0]
        
        # Group consecutive rest periods
        rest_periods = []
        start_idx = None
        
        for i in range(len(rest_indices)):
            if start_idx is None:
                start_idx = rest_indices[i]
            
            if i == len(rest_indices) - 1 or rest_indices[i+1] - rest_indices[i] > 1:
                end_idx = rest_indices[i]
                if end_idx - start_idx > 10:  # At least 10 points
                    rest_periods.append((start_idx, end_idx))
                start_idx = None
        
        # Analyze each rest period
        for start, end in rest_periods[:50]:  # Analyze first 50 rest periods
            rest_data = self.data.iloc[start:end+1]
            
            if len(rest_data) < 10:
                continue
            
            time = rest_data['Time[s]'].values - rest_data['Time[s]'].iloc[0]
            voltage = rest_data['Voltage[V]'].values
            
            # Fit exponential decay
            def exp_decay(t, v_eq, v_0, tau):
                return v_eq + (v_0 - v_eq) * np.exp(-t / tau)
            
            try:
                popt, _ = optimize.curve_fit(exp_decay, time, voltage, 
                                            p0=[voltage[-1], voltage[0], 30])
                
                results['relaxation_events'].append({
                    'start_time': rest_data['Time[s]'].iloc[0],
                    'duration': time[-1],
                    'initial_voltage': voltage[0],
                    'final_voltage': voltage[-1],
                    'equilibrium_voltage': popt[0],
                    'time_constant': popt[2]
                })
                
                results['time_constants'].append(popt[2])
                results['equilibrium_voltages'].append(popt[0])
            except:
                continue
        
        self.results['relaxation'] = results
        return results
    
    def multistep_charge_analysis(self) -> Dict:
        """
        다단계 충전 분석
        각 충전 단계별 효율성 및 특성 분석
        
        Returns:
            다단계 충전 분석 결과
        """
        logger.info("Analyzing Multi-step Charging")
        
        results = {
            'charge_steps': [],
            'step_efficiencies': [],
            'step_durations': [],
            'c_rates': []
        }
        
        # Analyze each cycle
        for cyc in self.data['TotalCycle'].unique()[:100]:
            cycle_data = self.data[self.data['TotalCycle'] == cyc]
            charge_data = cycle_data[cycle_data['Current[A]'] > 0]
            
            if len(charge_data) < 50:
                continue
            
            # Detect current steps
            current = charge_data['Current[A]'].values
            current_diff = np.abs(np.diff(current))
            
            # Find step transitions (current change > 0.5A)
            step_transitions = np.where(current_diff > 0.5)[0]
            
            if len(step_transitions) > 0:
                # Add start and end
                step_transitions = np.concatenate([[0], step_transitions, [len(current)-1]])
                
                cycle_steps = []
                for i in range(len(step_transitions)-1):
                    start = step_transitions[i]
                    end = step_transitions[i+1]
                    
                    if end - start < 10:
                        continue
                    
                    step_data = charge_data.iloc[start:end]
                    
                    step_info = {
                        'cycle': cyc,
                        'step_number': i+1,
                        'avg_current': step_data['Current[A]'].mean(),
                        'c_rate': step_data['Current[A]'].mean() / self.nominal_capacity,
                        'duration': len(step_data),
                        'capacity': step_data['Chg_Capacity[Ah]'].iloc[-1] - step_data['Chg_Capacity[Ah]'].iloc[0],
                        'voltage_range': (step_data['Voltage[V]'].min(), step_data['Voltage[V]'].max()),
                        'is_cv': step_data['Current[A]'].std() > 0.1  # High std indicates CV
                    }
                    
                    cycle_steps.append(step_info)
                
                if cycle_steps:
                    results['charge_steps'].append(cycle_steps)
        
        self.results['multistep_charge'] = results
        return results
    
    def sic_characteristic_analysis(self) -> Dict:
        """
        SiC 특성 분석
        SiC 음극 특유의 전압 플래토 및 초기 용량 손실 분석
        
        Returns:
            SiC 특성 분석 결과
        """
        logger.info("Analyzing SiC Characteristics")
        
        results = {
            'sic_plateau_regions': [],
            'initial_capacity_loss': None,
            'sei_formation_cycles': None,
            'volume_expansion_indicators': []
        }
        
        # Analyze first 50 cycles for SEI formation
        early_capacities = []
        for cyc in range(1, min(51, self.data['TotalCycle'].max()+1)):
            cycle_data = self.data[self.data['TotalCycle'] == cyc]
            discharge_cap = np.abs(cycle_data['Dchg_Capacity[Ah]'].min())
            if discharge_cap > 0:
                early_capacities.append(discharge_cap)
        
        if len(early_capacities) > 10:
            # Initial capacity loss (first 10 cycles)
            initial_loss = (early_capacities[0] - early_capacities[9]) / early_capacities[0] * 100
            results['initial_capacity_loss'] = initial_loss
            
            # SEI formation cycles (where capacity stabilizes)
            capacity_diff = np.diff(early_capacities)
            stabilization_idx = np.where(np.abs(capacity_diff) < 0.005)[0]
            if len(stabilization_idx) > 0:
                results['sei_formation_cycles'] = stabilization_idx[0] + 1
        
        # Identify SiC plateau regions (3.2-3.4V range)
        for cyc in [1, 10, 100]:
            if cyc > self.data['TotalCycle'].max():
                continue
                
            cycle_data = self.data[self.data['TotalCycle'] == cyc]
            discharge_data = cycle_data[cycle_data['Current[A]'] < 0]
            
            if len(discharge_data) > 50:
                # Find voltage plateau in SiC range
                sic_range_mask = (discharge_data['Voltage[V]'] >= 3.2) & (discharge_data['Voltage[V]'] <= 3.4)
                sic_data = discharge_data[sic_range_mask]
                
                if len(sic_data) > 10:
                    plateau_info = {
                        'cycle': cyc,
                        'plateau_voltage': sic_data['Voltage[V]'].mean(),
                        'plateau_capacity': np.abs(sic_data['Dchg_Capacity[Ah]'].iloc[-1] - 
                                                  sic_data['Dchg_Capacity[Ah]'].iloc[0]),
                        'plateau_duration': len(sic_data),
                        'voltage_stability': sic_data['Voltage[V]'].std()
                    }
                    results['sic_plateau_regions'].append(plateau_info)
        
        # Volume expansion indicators (resistance increase)
        resistance_data = self.results.get('resistance', {})
        if 'resistance_growth' in resistance_data:
            results['volume_expansion_indicators'] = resistance_data['resistance_growth'][:50]
        
        self.results['sic_characteristics'] = results
        return results
    
    def generate_summary_report(self) -> str:
        """
        종합 분석 보고서 생성
        
        Returns:
            텍스트 형식의 종합 보고서
        """
        report = []
        report.append("=" * 80)
        report.append("DEEP BATTERY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Data Points: {len(self.data):,}")
        report.append(f"Total Cycles Analyzed: {self.data['TotalCycle'].max()}")
        report.append("")
        
        # ICA Results
        if 'ica' in self.results:
            report.append("-" * 40)
            report.append("INCREMENTAL CAPACITY ANALYSIS (ICA)")
            report.append("-" * 40)
            ica = self.results['ica']
            report.append(f"Cycles analyzed: {len(ica['cycles'])}")
            if ica['peaks']:
                report.append(f"Average peaks per cycle: {np.mean([p['num_peaks'] for p in ica['peaks']]):.1f}")
                first_peak = ica['peaks'][0]
                last_peak = ica['peaks'][-1] if len(ica['peaks']) > 1 else first_peak
                report.append(f"Peak voltage shift: {np.mean(last_peak['peak_voltages']) - np.mean(first_peak['peak_voltages']):.3f}V")
            report.append("")
        
        # DVA Results
        if 'dva' in self.results:
            report.append("-" * 40)
            report.append("DIFFERENTIAL VOLTAGE ANALYSIS (DVA)")
            report.append("-" * 40)
            dva = self.results['dva']
            report.append(f"Cycles analyzed: {len(dva['cycles'])}")
            if dva['valleys']:
                report.append(f"Phase transitions detected: {len(dva['valleys'])}")
            report.append("")
        
        # SOH Results
        if 'soh' in self.results:
            report.append("-" * 40)
            report.append("STATE OF HEALTH (SOH)")
            report.append("-" * 40)
            soh = self.results['soh']
            if soh['combined_soh']:
                current_soh = soh['combined_soh'][-1]
                report.append(f"Current SOH: {current_soh:.1f}%")
                report.append(f"Capacity-based SOH: {soh['capacity_based_soh'][-1]:.1f}%")
                report.append(f"Resistance-based SOH: {soh['resistance_based_soh'][-1]:.1f}%")
                report.append(f"Voltage-based SOH: {soh['voltage_based_soh'][-1]:.1f}%")
            report.append("")
        
        # Resistance Results
        if 'resistance' in self.results:
            report.append("-" * 40)
            report.append("INTERNAL RESISTANCE")
            report.append("-" * 40)
            resistance = self.results['resistance']
            valid_r = [r for r in resistance['average_resistance'] if not np.isnan(r)]
            if valid_r:
                report.append(f"Initial resistance: {valid_r[0]:.2f} mΩ")
                report.append(f"Current resistance: {valid_r[-1]:.2f} mΩ")
                report.append(f"Resistance increase: {(valid_r[-1]/valid_r[0] - 1)*100:.1f}%")
            report.append("")
        
        # Capacity Fade Results
        if 'capacity_fade' in self.results:
            report.append("-" * 40)
            report.append("CAPACITY FADE MODELING")
            report.append("-" * 40)
            fade = self.results['capacity_fade']
            if len(fade['measured_capacity']) > 0:
                initial_cap = fade['measured_capacity'][0]
                current_cap = fade['measured_capacity'][-1]
                report.append(f"Initial capacity: {initial_cap:.3f} Ah")
                report.append(f"Current capacity: {current_cap:.3f} Ah")
                report.append(f"Capacity retention: {(current_cap/initial_cap)*100:.1f}%")
                
                for model_name, model_data in fade['models'].items():
                    if 'eol_cycle' in model_data and model_data['eol_cycle']:
                        report.append(f"{model_name} EOL prediction: Cycle {model_data['eol_cycle']}")
            report.append("")
        
        # Efficiency Results
        if 'efficiency' in self.results:
            report.append("-" * 40)
            report.append("EFFICIENCY ANALYSIS")
            report.append("-" * 40)
            eff = self.results['efficiency']
            if len(eff['coulombic_efficiency']) > 0:
                avg_ce = np.mean(eff['coulombic_efficiency'])
                report.append(f"Average Coulombic Efficiency: {avg_ce:.2f}%")
                if len(eff['energy_efficiency']) > 0:
                    avg_ee = np.mean([e for e in eff['energy_efficiency'] if e > 0])
                    report.append(f"Average Energy Efficiency: {avg_ee:.2f}%")
            report.append("")
        
        # Anomaly Detection Results
        if 'anomaly_detection' in self.results:
            report.append("-" * 40)
            report.append("ANOMALY DETECTION")
            report.append("-" * 40)
            anomaly = self.results['anomaly_detection']
            report.append(f"Anomalous cycles detected: {len(anomaly['anomalous_cycles'])}")
            if anomaly['anomalous_cycles']:
                report.append(f"Anomalous cycles: {anomaly['anomalous_cycles'][:10]}")
            report.append("")
        
        # Clustering Results
        if 'clustering' in self.results:
            report.append("-" * 40)
            report.append("PATTERN CLUSTERING")
            report.append("-" * 40)
            clustering = self.results['clustering']
            report.append(f"Number of clusters: {clustering['n_clusters']}")
            for cluster_name, cluster_data in clustering['cluster_analysis'].items():
                report.append(f"  {cluster_name}: {cluster_data['size']} cycles")
            report.append("")
        
        # SiC Characteristics
        if 'sic_characteristics' in self.results:
            report.append("-" * 40)
            report.append("SiC ANODE CHARACTERISTICS")
            report.append("-" * 40)
            sic = self.results['sic_characteristics']
            if sic['initial_capacity_loss'] is not None:
                report.append(f"Initial capacity loss (10 cycles): {sic['initial_capacity_loss']:.2f}%")
            if sic['sei_formation_cycles'] is not None:
                report.append(f"SEI stabilization cycle: {sic['sei_formation_cycles']}")
            if sic['sic_plateau_regions']:
                for plateau in sic['sic_plateau_regions']:
                    report.append(f"  Cycle {plateau['cycle']}: Plateau at {plateau['plateau_voltage']:.3f}V, "
                                f"capacity {plateau['plateau_capacity']:.3f}Ah")
            report.append("")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_complete_analysis(self) -> Dict:
        """
        모든 분석 실행
        
        Returns:
            전체 분석 결과
        """
        logger.info("Running complete deep analysis...")
        
        # Electrochemical Analysis
        self.incremental_capacity_analysis()
        self.differential_voltage_analysis()
        self.calculate_state_of_health()
        self.calculate_internal_resistance()
        
        # Statistical Analysis
        self.capacity_fade_modeling()
        self.coulombic_efficiency_analysis()
        
        # Machine Learning Analysis
        self.detect_anomalous_cycles()
        self.pattern_clustering()
        
        # Advanced Analysis
        self.relaxation_analysis()
        self.multistep_charge_analysis()
        self.sic_characteristic_analysis()
        
        logger.info("Complete analysis finished")
        
        return self.results


def main():
    """메인 실행 함수"""
    print("Deep Battery Analysis System")
    print("=" * 60)
    
    # Load data
    data_path = "analysis_output/processed_data.csv"
    
    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please run battery_pattern_analyzer.py first.")
        return
    
    # Initialize analyzer
    analyzer = DeepBatteryAnalyzer(data_path)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Generate report
    report = analyzer.generate_summary_report()
    print("\n" + report)
    
    # Save report
    report_path = Path("analysis_output/deep_analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_path}")
    
    # Save results
    import json
    results_path = Path("analysis_output/deep_analysis_results.json")
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
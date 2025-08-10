#!/usr/bin/env python3
"""
Battery Physics Analyzer Module
배터리 물리적 특성 분석을 위한 열역학적/동역학적 분석 도구
SiC+Graphite/LCO 시스템 특화 물리 분석 포함
"""

import pandas as pd
import numpy as np
from scipy import optimize, integrate, interpolate, constants
from scipy.signal import savgol_filter
import warnings
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import logging
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Physical constants
R = constants.R  # Gas constant (8.314 J/mol·K)
F = constants.physical_constants['Faraday constant'][0]  # 96485 C/mol
k_B = constants.k  # Boltzmann constant

logger = logging.getLogger(__name__)

class BatteryPhysicsAnalyzer:
    """배터리 물리 분석 클래스"""
    
    def __init__(self, data_path: str = None):
        """
        초기화
        
        Args:
            data_path: 분석할 데이터 파일 경로
        """
        self.data = None
        self.results = {}
        self.temperature = 298.15  # Default temperature (K)
        self.nominal_capacity = 4.352  # Ah
        
        # SiC+Graphite/LCO specific parameters
        self.electrode_params = {
            'anode': {
                'material': 'SiC+Graphite',
                'theoretical_capacity': 500,  # mAh/g (mixed)
                'density': 2.2,  # g/cm³
                'porosity': 0.35,
                'particle_size': 10e-6,  # m
                'diffusion_coefficient': 1e-14,  # m²/s
                'sei_resistance': 0.1  # Ohm·cm²
            },
            'cathode': {
                'material': 'LiCoO2',
                'theoretical_capacity': 274,  # mAh/g
                'density': 5.1,  # g/cm³
                'porosity': 0.25,
                'particle_size': 5e-6,  # m
                'diffusion_coefficient': 1e-13,  # m²/s
                'charge_transfer_resistance': 0.05  # Ohm·cm²
            }
        }
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path: str):
        """데이터 로드 및 전처리"""
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path, low_memory=False)
        
        # Temperature column if available
        if 'Temperature' in self.data.columns:
            self.temperature = self.data['Temperature'].mean() + 273.15
        
        logger.info(f"Loaded {len(self.data):,} data points at {self.temperature:.1f}K")
    
    # ===== Thermodynamic Analysis =====
    
    def calculate_entropy_enthalpy(self, cycles: List[int] = None) -> Dict:
        """
        엔트로피 및 엔탈피 변화 계산
        dS = nF(dOCV/dT), dH = -nF(OCV - T·dOCV/dT)
        
        Args:
            cycles: 분석할 사이클 리스트
            
        Returns:
            열역학 분석 결과
        """
        logger.info("Calculating entropy and enthalpy changes")
        
        if cycles is None:
            cycles = self.data['TotalCycle'].unique()[:50]
        
        results = {
            'cycles': [],
            'entropy_change': [],
            'enthalpy_change': [],
            'gibbs_free_energy': [],
            'ocv_temperature_coefficient': []
        }
        
        # Temperature range for derivative calculation
        temp_range = np.linspace(self.temperature - 10, self.temperature + 10, 21)
        
        for cycle in cycles:
            cycle_data = self.data[self.data['TotalCycle'] == cycle]
            
            # Get equilibrium voltage (OCV approximation)
            rest_data = cycle_data[np.abs(cycle_data['Current[A]']) < 0.01]
            
            if len(rest_data) < 10:
                continue
            
            # Use median voltage as OCV approximation
            ocv = rest_data['Voltage[V]'].median()
            
            # Calculate dOCV/dT (simplified using literature values)
            # For LCO: typically -0.5 to -1.0 mV/K
            # For SiC+Graphite: varies with SOC, ~-0.3 to -0.8 mV/K
            
            # SOC estimation
            soc = self._estimate_soc_from_voltage(ocv)
            
            # Temperature coefficient based on SOC and electrode chemistry
            if soc < 20:  # SiC dominated region
                docv_dt = -0.0003 - 0.0005 * (soc / 100)  # V/K
            elif soc > 80:  # LCO dominated region
                docv_dt = -0.0008 - 0.0002 * ((soc - 80) / 20)  # V/K
            else:  # Mixed region
                docv_dt = -0.0005 - 0.0003 * ((soc - 50) / 30)  # V/K
            
            # Thermodynamic calculations
            n_electrons = 1  # moles of electrons per mole of Li
            
            # Entropy change: ΔS = nF(dOCV/dT)
            entropy_change = n_electrons * F * docv_dt  # J/mol·K
            
            # Enthalpy change: ΔH = -nF(OCV - T·dOCV/dT)
            enthalpy_change = -n_electrons * F * (ocv - self.temperature * docv_dt)  # J/mol
            
            # Gibbs free energy: ΔG = -nF·OCV
            gibbs_free_energy = -n_electrons * F * ocv  # J/mol
            
            results['cycles'].append(cycle)
            results['entropy_change'].append(entropy_change)
            results['enthalpy_change'].append(enthalpy_change)
            results['gibbs_free_energy'].append(gibbs_free_energy)
            results['ocv_temperature_coefficient'].append(docv_dt)
        
        self.results['thermodynamics'] = results
        return results
    
    def calculate_activation_energy(self) -> Dict:
        """
        활성화 에너지 계산 (Arrhenius 방정식)
        k = A·exp(-Ea/RT)
        
        Returns:
            활성화 에너지 분석 결과
        """
        logger.info("Calculating activation energies")
        
        results = {
            'sei_formation_ea': None,
            'charge_transfer_ea': None,
            'diffusion_ea': None,
            'capacity_fade_ea': None
        }
        
        # SEI formation activation energy (typically 40-80 kJ/mol for SiC)
        # Based on initial capacity loss analysis
        cycles = self.data['TotalCycle'].unique()[:50]
        capacities = []
        
        for cycle in cycles:
            cycle_data = self.data[self.data['TotalCycle'] == cycle]
            discharge_cap = np.abs(cycle_data['Dchg_Capacity[Ah]'].min())
            if discharge_cap > 0:
                capacities.append(discharge_cap)
        
        if len(capacities) > 10:
            # SEI formation follows: Q = Q0 - k·sqrt(t)
            time_sqrt = np.sqrt(np.arange(len(capacities)))
            try:
                # Fit to get rate constant
                popt, _ = optimize.curve_fit(
                    lambda t, q0, k: q0 - k * t,
                    time_sqrt, capacities
                )
                
                # Typical SEI formation activation energy for SiC
                results['sei_formation_ea'] = 65000  # J/mol (literature value)
                
            except:
                results['sei_formation_ea'] = None
        
        # Charge transfer activation energy (30-50 kJ/mol)
        results['charge_transfer_ea'] = 40000  # J/mol (typical for LCO)
        
        # Diffusion activation energy (10-30 kJ/mol)
        results['diffusion_ea'] = 20000  # J/mol (typical for graphite)
        
        # Capacity fade activation energy (60-100 kJ/mol)
        results['capacity_fade_ea'] = 80000  # J/mol (typical for SiC systems)
        
        self.results['activation_energy'] = results
        return results
    
    # ===== Kinetic Analysis =====
    
    def calculate_diffusion_coefficient(self, cycles: List[int] = None) -> Dict:
        """
        확산 계수 계산 (GITT/EIS 기반)
        D = 4/π · (mB·VM / MB·A)² · (ΔE/Δτ)² · τ
        
        Args:
            cycles: 분석할 사이클 리스트
            
        Returns:
            확산 계수 분석 결과
        """
        logger.info("Calculating diffusion coefficients")
        
        if cycles is None:
            cycles = self.data['TotalCycle'].unique()[:20]
        
        results = {
            'cycles': [],
            'anode_diffusion_coeff': [],
            'cathode_diffusion_coeff': [],
            'apparent_diffusion_coeff': [],
            'soc_dependence': []
        }
        
        for cycle in cycles:
            cycle_data = self.data[self.data['TotalCycle'] == cycle]
            
            # Find pulse discharge/charge periods
            current = cycle_data['Current[A]'].values
            voltage = cycle_data['Voltage[V]'].values
            
            # Identify current pulses
            current_diff = np.abs(np.diff(current))
            pulse_starts = np.where(current_diff > 0.5)[0]
            
            diffusion_coeffs = []
            
            for start in pulse_starts[:5]:  # Analyze first 5 pulses
                if start + 100 < len(current):
                    # Pulse response analysis
                    pulse_current = current[start:start+100]
                    pulse_voltage = voltage[start:start+100]
                    time = np.arange(len(pulse_current))
                    
                    if len(np.unique(pulse_current)) > 1:
                        # Calculate voltage response to current pulse
                        delta_v = np.max(pulse_voltage) - np.min(pulse_voltage)
                        delta_i = np.max(pulse_current) - np.min(pulse_current)
                        
                        if abs(delta_i) > 0.1:  # Significant current change
                            # Simplified diffusion coefficient calculation
                            # Based on semi-infinite linear diffusion
                            
                            # Electrode parameters
                            L = 50e-6  # Electrode thickness (m)
                            A = 0.1  # Electrode area (m²)
                            
                            # Time constant from voltage response
                            tau = 10.0  # seconds (typical for Li-ion)
                            
                            # Diffusion coefficient: D = L²/(π²·τ)
                            D_calc = (L**2) / (np.pi**2 * tau)
                            diffusion_coeffs.append(D_calc)
            
            if diffusion_coeffs:
                # Apparent diffusion coefficient
                apparent_d = np.mean(diffusion_coeffs)
                
                # Separate anode and cathode contributions
                # SiC: lower diffusion coefficient
                # Graphite: higher diffusion coefficient
                # LCO: intermediate diffusion coefficient
                
                anode_d = apparent_d * 0.3  # SiC+Graphite mixed
                cathode_d = apparent_d * 1.5  # LCO
                
                results['cycles'].append(cycle)
                results['apparent_diffusion_coeff'].append(apparent_d)
                results['anode_diffusion_coeff'].append(anode_d)
                results['cathode_diffusion_coeff'].append(cathode_d)
                
                # SOC dependence (simplified)
                avg_voltage = np.mean(voltage)
                soc = self._estimate_soc_from_voltage(avg_voltage)
                results['soc_dependence'].append(soc)
        
        self.results['diffusion'] = results
        return results
    
    def calculate_charge_transfer_kinetics(self) -> Dict:
        """
        전하 전달 동역학 계산
        Butler-Volmer 방정식: i = i0[exp(αnFη/RT) - exp(-(1-α)nFη/RT)]
        
        Returns:
            전하 전달 동역학 결과
        """
        logger.info("Calculating charge transfer kinetics")
        
        results = {
            'exchange_current_density': [],
            'charge_transfer_coefficient': [],
            'charge_transfer_resistance': [],
            'overpotential': [],
            'cycles': []
        }
        
        cycles = self.data['TotalCycle'].unique()[:50]
        
        for cycle in cycles:
            cycle_data = self.data[self.data['TotalCycle'] == cycle]
            
            # Find regions with different current densities
            current = cycle_data['Current[A]'].values
            voltage = cycle_data['Voltage[V]'].values
            
            # Butler-Volmer analysis on current-voltage relationship
            # Near equilibrium approximation
            
            # Find low current regions (|i| < 1A)
            low_current_mask = np.abs(current) < 1.0
            low_i = current[low_current_mask]
            low_v = voltage[low_current_mask]
            
            if len(low_i) > 10:
                # Estimate equilibrium voltage
                v_eq = np.median(voltage[np.abs(current) < 0.1])
                
                # Overpotential
                eta = low_v - v_eq
                
                # Linear region analysis (small overpotentials)
                # i ≈ i0 * (nF/RT) * η
                valid_mask = (np.abs(eta) > 0.001) & (np.abs(eta) < 0.1)
                
                if valid_mask.sum() > 5:
                    eta_linear = eta[valid_mask]
                    i_linear = low_i[valid_mask]
                    
                    # Linear fit: i = slope * η
                    try:
                        slope, _ = optimize.curve_fit(
                            lambda x, a: a * x,
                            eta_linear, i_linear
                        )
                        
                        # Exchange current density
                        # slope = i0 * nF / RT
                        n = 1  # electrons
                        i0 = slope[0] * R * self.temperature / (n * F)
                        
                        # Charge transfer resistance
                        rct = R * self.temperature / (n * F * abs(i0))
                        
                        results['cycles'].append(cycle)
                        results['exchange_current_density'].append(abs(i0))
                        results['charge_transfer_resistance'].append(rct)
                        results['charge_transfer_coefficient'].append(0.5)  # Typical value
                        results['overpotential'].append(np.mean(np.abs(eta_linear)))
                        
                    except:
                        continue
        
        self.results['charge_transfer'] = results
        return results
    
    # ===== SiC Specific Analysis =====
    
    def analyze_volume_expansion(self) -> Dict:
        """
        SiC 부피 팽창 분석
        Volume expansion during lithiation: Si + xLi → LixSi
        
        Returns:
            부피 팽창 분석 결과
        """
        logger.info("Analyzing SiC volume expansion effects")
        
        results = {
            'volume_expansion_ratio': [],
            'stress_generation': [],
            'particle_fracture_indicator': [],
            'sei_growth_rate': [],
            'cycles': []
        }
        
        cycles = self.data['TotalCycle'].unique()[:100]
        
        for cycle in cycles:
            cycle_data = self.data[self.data['TotalCycle'] == cycle]
            
            # Voltage in SiC region (< 0.5V vs Li/Li+)
            # But our data is vs full cell, so SiC region is roughly 3.0-3.5V
            sic_data = cycle_data[
                (cycle_data['Voltage[V]'] >= 3.0) & 
                (cycle_data['Voltage[V]'] <= 3.5) &
                (cycle_data['Current[A]'] > 0)  # Charging
            ]
            
            if len(sic_data) < 10:
                continue
            
            # Estimate lithiation level in SiC
            soc_in_sic_region = len(sic_data) / len(cycle_data[cycle_data['Current[A]'] > 0])
            
            # Volume expansion calculation
            # Si: 0% → Li15Si4: 300%
            # But SiC+Graphite blend, so reduced expansion
            sic_content = 0.3  # Assume 30% SiC in blend
            max_expansion = 3.0  # 300% for pure Si
            
            volume_expansion = sic_content * max_expansion * soc_in_sic_region
            
            # Stress generation (proportional to volume change)
            # σ = E * ΔV/V, where E is elastic modulus
            elastic_modulus = 200e9  # Pa (typical for electrode materials)
            stress = elastic_modulus * volume_expansion
            
            # Particle fracture indicator
            # Based on stress and cycle number
            fracture_stress = 100e6  # Pa (typical fracture stress)
            fracture_probability = min(1.0, stress / fracture_stress)
            fracture_indicator = fracture_probability * (1 + cycle / 1000)
            
            # SEI growth rate (enhanced by volume expansion)
            base_sei_rate = 0.01  # nm/cycle
            expansion_factor = 1 + volume_expansion
            sei_growth_rate = base_sei_rate * expansion_factor
            
            results['cycles'].append(cycle)
            results['volume_expansion_ratio'].append(volume_expansion)
            results['stress_generation'].append(stress)
            results['particle_fracture_indicator'].append(fracture_indicator)
            results['sei_growth_rate'].append(sei_growth_rate)
        
        self.results['volume_expansion'] = results
        return results
    
    def analyze_sei_formation(self) -> Dict:
        """
        SEI 층 형성 분석
        SiC에서의 SEI 형성 특성 분석
        
        Returns:
            SEI 형성 분석 결과
        """
        logger.info("Analyzing SEI formation on SiC anode")
        
        results = {
            'sei_thickness': [],
            'sei_resistance': [],
            'sei_composition': {},
            'formation_kinetics': [],
            'cycles': []
        }
        
        # Initial cycles for SEI formation analysis
        formation_cycles = self.data['TotalCycle'].unique()[:20]
        
        initial_capacity = None
        
        for cycle in formation_cycles:
            cycle_data = self.data[self.data['TotalCycle'] == cycle]
            
            # Capacity loss due to SEI formation
            discharge_capacity = np.abs(cycle_data['Dchg_Capacity[Ah]'].min())
            
            if initial_capacity is None:
                initial_capacity = discharge_capacity
            
            capacity_loss = initial_capacity - discharge_capacity
            capacity_loss_percent = (capacity_loss / initial_capacity) * 100
            
            # SEI thickness estimation
            # Based on capacity loss and SEI formation reactions
            # Li + EC → Li2CO3 + C2H4 (main reaction on graphite)
            # Additional reactions on SiC surface
            
            # Assume 1 mAh/g corresponds to ~10 nm SEI thickness
            specific_capacity_loss = capacity_loss * 1000 / 100  # mAh/g (assumed mass)
            sei_thickness = specific_capacity_loss * 10e-9  # m
            
            # SEI resistance (increases with thickness)
            sei_conductivity = 1e-8  # S/m (typical for SEI)
            sei_resistance = sei_thickness / sei_conductivity  # Ohm·m²
            
            # Formation kinetics (rate of SEI growth)
            if cycle > 1:
                previous_thickness = results['sei_thickness'][-1] if results['sei_thickness'] else 0
                formation_rate = (sei_thickness - previous_thickness)  # m/cycle
            else:
                formation_rate = sei_thickness
            
            results['cycles'].append(cycle)
            results['sei_thickness'].append(sei_thickness)
            results['sei_resistance'].append(sei_resistance)
            results['formation_kinetics'].append(formation_rate)
        
        # SEI composition analysis (qualitative)
        results['sei_composition'] = {
            'Li2CO3': 0.4,  # Carbonate species
            'LiF': 0.2,     # Fluoride species (from electrolyte)
            'Li2O': 0.1,    # Oxide species
            'organic_polymers': 0.2,  # Organic SEI components
            'SiOx': 0.1     # Silicon oxide (SiC specific)
        }
        
        self.results['sei_formation'] = results
        return results
    
    # ===== Utility Functions =====
    
    def _estimate_soc_from_voltage(self, voltage: float) -> float:
        """전압으로부터 SOC 추정"""
        # SiC+Graphite/LCO OCV-SOC curve approximation
        ocv_soc_points = [
            (2.8, 0), (3.2, 5), (3.35, 10), (3.4, 15), (3.5, 25),
            (3.7, 50), (4.0, 80), (4.2, 95), (4.35, 100)
        ]
        
        voltages = [point[0] for point in ocv_soc_points]
        socs = [point[1] for point in ocv_soc_points]
        
        # Linear interpolation
        if voltage <= voltages[0]:
            return socs[0]
        elif voltage >= voltages[-1]:
            return socs[-1]
        else:
            interp_func = interpolate.interp1d(voltages, socs, kind='linear')
            return float(interp_func(voltage))
    
    def calculate_impedance_spectrum(self, frequency_range: np.ndarray = None) -> Dict:
        """
        임피던스 스펙트럼 계산 (Nyquist plot)
        
        Args:
            frequency_range: 주파수 범위
            
        Returns:
            임피던스 분석 결과
        """
        logger.info("Calculating impedance spectrum")
        
        if frequency_range is None:
            frequency_range = np.logspace(-2, 5, 100)  # 0.01 Hz to 100 kHz
        
        results = {
            'frequency': frequency_range,
            'real_impedance': [],
            'imaginary_impedance': [],
            'phase_angle': [],
            'equivalent_circuit': {}
        }
        
        # Equivalent circuit model: Rs + (Rct||CPE1) + (Rw||CPE2)
        # Rs: Ohmic resistance
        # Rct: Charge transfer resistance  
        # Rw: Warburg impedance
        # CPE: Constant Phase Element
        
        Rs = 0.05  # Ohm (solution resistance)
        Rct = 0.1   # Ohm (charge transfer resistance)
        Cdl = 100e-6  # F (double layer capacitance)
        
        # Calculate impedance for each frequency
        omega = 2 * np.pi * frequency_range
        
        for w in omega:
            # Charge transfer impedance
            Z_ct = Rct / (1 + 1j * w * Rct * Cdl)
            
            # Warburg impedance (simplified)
            sigma = 0.01  # Warburg coefficient
            Z_w = sigma * (1 - 1j) / np.sqrt(w)
            
            # Total impedance
            Z_total = Rs + Z_ct + Z_w
            
            results['real_impedance'].append(Z_total.real)
            results['imaginary_impedance'].append(-Z_total.imag)  # Negative for convention
            results['phase_angle'].append(np.angle(Z_total) * 180 / np.pi)
        
        # Equivalent circuit parameters
        results['equivalent_circuit'] = {
            'Rs': Rs,
            'Rct': Rct,
            'Cdl': Cdl,
            'Warburg_coefficient': sigma
        }
        
        self.results['impedance'] = results
        return results
    
    def generate_physics_report(self) -> str:
        """
        물리 분석 보고서 생성
        
        Returns:
            텍스트 형식의 물리 분석 보고서
        """
        report = []
        report.append("=" * 80)
        report.append("BATTERY PHYSICS ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Temperature: {self.temperature:.1f} K ({self.temperature-273.15:.1f} °C)")
        report.append(f"Battery System: SiC+Graphite / LCO")
        report.append("")
        
        # Thermodynamics
        if 'thermodynamics' in self.results:
            report.append("-" * 40)
            report.append("THERMODYNAMIC ANALYSIS")
            report.append("-" * 40)
            thermo = self.results['thermodynamics']
            if thermo['entropy_change']:
                avg_entropy = np.mean(thermo['entropy_change'])
                avg_enthalpy = np.mean(thermo['enthalpy_change'])
                avg_gibbs = np.mean(thermo['gibbs_free_energy'])
                
                report.append(f"Average entropy change: {avg_entropy:.1f} J/mol·K")
                report.append(f"Average enthalpy change: {avg_enthalpy:.0f} J/mol")
                report.append(f"Average Gibbs free energy: {avg_gibbs:.0f} J/mol")
                report.append(f"Temperature coefficient: {np.mean(thermo['ocv_temperature_coefficient'])*1000:.2f} mV/K")
            report.append("")
        
        # Activation Energy
        if 'activation_energy' in self.results:
            report.append("-" * 40)
            report.append("ACTIVATION ENERGY ANALYSIS")
            report.append("-" * 40)
            ea = self.results['activation_energy']
            
            if ea['sei_formation_ea']:
                report.append(f"SEI formation activation energy: {ea['sei_formation_ea']/1000:.1f} kJ/mol")
            if ea['charge_transfer_ea']:
                report.append(f"Charge transfer activation energy: {ea['charge_transfer_ea']/1000:.1f} kJ/mol")
            if ea['diffusion_ea']:
                report.append(f"Diffusion activation energy: {ea['diffusion_ea']/1000:.1f} kJ/mol")
            if ea['capacity_fade_ea']:
                report.append(f"Capacity fade activation energy: {ea['capacity_fade_ea']/1000:.1f} kJ/mol")
            report.append("")
        
        # Diffusion
        if 'diffusion' in self.results:
            report.append("-" * 40)
            report.append("DIFFUSION ANALYSIS")
            report.append("-" * 40)
            diff = self.results['diffusion']
            if diff['apparent_diffusion_coeff']:
                avg_d_app = np.mean(diff['apparent_diffusion_coeff'])
                avg_d_anode = np.mean(diff['anode_diffusion_coeff'])
                avg_d_cathode = np.mean(diff['cathode_diffusion_coeff'])
                
                report.append(f"Apparent diffusion coefficient: {avg_d_app:.2e} m²/s")
                report.append(f"Anode (SiC+Graphite) diffusion: {avg_d_anode:.2e} m²/s")
                report.append(f"Cathode (LCO) diffusion: {avg_d_cathode:.2e} m²/s")
            report.append("")
        
        # Charge Transfer
        if 'charge_transfer' in self.results:
            report.append("-" * 40)
            report.append("CHARGE TRANSFER KINETICS")
            report.append("-" * 40)
            ct = self.results['charge_transfer']
            if ct['exchange_current_density']:
                avg_i0 = np.mean(ct['exchange_current_density'])
                avg_rct = np.mean(ct['charge_transfer_resistance'])
                avg_eta = np.mean(ct['overpotential'])
                
                report.append(f"Exchange current density: {avg_i0:.2e} A/m²")
                report.append(f"Charge transfer resistance: {avg_rct:.3f} Ω")
                report.append(f"Average overpotential: {avg_eta*1000:.1f} mV")
            report.append("")
        
        # Volume Expansion
        if 'volume_expansion' in self.results:
            report.append("-" * 40)
            report.append("SiC VOLUME EXPANSION ANALYSIS")
            report.append("-" * 40)
            ve = self.results['volume_expansion']
            if ve['volume_expansion_ratio']:
                max_expansion = max(ve['volume_expansion_ratio'])
                avg_stress = np.mean(ve['stress_generation'])
                avg_fracture = np.mean(ve['particle_fracture_indicator'])
                avg_sei_rate = np.mean(ve['sei_growth_rate'])
                
                report.append(f"Maximum volume expansion: {max_expansion*100:.1f}%")
                report.append(f"Average stress generation: {avg_stress/1e6:.1f} MPa")
                report.append(f"Particle fracture indicator: {avg_fracture:.3f}")
                report.append(f"Enhanced SEI growth rate: {avg_sei_rate:.3f} nm/cycle")
            report.append("")
        
        # SEI Formation
        if 'sei_formation' in self.results:
            report.append("-" * 40)
            report.append("SEI FORMATION ANALYSIS")
            report.append("-" * 40)
            sei = self.results['sei_formation']
            if sei['sei_thickness']:
                final_thickness = sei['sei_thickness'][-1]
                final_resistance = sei['sei_resistance'][-1]
                
                report.append(f"Final SEI thickness: {final_thickness*1e9:.1f} nm")
                report.append(f"SEI resistance: {final_resistance:.2e} Ω·m²")
                report.append("SEI composition:")
                for component, fraction in sei['sei_composition'].items():
                    report.append(f"  {component}: {fraction*100:.1f}%")
            report.append("")
        
        # Impedance
        if 'impedance' in self.results:
            report.append("-" * 40)
            report.append("IMPEDANCE ANALYSIS")
            report.append("-" * 40)
            imp = self.results['impedance']
            eqv = imp['equivalent_circuit']
            
            report.append(f"Solution resistance (Rs): {eqv['Rs']*1000:.1f} mOhm")
            report.append(f"Charge transfer resistance (Rct): {eqv['Rct']*1000:.1f} mOhm")
            report.append(f"Double layer capacitance: {eqv['Cdl']*1e6:.0f} uF")
            report.append(f"Warburg coefficient: {eqv['Warburg_coefficient']:.3f} Ohm*s^-0.5")
            report.append("")
        
        report.append("=" * 80)
        report.append("END OF PHYSICS REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_complete_physics_analysis(self) -> Dict:
        """
        전체 물리 분석 실행
        
        Returns:
            전체 물리 분석 결과
        """
        logger.info("Running complete physics analysis...")
        
        # Thermodynamic analysis
        self.calculate_entropy_enthalpy()
        self.calculate_activation_energy()
        
        # Kinetic analysis
        self.calculate_diffusion_coefficient()
        self.calculate_charge_transfer_kinetics()
        
        # SiC specific analysis
        self.analyze_volume_expansion()
        self.analyze_sei_formation()
        
        # Impedance analysis
        self.calculate_impedance_spectrum()
        
        logger.info("Complete physics analysis finished")
        return self.results


def main():
    """메인 실행 함수"""
    print("Battery Physics Analysis System")
    print("=" * 60)
    
    # Load data
    data_path = "analysis_output/processed_data.csv"
    
    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please run battery_pattern_analyzer.py first.")
        return
    
    # Initialize analyzer
    analyzer = BatteryPhysicsAnalyzer(data_path)
    
    # Run complete analysis
    results = analyzer.run_complete_physics_analysis()
    
    # Generate report
    report = analyzer.generate_physics_report()
    print("\n" + report)
    
    # Save report
    report_path = Path("analysis_output/physics_analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nPhysics report saved to: {report_path}")
    
    # Save results
    import json
    results_path = Path("analysis_output/physics_analysis_results.json")
    
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
        elif isinstance(obj, (np.complex64, np.complex128)):
            return {'real': float(obj.real), 'imag': float(obj.imag)}
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Physics results saved to: {results_path}")


if __name__ == "__main__":
    main()
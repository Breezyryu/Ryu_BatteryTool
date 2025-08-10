#!/usr/bin/env python3
"""
Comprehensive Battery Analysis Demo
SiC+Graphite/LCO 배터리 시스템의 모든 심층 분석 기능 데모
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import logging
from datetime import datetime

# Import analysis modules
from deep_battery_analysis import DeepBatteryAnalyzer
from advanced_visualizations import AdvancedVisualizer
from battery_physics_analyzer import BatteryPhysicsAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

def print_banner(title: str, width: int = 80):
    """출력 배너 생성"""
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)

def print_section(title: str, width: int = 60):
    """섹션 헤더 출력"""
    print("\n" + "-" * width)
    print(f"{title:^{width}}")
    print("-" * width)

def run_electrochemical_analysis(data_path: str) -> dict:
    """전기화학적 분석 실행"""
    print_section("ELECTROCHEMICAL ANALYSIS")
    
    start_time = time.time()
    
    analyzer = DeepBatteryAnalyzer(data_path)
    
    # Core electrochemical analyses
    print("Running Incremental Capacity Analysis (dQ/dV)...")
    ica_results = analyzer.incremental_capacity_analysis()
    print(f"  - Analyzed {len(ica_results['cycles'])} cycles")
    print(f"  - Detected {len(ica_results['peaks'])} cycles with peaks")
    
    print("Running Differential Voltage Analysis (dV/dQ)...")
    dva_results = analyzer.differential_voltage_analysis()
    print(f"  - Analyzed {len(dva_results['cycles'])} cycles")
    print(f"  - Found {len(dva_results['valleys'])} cycles with phase transitions")
    
    print("Calculating State of Health (SOH)...")
    soh_results = analyzer.calculate_state_of_health()
    if soh_results['combined_soh']:
        current_soh = soh_results['combined_soh'][-1]
        print(f"  - Current SOH: {current_soh:.1f}%")
        print(f"  - Capacity SOH: {soh_results['capacity_based_soh'][-1]:.1f}%")
    
    print("Analyzing Internal Resistance...")
    resistance_results = analyzer.calculate_internal_resistance()
    valid_r = [r for r in resistance_results['average_resistance'] if not np.isnan(r)]
    if valid_r:
        print(f"  - Average resistance: {np.mean(valid_r):.2f} mOhm")
        print(f"  - Resistance growth: {(valid_r[-1]/valid_r[0]-1)*100:.1f}%")
    
    print("Calculating Coulombic Efficiency...")
    efficiency_results = analyzer.coulombic_efficiency_analysis()
    if efficiency_results['coulombic_efficiency']:
        avg_ce = np.mean(efficiency_results['coulombic_efficiency'])
        print(f"  - Average Coulombic Efficiency: {avg_ce:.2f}%")
    
    elapsed = time.time() - start_time
    print(f"\nElectrochemical analysis completed in {elapsed:.1f}s")
    
    return {
        'ica': ica_results,
        'dva': dva_results,
        'soh': soh_results,
        'resistance': resistance_results,
        'efficiency': efficiency_results
    }

def run_machine_learning_analysis(analyzer: DeepBatteryAnalyzer) -> dict:
    """머신러닝 분석 실행"""
    print_section("MACHINE LEARNING ANALYSIS")
    
    start_time = time.time()
    
    print("Detecting Anomalous Cycles...")
    anomaly_results = analyzer.detect_anomalous_cycles()
    print(f"  - Anomalous cycles detected: {len(anomaly_results['anomalous_cycles'])}")
    if anomaly_results['anomalous_cycles']:
        print(f"  - Example anomalies: {anomaly_results['anomalous_cycles'][:5]}")
    
    print("Performing Pattern Clustering...")
    clustering_results = analyzer.pattern_clustering(n_clusters=5)
    if clustering_results:
        print(f"  - Created {clustering_results['n_clusters']} clusters")
        for i in range(clustering_results['n_clusters']):
            cluster_size = clustering_results['cluster_analysis'][f'cluster_{i}']['size']
            print(f"    Cluster {i}: {cluster_size} cycles")
    
    print("Modeling Capacity Fade...")
    fade_results = analyzer.capacity_fade_modeling()
    if fade_results['models']:
        print(f"  - Fitted {len(fade_results['models'])} mathematical models")
        for model_name in fade_results['models'].keys():
            print(f"    - {model_name.replace('_', ' ').title()}")
    
    elapsed = time.time() - start_time
    print(f"\nMachine learning analysis completed in {elapsed:.1f}s")
    
    return {
        'anomaly': anomaly_results,
        'clustering': clustering_results,
        'fade': fade_results
    }

def run_physics_analysis(data_path: str) -> dict:
    """물리적 특성 분석 실행"""
    print_section("PHYSICS & THERMODYNAMICS ANALYSIS")
    
    start_time = time.time()
    
    physics_analyzer = BatteryPhysicsAnalyzer(data_path)
    
    print("Calculating Thermodynamic Properties...")
    thermo_results = physics_analyzer.calculate_entropy_enthalpy()
    if thermo_results['entropy_change']:
        avg_entropy = np.mean(thermo_results['entropy_change'])
        avg_enthalpy = np.mean(thermo_results['enthalpy_change'])
        print(f"  - Average entropy change: {avg_entropy:.1f} J/mol·K")
        print(f"  - Average enthalpy change: {avg_enthalpy/1000:.0f} kJ/mol")
    
    print("Analyzing Diffusion Kinetics...")
    diffusion_results = physics_analyzer.calculate_diffusion_coefficient()
    if diffusion_results['apparent_diffusion_coeff']:
        avg_d = np.mean(diffusion_results['apparent_diffusion_coeff'])
        print(f"  - Apparent diffusion coefficient: {avg_d:.2e} m²/s")
    
    print("Calculating Charge Transfer Kinetics...")
    ct_results = physics_analyzer.calculate_charge_transfer_kinetics()
    if ct_results['exchange_current_density']:
        avg_i0 = np.mean(ct_results['exchange_current_density'])
        print(f"  - Exchange current density: {avg_i0:.2e} A/m²")
    
    print("Analyzing SiC Volume Expansion...")
    ve_results = physics_analyzer.analyze_volume_expansion()
    if ve_results['volume_expansion_ratio']:
        max_expansion = max(ve_results['volume_expansion_ratio'])
        print(f"  - Maximum volume expansion: {max_expansion*100:.1f}%")
    
    print("Studying SEI Formation...")
    sei_results = physics_analyzer.analyze_sei_formation()
    print(f"  - SEI composition analyzed with {len(sei_results['sei_composition'])} components")
    
    print("Calculating Impedance Spectrum...")
    impedance_results = physics_analyzer.calculate_impedance_spectrum()
    eqv = impedance_results['equivalent_circuit']
    print(f"  - Solution resistance: {eqv['Rs']*1000:.1f} mOhm")
    print(f"  - Charge transfer resistance: {eqv['Rct']*1000:.1f} mOhm")
    
    elapsed = time.time() - start_time
    print(f"\nPhysics analysis completed in {elapsed:.1f}s")
    
    return {
        'thermodynamics': thermo_results,
        'diffusion': diffusion_results,
        'charge_transfer': ct_results,
        'volume_expansion': ve_results,
        'sei_formation': sei_results,
        'impedance': impedance_results
    }

def create_advanced_visualizations(electrochemical_results: dict, ml_results: dict, 
                                 data: pd.DataFrame = None) -> None:
    """고급 시각화 생성"""
    print_section("ADVANCED VISUALIZATIONS")
    
    start_time = time.time()
    
    visualizer = AdvancedVisualizer("analysis_output/demo_plots")
    
    plot_count = 0
    
    if electrochemical_results['ica']:
        print("Creating Incremental Capacity Analysis plot...")
        visualizer.plot_incremental_capacity(electrochemical_results['ica'])
        plot_count += 1
    
    if electrochemical_results['dva']:
        print("Creating Differential Voltage Analysis plot...")
        visualizer.plot_differential_voltage(electrochemical_results['dva'])
        plot_count += 1
    
    if electrochemical_results['soh']:
        print("Creating Comprehensive SOH Analysis plot...")
        visualizer.plot_soh_comprehensive(electrochemical_results['soh'])
        plot_count += 1
    
    if electrochemical_results['resistance']:
        print("Creating Resistance Evolution plot...")
        visualizer.plot_resistance_evolution(electrochemical_results['resistance'])
        plot_count += 1
    
    if electrochemical_results['efficiency']:
        print("Creating Efficiency Analysis plot...")
        visualizer.plot_efficiency_analysis(electrochemical_results['efficiency'])
        plot_count += 1
    
    if ml_results['anomaly']:
        print("Creating Anomaly Detection plot...")
        visualizer.plot_anomaly_detection(ml_results['anomaly'])
        plot_count += 1
    
    if ml_results['fade']:
        print("Creating Capacity Fade Models plot...")
        visualizer.plot_capacity_fade_models(ml_results['fade'])
        plot_count += 1
    
    # Advanced 3D and heatmap plots
    if data is not None and len(data) > 1000:
        print("Creating 3D Surface plot...")
        visualizer.plot_3d_surface(data)
        plot_count += 1
        
        print("Creating Capacity Heatmap...")
        visualizer.plot_capacity_heatmap(data)
        plot_count += 1
    
    elapsed = time.time() - start_time
    print(f"\nCreated {plot_count} advanced visualizations in {elapsed:.1f}s")
    print(f"Plots saved to: analysis_output/demo_plots/")

def generate_comprehensive_report(electrochemical_results: dict, ml_results: dict, 
                                physics_results: dict) -> str:
    """종합 분석 보고서 생성"""
    print_section("COMPREHENSIVE REPORT GENERATION")
    
    report = []
    
    # Header
    report.append("=" * 100)
    report.append("COMPREHENSIVE BATTERY ANALYSIS REPORT")
    report.append("SiC+Graphite Anode / LiCoO2 Cathode System")
    report.append("=" * 100)
    report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Battery Technology: SiC+Graphite / LCO")
    report.append(f"Nominal Capacity: 4352 mAh")
    report.append("")
    
    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 50)
    
    # SOH status
    if electrochemical_results['soh']['combined_soh']:
        current_soh = electrochemical_results['soh']['combined_soh'][-1]
        report.append(f"Overall Battery Health: {current_soh:.1f}%")
        
        if current_soh > 90:
            status = "EXCELLENT - Battery performing within specifications"
        elif current_soh > 80:
            status = "GOOD - Minor degradation observed"
        elif current_soh > 70:
            status = "MODERATE - Noticeable degradation, monitor closely"
        else:
            status = "POOR - Significant degradation, consider replacement"
        
        report.append(f"Battery Status: {status}")
    
    # Key findings
    report.append("")
    report.append("Key Findings:")
    
    # Electrochemical findings
    if electrochemical_results['ica']['peaks']:
        avg_peaks = np.mean([p['num_peaks'] for p in electrochemical_results['ica']['peaks']])
        report.append(f"  - Average {avg_peaks:.1f} electrochemical peaks detected per cycle")
    
    if electrochemical_results['efficiency']['coulombic_efficiency']:
        avg_ce = np.mean(electrochemical_results['efficiency']['coulombic_efficiency'])
        report.append(f"  - Average Coulombic Efficiency: {avg_ce:.2f}%")
    
    # ML findings
    if ml_results['anomaly']['anomalous_cycles']:
        anomaly_rate = len(ml_results['anomaly']['anomalous_cycles']) / len(ml_results['anomaly']['cycle_numbers']) * 100
        report.append(f"  - {anomaly_rate:.1f}% of cycles identified as anomalous")
    
    # Physics findings
    if physics_results['volume_expansion']['volume_expansion_ratio']:
        max_expansion = max(physics_results['volume_expansion']['volume_expansion_ratio'])
        report.append(f"  - Maximum SiC volume expansion: {max_expansion*100:.1f}%")
    
    report.append("")
    
    # Detailed Analysis Sections
    report.append("DETAILED ANALYSIS RESULTS")
    report.append("-" * 50)
    
    # Electrochemical Section
    report.append("")
    report.append("1. ELECTROCHEMICAL ANALYSIS")
    report.append("   " + "=" * 30)
    
    if electrochemical_results['soh']:
        soh = electrochemical_results['soh']
        report.append(f"   State of Health:")
        if soh['capacity_based_soh']:
            report.append(f"     - Capacity-based SOH: {soh['capacity_based_soh'][-1]:.1f}%")
        if soh['resistance_based_soh']:
            report.append(f"     - Resistance-based SOH: {soh['resistance_based_soh'][-1]:.1f}%")
        if soh['voltage_based_soh']:
            report.append(f"     - Voltage-based SOH: {soh['voltage_based_soh'][-1]:.1f}%")
    
    # Internal Resistance
    if electrochemical_results['resistance']:
        resistance = electrochemical_results['resistance']
        valid_r = [r for r in resistance['average_resistance'] if not np.isnan(r)]
        if valid_r:
            report.append(f"   Internal Resistance:")
            report.append(f"     - Current resistance: {valid_r[-1]:.2f} mOhm")
            report.append(f"     - Resistance increase: {(valid_r[-1]/valid_r[0]-1)*100:.1f}%")
    
    # Machine Learning Section
    report.append("")
    report.append("2. MACHINE LEARNING ANALYSIS")
    report.append("   " + "=" * 30)
    
    if ml_results['clustering'] and 'cluster_analysis' in ml_results['clustering']:
        report.append(f"   Pattern Clustering:")
        for cluster_name, cluster_data in ml_results['clustering']['cluster_analysis'].items():
            report.append(f"     - {cluster_name}: {cluster_data['size']} cycles")
    
    if ml_results['anomaly']:
        report.append(f"   Anomaly Detection:")
        report.append(f"     - Anomalous cycles: {len(ml_results['anomaly']['anomalous_cycles'])}")
    
    # Physics Section  
    report.append("")
    report.append("3. PHYSICS & THERMODYNAMICS")
    report.append("   " + "=" * 30)
    
    if physics_results['thermodynamics']['entropy_change']:
        avg_entropy = np.mean(physics_results['thermodynamics']['entropy_change'])
        report.append(f"   Thermodynamic Properties:")
        report.append(f"     - Entropy change: {avg_entropy:.1f} J/mol·K")
    
    if physics_results['volume_expansion']['volume_expansion_ratio']:
        max_expansion = max(physics_results['volume_expansion']['volume_expansion_ratio'])
        report.append(f"   SiC Anode Characteristics:")
        report.append(f"     - Maximum volume expansion: {max_expansion*100:.1f}%")
    
    # Recommendations
    report.append("")
    report.append("RECOMMENDATIONS")
    report.append("-" * 50)
    
    # Based on SOH
    if electrochemical_results['soh']['combined_soh']:
        current_soh = electrochemical_results['soh']['combined_soh'][-1]
        
        if current_soh > 80:
            report.append("• Continue normal operation with regular monitoring")
            report.append("• Maintain optimal operating temperature range (20-25°C)")
        else:
            report.append("• Consider reducing C-rates to extend battery life")
            report.append("• Implement more frequent health monitoring")
            report.append("• Plan for battery replacement within next service cycle")
    
    # Based on anomalies
    if ml_results['anomaly']['anomalous_cycles']:
        report.append("• Investigate root causes of anomalous cycles")
        report.append("• Review operating conditions during anomalous periods")
    
    # SiC specific
    if physics_results['volume_expansion']['volume_expansion_ratio']:
        max_expansion = max(physics_results['volume_expansion']['volume_expansion_ratio'])
        if max_expansion > 0.1:  # >10%
            report.append("• Monitor for potential mechanical stress from SiC expansion")
            report.append("• Consider electrolyte additives to stabilize SEI layer")
    
    report.append("")
    report.append("=" * 100)
    report.append("END OF COMPREHENSIVE ANALYSIS REPORT")
    report.append("=" * 100)
    
    return "\n".join(report)

def main():
    """메인 데모 실행 함수"""
    print_banner("COMPREHENSIVE BATTERY ANALYSIS DEMO")
    print("SiC+Graphite / LCO Battery System Deep Analysis")
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for data
    data_path = "analysis_output/processed_data.csv"
    
    if not Path(data_path).exists():
        print(f"\nError: Data file not found at {data_path}")
        print("Please run battery_pattern_analyzer.py first to generate processed data.")
        return
    
    print(f"\nData source: {data_path}")
    
    # Load data for some analyses
    data = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded {len(data):,} data points covering {data['TotalCycle'].max()} cycles")
    
    total_start_time = time.time()
    
    # 1. Electrochemical Analysis
    electrochemical_results = run_electrochemical_analysis(data_path)
    
    # 2. Machine Learning Analysis (using same analyzer)
    analyzer = DeepBatteryAnalyzer(data_path)
    ml_results = run_machine_learning_analysis(analyzer)
    
    # 3. Physics Analysis
    physics_results = run_physics_analysis(data_path)
    
    # 4. Advanced Visualizations
    create_advanced_visualizations(electrochemical_results, ml_results, data)
    
    # 5. Comprehensive Report
    comprehensive_report = generate_comprehensive_report(
        electrochemical_results, ml_results, physics_results
    )
    
    # Save comprehensive report
    report_path = Path("analysis_output/comprehensive_analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(comprehensive_report)
    
    # Display final summary
    total_elapsed = time.time() - total_start_time
    
    print_section("DEMO COMPLETION SUMMARY")
    
    print(f"Total analysis time: {total_elapsed:.1f} seconds")
    print(f"Comprehensive report: {report_path}")
    print(f"Advanced plots: analysis_output/demo_plots/")
    
    print("\nGenerated Files:")
    output_dir = Path("analysis_output")
    
    # List all generated files
    report_files = list(output_dir.glob("*report*.txt"))
    plot_dirs = [d for d in output_dir.iterdir() if d.is_dir() and 'plot' in d.name]
    
    for file in report_files:
        size_kb = file.stat().st_size / 1024
        print(f"  [REPORT] {file.name} ({size_kb:.1f} KB)")
    
    for plot_dir in plot_dirs:
        plot_files = list(plot_dir.glob("*.png"))
        print(f"  [PLOTS] {plot_dir.name}/ ({len(plot_files)} plots)")
    
    print("\n" + "="*80)
    print("[SUCCESS] COMPREHENSIVE BATTERY ANALYSIS DEMO COMPLETED!")
    print("="*80)
    
    # Display key metrics
    if electrochemical_results['soh']['combined_soh']:
        soh = electrochemical_results['soh']['combined_soh'][-1]
        print(f"\n[BATTERY] Final Battery Health: {soh:.1f}%")
    
    if ml_results['anomaly']['anomalous_cycles']:
        anomaly_count = len(ml_results['anomaly']['anomalous_cycles'])
        print(f"[ANOMALY] Anomalous Cycles Detected: {anomaly_count}")
    
    if physics_results['volume_expansion']['volume_expansion_ratio']:
        max_expansion = max(physics_results['volume_expansion']['volume_expansion_ratio'])
        print(f"[PHYSICS] Maximum SiC Volume Expansion: {max_expansion*100:.1f}%")
    
    print(f"[TIMING] Total Processing Time: {total_elapsed:.1f}s")
    print("\nThank you for using the Comprehensive Battery Analysis System!")


if __name__ == "__main__":
    main()
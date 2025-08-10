#!/usr/bin/env python3
"""
Journal-Quality Battery Test Data Visualization
Creates publication-ready plots with Nature/Science journal styling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set journal-quality plot style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# High-resolution settings for publication
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'lines.markersize': 3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def load_and_prepare_data(file_path, sample_rate=100):
    """Load and prepare battery test data with proper time axis"""
    print(f"Loading data from {file_path}...")
    
    # Load data
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Loaded {len(df):,} data points")
    
    # Create proper time axis (assuming 1 second sampling interval)
    df['Time_s'] = np.arange(len(df))
    df['Time_h'] = df['Time_s'] / 3600  # Convert to hours
    
    # Sample data for better visualization (every Nth point)
    if len(df) > 50000:
        df_sampled = df.iloc[::sample_rate].copy()
        print(f"Sampled to {len(df_sampled):,} points for visualization")
    else:
        df_sampled = df.copy()
    
    return df, df_sampled

def create_voltage_current_profile(df, df_sampled, output_dir):
    """Create journal-quality voltage and current profile plot"""
    
    # Create figure with golden ratio
    fig = plt.figure(figsize=(10, 6.18))  # Golden ratio
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.05)
    
    # Voltage subplot
    ax1 = fig.add_subplot(gs[0])
    
    # Separate charge and discharge data
    charge_mask = df_sampled['Current[A]'] > 0
    discharge_mask = df_sampled['Current[A]'] < 0
    
    # Plot voltage with different colors for charge/discharge
    ax1.plot(df_sampled.loc[charge_mask, 'Time_h'], 
             df_sampled.loc[charge_mask, 'Voltage[V]'],
             color='#E74C3C', linewidth=0.8, alpha=0.8, label='Charge')
    ax1.plot(df_sampled.loc[discharge_mask, 'Time_h'], 
             df_sampled.loc[discharge_mask, 'Voltage[V]'],
             color='#3498DB', linewidth=0.8, alpha=0.8, label='Discharge')
    
    ax1.set_ylabel('Voltage (V)', fontweight='bold')
    ax1.set_ylim([2.5, 4.6])
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=False)
    ax1.set_xticklabels([])  # Hide x labels for top plot
    
    # Current subplot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Plot current with fill
    ax2.fill_between(df_sampled.loc[charge_mask, 'Time_h'], 
                     0, df_sampled.loc[charge_mask, 'Current[A]'],
                     color='#E74C3C', alpha=0.6, label='Charge')
    ax2.fill_between(df_sampled.loc[discharge_mask, 'Time_h'], 
                     0, df_sampled.loc[discharge_mask, 'Current[A]'],
                     color='#3498DB', alpha=0.6, label='Discharge')
    
    ax2.plot(df_sampled['Time_h'], df_sampled['Current[A]'],
             color='#2C3E50', linewidth=0.5, alpha=0.8)
    
    ax2.set_xlabel('Time (hours)', fontweight='bold')
    ax2.set_ylabel('Current (A)', fontweight='bold')
    ax2.set_ylim([-5, 10])
    
    # Add zero line
    ax2.axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
    
    # Set x-axis limit
    max_time = df_sampled['Time_h'].max()
    ax2.set_xlim([0, min(max_time, 100)])  # Show first 100 hours
    
    # Add title
    fig.suptitle('Battery Charge-Discharge Profile', fontsize=14, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path = output_dir / '02_voltage_current_profile.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")
    plt.close()

def create_cycle_performance_plot(df, output_dir):
    """Create cycle performance plot showing capacity fade and efficiency"""
    
    # Calculate cycle statistics
    cycle_stats = []
    for cycle in df['TotalCycle'].unique()[:100]:  # First 100 cycles
        cycle_data = df[df['TotalCycle'] == cycle]
        if len(cycle_data) > 0:
            max_charge = cycle_data['Chg_Capacity[Ah]'].max()
            max_discharge = abs(cycle_data['Dchg_Capacity[Ah]'].min())
            efficiency = (max_discharge / max_charge * 100) if max_charge > 0 else 0
            
            cycle_stats.append({
                'Cycle': cycle,
                'Charge_Capacity': max_charge,
                'Discharge_Capacity': max_discharge,
                'Efficiency': efficiency
            })
    
    if not cycle_stats:
        print("No cycle statistics to plot")
        return
    
    stats_df = pd.DataFrame(cycle_stats)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    # Capacity plot
    ax1.plot(stats_df['Cycle'], stats_df['Charge_Capacity'], 
             marker='o', markersize=4, color='#E74C3C', 
             linewidth=1.5, label='Charge Capacity', alpha=0.8)
    ax1.plot(stats_df['Cycle'], stats_df['Discharge_Capacity'], 
             marker='s', markersize=4, color='#3498DB', 
             linewidth=1.5, label='Discharge Capacity', alpha=0.8)
    
    ax1.set_ylabel('Capacity (Ah)', fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, fancybox=True)
    ax1.grid(True, alpha=0.3)
    
    # Efficiency plot  
    ax2.plot(stats_df['Cycle'], stats_df['Efficiency'], 
             marker='^', markersize=4, color='#27AE60', 
             linewidth=1.5, alpha=0.8)
    ax2.fill_between(stats_df['Cycle'], stats_df['Efficiency'], 
                     alpha=0.3, color='#27AE60')
    
    ax2.set_xlabel('Cycle Number', fontweight='bold')
    ax2.set_ylabel('Coulombic Efficiency (%)', fontweight='bold')
    ax2.set_ylim([95, 101])
    ax2.grid(True, alpha=0.3)
    
    # Add title
    fig.suptitle('Cycle Performance Analysis', fontsize=14, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path = output_dir / '04_cycle_performance.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")
    plt.close()

def create_multi_cycle_overlay(df, output_dir, cycles_to_show=[1, 100, 200, 500, 1000]):
    """Create overlay plot of multiple cycles for comparison"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Color palette for different cycles
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(cycles_to_show)))
    
    for i, cycle in enumerate(cycles_to_show):
        cycle_data = df[df['TotalCycle'] == cycle].head(10000)  # Limit points
        
        if len(cycle_data) > 0:
            # Create time axis for this cycle
            cycle_data['Time_h'] = np.arange(len(cycle_data)) / 3600
            
            # Voltage vs Time
            ax1.plot(cycle_data['Time_h'], cycle_data['Voltage[V]'],
                    linewidth=1.2, alpha=0.7, color=colors[i],
                    label=f'Cycle {cycle}')
            
            # Voltage vs Capacity
            ax2.plot(cycle_data['Chg_Capacity[Ah]'], cycle_data['Voltage[V]'],
                    linewidth=1.2, alpha=0.7, color=colors[i],
                    label=f'Cycle {cycle}')
    
    # Format voltage vs time plot
    ax1.set_xlabel('Time (hours)', fontweight='bold')
    ax1.set_ylabel('Voltage (V)', fontweight='bold')
    ax1.set_title('Voltage Profile Evolution', fontweight='bold')
    ax1.legend(loc='best', frameon=True, fancybox=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 5])  # Show first 5 hours
    
    # Format voltage vs capacity plot
    ax2.set_xlabel('Capacity (Ah)', fontweight='bold')
    ax2.set_ylabel('Voltage (V)', fontweight='bold')
    ax2.set_title('Voltage-Capacity Curves', fontweight='bold')
    ax2.legend(loc='best', frameon=True, fancybox=True)
    ax2.grid(True, alpha=0.3)
    
    # Add main title
    fig.suptitle('Multi-Cycle Comparison', fontsize=14, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path = output_dir / '05_multi_cycle_comparison.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")
    plt.close()

def create_rate_capability_plot(df, output_dir):
    """Create rate capability plot showing C-rate vs capacity"""
    
    # Calculate C-rates and capacities for different cycles
    rate_data = []
    
    for cycle in df['TotalCycle'].unique()[:200]:  # First 200 cycles
        cycle_data = df[df['TotalCycle'] == cycle]
        if len(cycle_data) > 0:
            # Calculate average current (C-rate)
            avg_current = cycle_data['Current[A]'].abs().mean()
            c_rate = avg_current / 4.352  # Assuming 4.352Ah nominal capacity
            
            # Get discharge capacity
            discharge_cap = abs(cycle_data['Dchg_Capacity[Ah]'].min())
            
            if discharge_cap > 0:
                rate_data.append({
                    'Cycle': cycle,
                    'C_rate': c_rate,
                    'Capacity': discharge_cap,
                    'Retention': (discharge_cap / 4.352) * 100
                })
    
    if rate_data:
        rate_df = pd.DataFrame(rate_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create scatter plot with color gradient
        scatter = ax.scatter(rate_df['C_rate'], rate_df['Retention'], 
                           c=rate_df['Cycle'], cmap='coolwarm',
                           s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cycle Number', fontweight='bold')
        
        # Format plot
        ax.set_xlabel('C-rate', fontweight='bold')
        ax.set_ylabel('Capacity Retention (%)', fontweight='bold')
        ax.set_title('Rate Capability Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(rate_df['C_rate'], rate_df['Retention'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(rate_df['C_rate'].min(), rate_df['C_rate'].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=1, label='Trend')
        
        ax.legend(loc='best', frameon=True, fancybox=True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / '06_rate_capability.png'
        plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")
        plt.close()

def main():
    """Main execution function"""
    print("=" * 60)
    print("Creating Journal-Quality Battery Test Visualizations")
    print("=" * 60)
    
    # Set paths
    data_file = Path("analysis_output/processed_data.csv")
    output_dir = Path("analysis_output/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data file exists
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        print("Please run battery_pattern_analyzer.py first to generate processed data.")
        return
    
    # Load and prepare data
    df, df_sampled = load_and_prepare_data(data_file, sample_rate=100)
    
    # Create visualizations
    print("\nGenerating plots...")
    
    # 1. Voltage-Current Profile
    print("\n1. Creating voltage-current profile...")
    create_voltage_current_profile(df, df_sampled, output_dir)
    
    # 2. Cycle Performance
    print("\n2. Creating cycle performance plot...")
    create_cycle_performance_plot(df, output_dir)
    
    # 3. Multi-Cycle Overlay
    print("\n3. Creating multi-cycle comparison...")
    create_multi_cycle_overlay(df, output_dir)
    
    # 4. Rate Capability
    print("\n4. Creating rate capability plot...")
    create_rate_capability_plot(df, output_dir)
    
    print("\n" + "=" * 60)
    print("[OK] All visualizations created successfully!")
    print(f"[Output] directory: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
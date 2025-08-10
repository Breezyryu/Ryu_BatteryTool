# 🔋 Battery Analyzer Tool

Comprehensive battery data analysis system with manufacturer recognition and automated file naming.

## 🎯 Features

### Core Capabilities
- **Single/Multiple Path Analysis**: Analyze one or multiple battery datasets
- **Manufacturer Auto-Detection**: Automatically recognizes SDI, ATL, LGES, COSMX, BYD, Panasonic, SK, EVE, Northvolt, SVOLT
- **Smart File Naming**: Output files include manufacturer, capacity, model, test conditions, and date/time
- **Comprehensive Visualization**: Advanced Seaborn plots, domain-specific analysis, statistical modeling

### File Structure
```
Ryu_BatteryTool/
├── battery_analyzer_main.py          # ⭐ Main execution file
├── battery_pattern_analyzer.py       # Data loading and pattern analysis  
├── integrated_reporter.py            # Comprehensive reporting system
├── enhanced_seaborn_visualizer.py    # Advanced Seaborn visualizations
├── battery_domain_visualizer.py      # Electrochemical analysis plots
├── statistical_battery_visualizer.py # Statistical analysis and modeling
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Usage Examples

#### Single Path Analysis
```bash
# Interactive mode
python battery_analyzer_main.py --interactive

# Direct path
python battery_analyzer_main.py --path "data/LGES_G3_4352mAh_상온수명"
```

#### Multiple Path Analysis
```bash
# Compare multiple manufacturers
python battery_analyzer_main.py --paths \
    "data/LGES_4352mAh" \
    "data/Samsung_SDI_4000mAh" \
    "data/ATL_4200mAh"
```

## 📊 Output Structure

### Smart File Naming
Files are automatically named based on extracted battery information:

**Format**: `{Manufacturer}_{Model}_{Capacity}_{TestCondition}_{AnalysisType}_{Date}_{Time}`

**Examples**:
- `LGES_G3_4352mAh_RT_Life_analysis_20250110_143022/`
- `SDI_NCM811_4000mAh_report_20250110_143100.html`
- `ATL_4200mAh_HT_facetgrid_20250110.png`

### Output Directory Structure
```
analysis_output/
├── LGES_G3_4352mAh_RT_Life_20250110_143022/
│   ├── data/
│   │   └── LGES_G3_4352mAh_processed_data_20250110_143022.csv
│   ├── reports/
│   │   ├── LGES_G3_4352mAh_report_20250110_143022.html
│   │   └── LGES_G3_4352mAh_summary_20250110_143022.json
│   └── visualizations/
│       ├── seaborn_analysis/
│       ├── domain_specific/
│       └── statistical_analysis/
└── Comparison_LGES_vs_SDI_20250110_143500/ (for multi-path analysis)
```

## 🔬 Analysis Modules

### 1. Enhanced Seaborn Visualizer
- **FacetGrid**: Multi-dimensional analysis by cycle range and phase
- **PairGrid**: Advanced correlation matrices with KDE plots
- **Clustermap**: Hierarchical clustering of battery states
- **JointGrid**: Detailed relationship analysis with marginal distributions
- **Advanced Violin Plots**: Distribution analysis with statistical overlays

### 2. Battery Domain Visualizer
- **Ragone Plots**: Energy vs power density analysis
- **Electrochemical Impedance**: Nyquist-style impedance evolution
- **Life Prediction**: Battery degradation and survival curves
- **Thermal Analysis**: Temperature effect simulations
- **Cycling Efficiency**: Charge/discharge efficiency analysis

### 3. Statistical Battery Visualizer
- **Bayesian Analysis**: Uncertainty quantification with credible intervals
- **Survival Analysis**: Kaplan-Meier curves for battery life prediction
- **Monte Carlo Simulation**: Probabilistic performance modeling
- **Time Series Decomposition**: Trend, seasonal, and residual analysis

## 🏭 Manufacturer Recognition

### Supported Manufacturers
| Standard Name | Variations Detected |
|---------------|-------------------|
| **SDI** | SDI, Samsung, SAMSUNG, samsung_sdi, SamsungSDI, 삼성, 삼성SDI |
| **ATL** | ATL, Amperex, CATL, amperex, AMPEREX |
| **LGES** | LGES, LG, LGChem, LG_Energy, LGEnergy, LG화학, LG에너지솔루션 |
| **COSMX** | COSMX, Cosmo, cosmo, 코스모, cosmos, 코스모신소재 |
| **BYD** | BYD, byd, Build_Your_Dreams, Blade |
| **Panasonic** | Panasonic, PANA, Tesla_Panasonic, 파나소닉 |
| **SK** | SK, SKI, SK_Innovation, SK이노베이션, SKInnovation |
| **EVE** | EVE, eve, EVE_Energy, EVEEnergy |
| **Northvolt** | Northvolt, NORTH, northvolt |
| **SVOLT** | SVOLT, svolt, 스볼트, Svolt |

### Path Examples
```bash
# These paths will be automatically classified:
"LGES_G3_MP1_4352mAh_상온수명"          → LGES
"Samsung_SDI_NCM811_4000mAh"            → SDI  
"ATL_high_capacity_4200mAh_고온"        → ATL
"BYD_Blade_Battery_4500mAh"             → BYD
"lg_energy_solution_test_3800mAh"       → LGES
```

## 📈 Analysis Workflow

1. **Path Analysis**: Extract manufacturer, capacity, model, and test conditions from file paths
2. **Data Loading**: Automatically detect PNE or Toyo format and load data
3. **Pattern Analysis**: Identify charge/discharge cycles and battery behavior patterns
4. **Visualization**: Generate comprehensive plots using all three visualization modules
5. **Reporting**: Create HTML reports and JSON summaries with smart file naming
6. **Comparison**: For multiple paths, generate manufacturer comparison reports

## 🛠️ Technical Requirements

### Dependencies
```txt
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
seaborn>=0.12.0
plotly>=5.0.0
scipy>=1.9.0
statsmodels>=0.13.0
kaleido>=0.2.1
```

### Data Format Support
- **PNE Format**: M01Ch### structure with SaveData CSV files
- **Toyo Format**: Numbered files (000001, 000002, etc.) with CAPACITY.LOG

## 📝 Generated Reports

### HTML Report Features
- Executive summary with battery specifications
- Analysis module status and results
- Interactive visualization gallery
- Error reporting and warnings
- Professional styling with responsive design

### JSON Summary Features
- Machine-readable analysis results
- Metadata including analysis duration and file counts
- Battery specifications and detected information
- Error logs and warnings
- Output file locations

## 🔧 Troubleshooting

### Common Issues
1. **Missing Dependencies**: Run `pip install -r requirements.txt`
2. **Unknown Manufacturer**: Add manufacturer variations to the mapping dictionary
3. **Data Format Issues**: Ensure data follows PNE or Toyo format conventions
4. **Memory Issues**: Use smaller data subsets for initial testing

### Getting Help
- Check the generated HTML reports for detailed error information
- Review the JSON summary for technical details
- Ensure input paths contain recognizable manufacturer and capacity information

---

**Last Updated**: 2025-01-10  
**Version**: 2.0.0  
**Status**: Production Ready ✅
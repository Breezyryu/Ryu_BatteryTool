#!/usr/bin/env python3
"""
Integrated Battery Analysis Reporter
원클릭 종합 배터리 분석 및 리포트 생성 시스템
모든 시각화 모듈을 통합하여 자동화된 분석 리포트 제공
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import traceback

# Import all visualization modules
try:
    from enhanced_seaborn_visualizer import EnhancedSeabornVisualizer
    from battery_domain_visualizer import BatteryDomainVisualizer
    from statistical_battery_visualizer import StatisticalBatteryVisualizer
    from multiscale_analyzer import MultiScaleAnalyzer
    from comparative_visualizer import ComparativeBatteryVisualizer
except ImportError as e:
    print(f"Warning: Could not import visualization modules: {e}")
    print("Please ensure all visualization modules are in the same directory")

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('battery_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

class IntegratedBatteryReporter:
    """통합 배터리 분석 리포터 클래스"""
    
    def __init__(self, output_dir: str = "analysis_output/integrated_report"):
        """
        초기화
        
        Args:
            output_dir: 출력 디렉토리 경로
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organized output
        self.subdirs = {
            'seaborn': self.output_dir / 'seaborn_analysis',
            'domain': self.output_dir / 'domain_specific',
            'statistical': self.output_dir / 'statistical_analysis',
            'multiscale': self.output_dir / 'multiscale_analysis',
            'comparative': self.output_dir / 'comparative_analysis',
            'reports': self.output_dir / 'reports'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizers
        self.visualizers = {}
        self.analysis_results = {}
        self.analysis_metadata = {
            'start_time': None,
            'end_time': None,
            'duration': None,
            'data_points': 0,
            'cycles_analyzed': 0,
            'analysis_modules': [],
            'errors': []
        }
        
        logger.info(f"Integrated Battery Reporter initialized, output: {self.output_dir}")
    
    def initialize_visualizers(self) -> bool:
        """
        모든 시각화 모듈 초기화
        
        Returns:
            초기화 성공 여부
        """
        logger.info("Initializing all visualization modules...")
        
        try:
            # Enhanced Seaborn Visualizer
            self.visualizers['seaborn'] = EnhancedSeabornVisualizer(
                output_dir=str(self.subdirs['seaborn'])
            )
            self.analysis_metadata['analysis_modules'].append('Enhanced Seaborn')
            
            # Battery Domain Visualizer
            self.visualizers['domain'] = BatteryDomainVisualizer(
                output_dir=str(self.subdirs['domain'])
            )
            self.analysis_metadata['analysis_modules'].append('Battery Domain')
            
            # Statistical Visualizer
            self.visualizers['statistical'] = StatisticalBatteryVisualizer(
                output_dir=str(self.subdirs['statistical'])
            )
            self.analysis_metadata['analysis_modules'].append('Statistical Analysis')
            
            # Multiscale Analyzer
            self.visualizers['multiscale'] = MultiScaleAnalyzer(
                output_dir=str(self.subdirs['multiscale'])
            )
            self.analysis_metadata['analysis_modules'].append('Multiscale Analysis')
            
            # Comparative Visualizer
            self.visualizers['comparative'] = ComparativeBatteryVisualizer(
                output_dir=str(self.subdirs['comparative'])
            )
            self.analysis_metadata['analysis_modules'].append('Comparative Analysis')
            
            logger.info(f"Successfully initialized {len(self.visualizers)} visualization modules")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize visualizers: {str(e)}"
            logger.error(error_msg)
            self.analysis_metadata['errors'].append(error_msg)
            return False
    
    def analyze_data_overview(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        데이터 개요 분석
        
        Args:
            data: 배터리 데이터
            
        Returns:
            분석 결과 딕셔너리
        """
        logger.info("Analyzing data overview...")
        
        try:
            overview = {
                'total_points': len(data),
                'unique_cycles': data['TotalCycle'].nunique(),
                'cycle_range': (data['TotalCycle'].min(), data['TotalCycle'].max()),
                'voltage_range': (data['Voltage[V]'].min(), data['Voltage[V]'].max()),
                'capacity_range': (data['Chg_Capacity[Ah]'].min(), data['Chg_Capacity[Ah]'].max()),
                'current_range': (data['Current[A]'].min(), data['Current[A]'].max()),
                'missing_values': data.isnull().sum().to_dict(),
                'data_types': data.dtypes.to_dict()
            }
            
            # Update metadata
            self.analysis_metadata['data_points'] = overview['total_points']
            self.analysis_metadata['cycles_analyzed'] = overview['unique_cycles']
            
            return overview
            
        except Exception as e:
            error_msg = f"Data overview analysis failed: {str(e)}"
            logger.error(error_msg)
            self.analysis_metadata['errors'].append(error_msg)
            return {}
    
    def run_parallel_analysis(self, data: pd.DataFrame, max_workers: int = 3) -> Dict[str, Any]:
        """
        병렬로 모든 분석 실행
        
        Args:
            data: 배터리 데이터
            max_workers: 최대 워커 수
            
        Returns:
            분석 결과 딕셔너리
        """
        logger.info(f"Starting parallel analysis with {max_workers} workers...")
        
        results = {}
        
        # Define analysis tasks
        analysis_tasks = [
            ('seaborn', self._run_seaborn_analysis),
            ('domain', self._run_domain_analysis),
            ('statistical', self._run_statistical_analysis),
            ('multiscale', self._run_multiscale_analysis),
            ('comparative', self._run_comparative_analysis)
        ]
        
        # Execute tasks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_analysis = {
                executor.submit(task_func, data): analysis_name 
                for analysis_name, task_func in analysis_tasks
            }
            
            # Collect results
            for future in as_completed(future_to_analysis):
                analysis_name = future_to_analysis[future]
                try:
                    result = future.result()
                    results[analysis_name] = result
                    logger.info(f"Completed {analysis_name} analysis")
                except Exception as e:
                    error_msg = f"{analysis_name} analysis failed: {str(e)}"
                    logger.error(error_msg)
                    self.analysis_metadata['errors'].append(error_msg)
                    results[analysis_name] = {'error': str(e)}
        
        return results
    
    def _run_seaborn_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced Seaborn 분석 실행"""
        try:
            visualizer = self.visualizers['seaborn']
            
            # Run all Seaborn visualizations
            visualizer.create_facet_grid_analysis(data)
            visualizer.create_pairgrid_correlation_matrix(data)
            visualizer.create_clustermap_analysis(data)
            visualizer.create_jointgrid_detailed_analysis(data)
            visualizer.create_advanced_violin_plots(data)
            visualizer.create_regression_analysis_grid(data)
            
            return {
                'status': 'completed',
                'visualizations': [
                    'facetgrid_voltage_capacity_analysis.png',
                    'pairgrid_correlation_matrix.png',
                    'clustermap_battery_states.png',
                    'jointgrid_voltage_capacity_analysis.png',
                    'advanced_violin_plots.png',
                    'regression_analysis_grid.png'
                ]
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _run_domain_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Battery Domain 분석 실행"""
        try:
            visualizer = self.visualizers['domain']
            
            # Run domain-specific visualizations
            visualizer.create_ragone_plot(data)
            visualizer.create_electrochemical_impedance_plot(data)
            visualizer.create_battery_life_prediction(data)
            visualizer.create_thermal_analysis(data)
            visualizer.create_cycling_efficiency_analysis(data)
            
            return {
                'status': 'completed',
                'visualizations': [
                    'ragone_plot_energy_power.png',
                    'electrochemical_impedance_evolution.png',
                    'battery_life_prediction.png',
                    'thermal_analysis_dashboard.png',
                    'cycling_efficiency_analysis.png'
                ]
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _run_statistical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Statistical 분석 실행"""
        try:
            visualizer = self.visualizers['statistical']
            
            # Run statistical analyses
            visualizer.create_bayesian_analysis(data)
            visualizer.create_survival_analysis(data)
            visualizer.create_monte_carlo_analysis(data)
            visualizer.create_advanced_time_series_decomposition(data)
            
            return {
                'status': 'completed',
                'visualizations': [
                    'bayesian_capacity_analysis.png',
                    'battery_survival_analysis.png',
                    'monte_carlo_uncertainty_analysis.png',
                    'advanced_time_series_decomposition.png'
                ]
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _run_multiscale_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Multiscale 분석 실행"""
        try:
            visualizer = self.visualizers['multiscale']
            
            # Run multiscale analyses
            visualizer.create_macro_scale_analysis(data)
            visualizer.create_meso_scale_analysis(data)
            visualizer.create_micro_scale_analysis(data)
            visualizer.create_cross_scale_correlation(data)
            
            return {
                'status': 'completed',
                'visualizations': [
                    'macro_scale_lifecycle_analysis.png',
                    'meso_scale_cycle_groups.png',
                    'micro_scale_individual_cycles.png',
                    'cross_scale_correlation_analysis.png'
                ]
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _run_comparative_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comparative 분석 실행"""
        try:
            visualizer = self.visualizers['comparative']
            
            # Create synthetic battery data for comparison
            batteries_data = visualizer.create_synthetic_battery_data()
            
            # Run comparative analyses
            visualizer.create_performance_radar_chart(batteries_data)
            visualizer.create_benchmark_comparison(batteries_data)
            visualizer.create_degradation_comparison(batteries_data)
            
            return {
                'status': 'completed',
                'visualizations': [
                    'performance_radar_comparison.png',
                    'industry_benchmark_comparison.png',
                    'degradation_pattern_comparison.png'
                ]
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def generate_html_report(self, data_overview: Dict, analysis_results: Dict) -> str:
        """
        HTML 형식의 종합 리포트 생성
        
        Args:
            data_overview: 데이터 개요
            analysis_results: 분석 결과
            
        Returns:
            HTML 리포트 파일 경로
        """
        logger.info("Generating comprehensive HTML report...")
        
        # HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Battery Analysis Comprehensive Report</title>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    padding: 20px 0;
                    border-bottom: 2px solid #3498db;
                    margin-bottom: 30px;
                }}
                .header h1 {{
                    color: #2c3e50;
                    margin: 0;
                    font-size: 2.5em;
                }}
                .header p {{
                    color: #7f8c8d;
                    margin: 10px 0;
                    font-size: 1.1em;
                }}
                .section {{
                    margin: 30px 0;
                    padding: 20px;
                    background-color: #fafafa;
                    border-radius: 8px;
                    border-left: 4px solid #3498db;
                }}
                .section h2 {{
                    color: #2c3e50;
                    margin-top: 0;
                }}
                .overview-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .overview-card {{
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .overview-card h3 {{
                    margin: 0 0 10px 0;
                    color: #3498db;
                }}
                .analysis-section {{
                    margin: 20px 0;
                }}
                .analysis-status {{
                    display: inline-block;
                    padding: 5px 15px;
                    border-radius: 20px;
                    color: white;
                    font-weight: bold;
                    margin: 10px 10px 10px 0;
                }}
                .status-completed {{ background-color: #27ae60; }}
                .status-error {{ background-color: #e74c3c; }}
                .visualizations {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 15px;
                    margin: 15px 0;
                }}
                .viz-card {{
                    background: white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                .footer {{
                    text-align: center;
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #ecf0f1;
                    color: #7f8c8d;
                }}
                .error-section {{
                    background-color: #fdf2f2;
                    border-left-color: #e74c3c;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔋 Battery Analysis Report</h1>
                    <p>Comprehensive Multi-Scale Battery Performance Analysis</p>
                    <p><strong>Generated:</strong> {timestamp}</p>
                    <p><strong>Analysis Duration:</strong> {duration}</p>
                </div>

                <div class="section">
                    <h2>📊 Data Overview</h2>
                    <div class="overview-grid">
                        <div class="overview-card">
                            <h3>Data Points</h3>
                            <p><strong>{total_points:,}</strong> measurements</p>
                        </div>
                        <div class="overview-card">
                            <h3>Cycles Analyzed</h3>
                            <p><strong>{unique_cycles:,}</strong> cycles</p>
                        </div>
                        <div class="overview-card">
                            <h3>Cycle Range</h3>
                            <p>{cycle_min} - {cycle_max}</p>
                        </div>
                        <div class="overview-card">
                            <h3>Voltage Range</h3>
                            <p>{voltage_min:.2f}V - {voltage_max:.2f}V</p>
                        </div>
                        <div class="overview-card">
                            <h3>Capacity Range</h3>
                            <p>{capacity_min:.2f} - {capacity_max:.2f} Ah</p>
                        </div>
                        <div class="overview-card">
                            <h3>Analysis Modules</h3>
                            <p><strong>{modules_count}</strong> modules executed</p>
                        </div>
                    </div>
                </div>

                {analysis_sections}

                {error_section}

                <div class="footer">
                    <p>Report generated by Integrated Battery Analysis System</p>
                    <p>Enhanced with Seaborn, Domain-Specific, Statistical, Multi-Scale, and Comparative Analysis</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Build analysis sections
        analysis_sections = []
        for analysis_name, result in analysis_results.items():
            if isinstance(result, dict):
                status = result.get('status', 'unknown')
                status_class = 'status-completed' if status == 'completed' else 'status-error'
                
                section_html = f"""
                <div class="section analysis-section">
                    <h2>📈 {analysis_name.title()} Analysis</h2>
                    <div class="analysis-status {status_class}">{status.upper()}</div>
                    """
                
                if 'visualizations' in result:
                    section_html += """
                    <div class="visualizations">
                    """
                    for viz in result['visualizations']:
                        section_html += f"""
                        <div class="viz-card">
                            <p>✅ {viz}</p>
                        </div>
                        """
                    section_html += "</div>"
                
                if 'error' in result:
                    section_html += f"""
                    <div style="color: #e74c3c; margin-top: 10px;">
                        <strong>Error:</strong> {result['error']}
                    </div>
                    """
                
                section_html += "</div>"
                analysis_sections.append(section_html)
        
        # Build error section
        error_section = ""
        if self.analysis_metadata.get('errors'):
            error_section = """
            <div class="section error-section">
                <h2>⚠️ Errors and Warnings</h2>
                <ul>
            """
            for error in self.analysis_metadata['errors']:
                error_section += f"<li>{error}</li>"
            error_section += "</ul></div>"
        
        # Fill template
        html_content = html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            duration=self.analysis_metadata.get('duration', 'Unknown'),
            total_points=data_overview.get('total_points', 0),
            unique_cycles=data_overview.get('unique_cycles', 0),
            cycle_min=data_overview.get('cycle_range', (0, 0))[0],
            cycle_max=data_overview.get('cycle_range', (0, 0))[1],
            voltage_min=data_overview.get('voltage_range', (0, 0))[0],
            voltage_max=data_overview.get('voltage_range', (0, 0))[1],
            capacity_min=data_overview.get('capacity_range', (0, 0))[0],
            capacity_max=data_overview.get('capacity_range', (0, 0))[1],
            modules_count=len(self.analysis_metadata.get('analysis_modules', [])),
            analysis_sections=''.join(analysis_sections),
            error_section=error_section
        )
        
        # Save HTML report
        report_path = self.subdirs['reports'] / 'comprehensive_analysis_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved: {report_path}")
        return str(report_path)
    
    def save_analysis_summary(self, data_overview: Dict, analysis_results: Dict) -> str:
        """
        분석 요약 JSON 파일 저장
        
        Args:
            data_overview: 데이터 개요
            analysis_results: 분석 결과
            
        Returns:
            JSON 파일 경로
        """
        summary = {
            'metadata': self.analysis_metadata,
            'data_overview': data_overview,
            'analysis_results': analysis_results,
            'output_directories': {k: str(v) for k, v in self.subdirs.items()}
        }
        
        summary_path = self.subdirs['reports'] / 'analysis_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Analysis summary saved: {summary_path}")
        return str(summary_path)
    
    def run_comprehensive_analysis(self, data_path: str, parallel: bool = True, max_workers: int = 3) -> Dict[str, str]:
        """
        종합 배터리 분석 실행
        
        Args:
            data_path: 데이터 파일 경로
            parallel: 병렬 처리 사용 여부
            max_workers: 최대 워커 수
            
        Returns:
            분석 결과 파일 경로들
        """
        logger.info("Starting comprehensive battery analysis...")
        self.analysis_metadata['start_time'] = datetime.now()
        
        try:
            # Load data
            logger.info(f"Loading data from: {data_path}")
            data = pd.read_csv(data_path, low_memory=False)
            logger.info(f"Loaded {len(data):,} data points")
            
            # Initialize visualizers
            if not self.initialize_visualizers():
                raise ValueError("Failed to initialize visualization modules")
            
            # Analyze data overview
            data_overview = self.analyze_data_overview(data)
            
            # Run analysis (parallel or sequential)
            if parallel and len(self.visualizers) > 1:
                analysis_results = self.run_parallel_analysis(data, max_workers)
            else:
                # Sequential analysis
                logger.info("Running sequential analysis...")
                analysis_results = {}
                for name in self.visualizers.keys():
                    try:
                        if name == 'seaborn':
                            analysis_results[name] = self._run_seaborn_analysis(data)
                        elif name == 'domain':
                            analysis_results[name] = self._run_domain_analysis(data)
                        elif name == 'statistical':
                            analysis_results[name] = self._run_statistical_analysis(data)
                        elif name == 'multiscale':
                            analysis_results[name] = self._run_multiscale_analysis(data)
                        elif name == 'comparative':
                            analysis_results[name] = self._run_comparative_analysis(data)
                        
                        logger.info(f"Completed {name} analysis")
                    except Exception as e:
                        error_msg = f"{name} analysis failed: {str(e)}"
                        logger.error(error_msg)
                        self.analysis_metadata['errors'].append(error_msg)
                        analysis_results[name] = {'status': 'error', 'error': str(e)}
            
            # Record completion time
            self.analysis_metadata['end_time'] = datetime.now()
            self.analysis_metadata['duration'] = str(self.analysis_metadata['end_time'] - self.analysis_metadata['start_time'])
            
            # Generate reports
            html_report = self.generate_html_report(data_overview, analysis_results)
            json_summary = self.save_analysis_summary(data_overview, analysis_results)
            
            # Return results
            result_paths = {
                'html_report': html_report,
                'json_summary': json_summary,
                'output_directory': str(self.output_dir),
                'subdirectories': {k: str(v) for k, v in self.subdirs.items()}
            }
            
            logger.info(f"Comprehensive analysis completed in {self.analysis_metadata['duration']}")
            logger.info(f"Results available in: {self.output_dir}")
            
            return result_paths
            
        except Exception as e:
            error_msg = f"Comprehensive analysis failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.analysis_metadata['errors'].append(error_msg)
            self.analysis_metadata['end_time'] = datetime.now()
            self.analysis_metadata['duration'] = str(self.analysis_metadata['end_time'] - self.analysis_metadata['start_time'])
            
            return {
                'error': error_msg,
                'output_directory': str(self.output_dir)
            }


def main():
    """메인 실행 함수"""
    print("🔋 Integrated Battery Analysis Reporter")
    print("=" * 80)
    print("원클릭 종합 배터리 분석 및 리포트 생성 시스템")
    print("=" * 80)
    
    # Check if data file exists
    data_path = Path("analysis_output/processed_data.csv")
    
    if not data_path.exists():
        print(f"❌ Error: Data file not found at {data_path}")
        print("Please run battery_pattern_analyzer.py first to generate processed data.")
        return
    
    # Create integrated reporter
    print("🚀 Initializing Integrated Battery Reporter...")
    reporter = IntegratedBatteryReporter()
    
    # Run comprehensive analysis
    print("📊 Starting comprehensive analysis...")
    print("This may take several minutes as all visualization modules will be executed...")
    
    start_time = time.time()
    
    try:
        results = reporter.run_comprehensive_analysis(
            data_path=str(data_path),
            parallel=True,  # Enable parallel processing
            max_workers=3   # Adjust based on your system
        )
        
        if 'error' in results:
            print(f"❌ Analysis failed: {results['error']}")
        else:
            elapsed_time = time.time() - start_time
            print(f"\n✅ Comprehensive analysis completed in {elapsed_time:.1f} seconds!")
            print("\n📋 Generated Reports:")
            print(f"  • HTML Report: {results['html_report']}")
            print(f"  • JSON Summary: {results['json_summary']}")
            print(f"  • Output Directory: {results['output_directory']}")
            
            print("\n📁 Analysis Subdirectories:")
            for name, path in results['subdirectories'].items():
                print(f"  • {name.title()}: {path}")
            
            print("\n🎯 Analysis Modules Executed:")
            modules = reporter.analysis_metadata.get('analysis_modules', [])
            for module in modules:
                print(f"  ✓ {module}")
            
            if reporter.analysis_metadata.get('errors'):
                print(f"\n⚠️  {len(reporter.analysis_metadata['errors'])} warnings/errors occurred")
                print("Check the HTML report for details.")
            
            print(f"\n🔍 Data Summary:")
            print(f"  • Total data points: {reporter.analysis_metadata.get('data_points', 'N/A'):,}")
            print(f"  • Cycles analyzed: {reporter.analysis_metadata.get('cycles_analyzed', 'N/A'):,}")
            
            print("\n🌟 Open the HTML report in your browser for interactive viewing!")
            
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
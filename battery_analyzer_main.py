#!/usr/bin/env python3
"""
Battery Analyzer Main
배터리 데이터 분석 통합 실행 파일
단일/다중 경로 입력 지원, 제조사 자동 인식, 날짜 기반 파일명 생성
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging
import json
import re

# Import analysis modules
from battery_pattern_analyzer import BatteryPatternAnalyzer
from integrated_reporter import IntegratedBatteryReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'battery_analysis_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Manufacturer mapping dictionary
MANUFACTURER_MAPPING = {
    # Samsung SDI
    'SDI': ['SDI', 'Samsung', 'SAMSUNG', 'samsung_sdi', 'SamsungSDI', '삼성', '삼성SDI'],
    
    # ATL (Amperex Technology Limited)
    'ATL': ['ATL', 'Amperex', 'CATL', 'amperex', 'AMPEREX'],
    
    # LG Energy Solution
    'LGES': ['LGES', 'LG', 'LGChem', 'LG_Energy', 'LGEnergy', 'LG화학', 'LG에너지솔루션'],
    
    # COSMX (코스모신소재)
    'COSMX': ['COSMX', 'Cosmo', 'cosmo', '코스모', 'cosmos', '코스모신소재'],
    
    # BYD
    'BYD': ['BYD', 'byd', 'Build_Your_Dreams', 'Blade'],
    
    # Panasonic
    'Panasonic': ['Panasonic', 'PANA', 'Tesla_Panasonic', '파나소닉'],
    
    # SK Innovation
    'SK': ['SK', 'SKI', 'SK_Innovation', 'SK이노베이션', 'SKInnovation'],
    
    # EVE Energy
    'EVE': ['EVE', 'eve', 'EVE_Energy', 'EVEEnergy'],
    
    # Northvolt
    'Northvolt': ['Northvolt', 'NORTH', 'northvolt'],
    
    # SVOLT
    'SVOLT': ['SVOLT', 'svolt', '스볼트', 'Svolt'],
}

class BatteryAnalyzerMain:
    """배터리 분석 메인 클래스"""
    
    def __init__(self):
        """초기화"""
        self.analyzer = BatteryPatternAnalyzer()
        self.reporter = None
        self.manufacturer_info = {}
        self.battery_info = {}
        self.output_base_dir = Path("analysis_output")
        
    def extract_manufacturer(self, path: str) -> str:
        """
        경로에서 제조사 정보 추출
        
        Args:
            path: 데이터 경로
            
        Returns:
            제조사명 (표준화된 이름)
        """
        try:
            # Handle Unicode characters safely
            path_str = str(path).encode('utf-8', errors='ignore').decode('utf-8')
            path_upper = path_str.upper()
            
            # Check each manufacturer and their variations
            for standard_name, variations in MANUFACTURER_MAPPING.items():
                for variation in variations:
                    if variation.upper() in path_upper:
                        logger.info(f"Detected manufacturer: {standard_name} from '{variation}'")
                        return standard_name
            
            logger.warning(f"Unknown manufacturer in path: {path_str}")
            return "Unknown"
            
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            logger.error(f"Unicode error processing path: {path} - {str(e)}")
            return "Unknown"
    
    def extract_capacity(self, path: str) -> Optional[int]:
        """
        경로에서 용량 정보 추출
        
        Args:
            path: 데이터 경로
            
        Returns:
            용량 (mAh) 또는 None
        """
        try:
            # Handle Unicode characters safely
            path_str = str(path).encode('utf-8', errors='ignore').decode('utf-8')
            
            # Try different capacity patterns
            patterns = [
                r'(\d+)mAh',
                r'(\d+)Ah',
                r'(\d+\.\d+)Ah',
                r'(\d+)_mAh',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, path_str, re.IGNORECASE)
                if match:
                    capacity = float(match.group(1))
                    # Convert Ah to mAh if necessary
                    if 'Ah' in match.group(0) and 'mAh' not in match.group(0):
                        capacity = capacity * 1000
                    return int(capacity)
            
            return None
            
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            logger.error(f"Unicode error extracting capacity from path: {path} - {str(e)}")
            return None
    
    def extract_model(self, path: str) -> Optional[str]:
        """
        경로에서 모델명 추출
        
        Args:
            path: 데이터 경로
            
        Returns:
            모델명 또는 None
        """
        try:
            # Handle Unicode characters safely
            path_str = str(path).encode('utf-8', errors='ignore').decode('utf-8')
            
            # Model patterns
            patterns = [
                r'G\d+',           # G3, G4, etc.
                r'MP\d+',          # MP1, MP2, etc.
                r'NCM\d+',         # NCM811, NCM622, etc.
                r'NCA',            # NCA
                r'LFP',            # LFP
                r'LCO',            # LCO
                r'Series[\s_]?\d+', # Series1, Series_2
                r'Gen[\s_]?\d+',   # Gen1, Gen_2
                r'Model[\s_]?\w+', # Model_A, Model B
                r'Blade',          # BYD Blade
                r'\d{4,5}',        # 18650, 21700, 4680
            ]
            
            for pattern in patterns:
                match = re.search(pattern, path_str, re.IGNORECASE)
                if match:
                    return match.group(0).upper()
            
            return None
            
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            logger.error(f"Unicode error extracting model from path: {path} - {str(e)}")
            return None
    
    def extract_test_condition(self, path: str) -> str:
        """
        경로에서 테스트 조건 추출
        
        Args:
            path: 데이터 경로
            
        Returns:
            테스트 조건
        """
        try:
            # Handle Unicode characters safely
            path_str = str(path).encode('utf-8', errors='ignore').decode('utf-8')
            conditions = []
            path_lower = path_str.lower()
            
            # Temperature conditions
            if any(kw in path_lower for kw in ['상온', 'rt', 'room_temp', '25c', '25도', '25℃']):
                conditions.append('RT')
            elif any(kw in path_lower for kw in ['고온', 'ht', 'high_temp', '45c', '45도', '60c', '60도']):
                conditions.append('HT')
            elif any(kw in path_lower for kw in ['저온', 'lt', 'low_temp', '-20c', '영하', '-10c']):
                conditions.append('LT')
            
            # Test type
            if any(kw in path_lower for kw in ['수명', 'life', 'cycle', 'aging', '사이클']):
                conditions.append('Life')
            if any(kw in path_lower for kw in ['급속', 'fast', 'quick', '급속충전', 'fastcharge']):
                conditions.append('Fast')
            if any(kw in path_lower for kw in ['안전', 'safety', 'abuse', '과충전']):
                conditions.append('Safety')
            
            return '_'.join(conditions) if conditions else 'Standard'
            
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            logger.error(f"Unicode error extracting test condition from path: {path} - {str(e)}")
            return 'Standard'
    
    def extract_battery_info(self, path: str) -> Dict[str, Any]:
        """
        경로에서 배터리 정보 종합 추출
        
        Args:
            path: 데이터 경로
            
        Returns:
            배터리 정보 딕셔너리
        """
        info = {
            'manufacturer': self.extract_manufacturer(path),
            'capacity_mah': self.extract_capacity(path),
            'model': self.extract_model(path),
            'test_condition': self.extract_test_condition(path),
            'date': datetime.now().strftime('%Y%m%d'),
            'time': datetime.now().strftime('%H%M%S'),
            'original_path': str(path)
        }
        
        logger.info(f"Extracted battery info: {info}")
        return info
    
    def generate_output_filename(self, battery_info: Dict, analysis_type: str, 
                                extension: str = '') -> str:
        """
        표준화된 출력 파일명 생성
        
        Args:
            battery_info: 배터리 정보
            analysis_type: 분석 타입
            extension: 파일 확장자
            
        Returns:
            출력 파일명
        """
        try:
            components = []
            
            # Add manufacturer
            if battery_info.get('manufacturer') and battery_info['manufacturer'] != 'Unknown':
                # Clean manufacturer name for filename
                mfr_clean = str(battery_info['manufacturer']).replace('•', '_').replace('!', '')
                components.append(mfr_clean)
            
            # Add model
            if battery_info.get('model'):
                model_clean = str(battery_info['model']).replace('•', '_').replace('!', '')
                components.append(model_clean)
            
            # Add capacity
            if battery_info.get('capacity_mah'):
                components.append(f"{battery_info['capacity_mah']}mAh")
            
            # Add test condition
            if battery_info.get('test_condition') and battery_info['test_condition'] != 'Standard':
                condition_clean = str(battery_info['test_condition']).replace('•', '_').replace('!', '')
                components.append(condition_clean)
            
            # Add analysis type
            type_clean = str(analysis_type).replace('•', '_').replace('!', '')
            components.append(type_clean)
            
            # Add date and time
            components.append(battery_info['date'])
            components.append(battery_info['time'])
            
            # Join and clean filename
            filename = '_'.join(components)
            # Remove any remaining problematic characters for Windows filenames
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            
            if extension:
                filename += f".{extension}"
            
            return filename
            
        except Exception as e:
            logger.error(f"Error generating filename: {str(e)}")
            # Fallback filename
            fallback = f"analysis_{analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if extension:
                fallback += f".{extension}"
            return fallback
    
    def create_output_directory(self, battery_info: Dict, is_multi_channel: bool = False) -> Path:
        """
        출력 디렉토리 생성 (다중 채널 지원)
        
        Args:
            battery_info: 배터리 정보
            is_multi_channel: 다중 채널 여부
            
        Returns:
            생성된 디렉토리 경로
        """
        analysis_type = 'MultiChannel' if is_multi_channel else 'analysis'
        dir_name = self.generate_output_filename(battery_info, analysis_type)
        output_dir = self.output_base_dir / dir_name
        
        # Create subdirectories
        (output_dir / 'data').mkdir(parents=True, exist_ok=True)
        (output_dir / 'reports').mkdir(parents=True, exist_ok=True)
        (output_dir / 'visualizations').mkdir(parents=True, exist_ok=True)
        (output_dir / 'visualizations' / 'seaborn').mkdir(parents=True, exist_ok=True)
        (output_dir / 'visualizations' / 'domain').mkdir(parents=True, exist_ok=True)
        (output_dir / 'visualizations' / 'statistical').mkdir(parents=True, exist_ok=True)
        
        # 다중 채널인 경우 추가 디렉토리
        if is_multi_channel:
            (output_dir / 'cross_channel_analysis').mkdir(parents=True, exist_ok=True)
            (output_dir / 'channel_reports').mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created output directory: {output_dir}")
        return output_dir
    
    def analyze_single_path(self, path: str) -> Dict[str, Any]:
        """
        단일 경로 다중 채널 데이터 분석
        
        Args:
            path: 데이터 경로
            
        Returns:
            분석 결과 (다중 채널 지원)
        """
        logger.info(f"Analyzing single path with multi-channel support: {path}")
        
        # Check if path exists
        if not os.path.exists(path):
            error_msg = f"WARNING - 경로가 존재하지 않습니다: {path}"
            logger.warning(error_msg)
            print(error_msg)
            return {'error': error_msg, 'path': path}
        
        # Extract battery info
        battery_info = self.extract_battery_info(path)
        
        # Detect channels first
        channels = self.analyzer.detect_channels(path)
        is_multi_channel = len(channels) > 1
        
        # Create output directory with multi-channel support
        output_dir = self.create_output_directory(battery_info, is_multi_channel)
        
        # Update analyzer with battery info
        self.analyzer.capacity_info = battery_info
        
        try:
            # Run multi-channel pattern analysis
            result = self.analyzer.run_analysis(path, output_path=str(output_dir))
            
            # Process results
            if 'channels' in result:
                logger.info(f"처리된 채널: {len(result['channels'])}개")
                
                # Save individual channel data
                for channel_id, channel_result in result['channels'].items():
                    if 'error' not in channel_result:
                        # Create channel-specific data file
                        channel_data_filename = self.generate_output_filename(
                            battery_info, f'channel_{channel_id}_data', 'csv'
                        )
                        
                        # Save channel data if available
                        channel_data = result.get('data')
                        if channel_data is not None and not channel_data.empty:
                            channel_specific_data = channel_data[channel_data['channel_id'] == channel_id]
                            if not channel_specific_data.empty:
                                channel_data_path = output_dir / 'data' / channel_data_filename
                                channel_specific_data.to_csv(channel_data_path, index=False, encoding='utf-8')
                                logger.info(f"Saved channel {channel_id} data: {channel_data_path}")
                
                # Save cross-channel analysis if available
                if 'cross_channel_analysis' in result and result['cross_channel_analysis']:
                    cross_analysis_filename = self.generate_output_filename(
                        battery_info, 'cross_channel_analysis', 'json'
                    )
                    cross_analysis_path = output_dir / 'cross_channel_analysis' / cross_analysis_filename
                    
                    with open(cross_analysis_path, 'w', encoding='utf-8') as f:
                        json.dump(result['cross_channel_analysis'], f, indent=2, ensure_ascii=False, default=str)
                    
                    logger.info(f"Saved cross-channel analysis: {cross_analysis_path}")
                
                # Save complete dataset
                if 'data' in result and result['data'] is not None and not result['data'].empty:
                    complete_data_filename = self.generate_output_filename(
                        battery_info, 'complete_dataset', 'csv'
                    )
                    complete_data_path = output_dir / 'data' / complete_data_filename
                    result['data'].to_csv(complete_data_path, index=False, encoding='utf-8')
                    logger.info(f"Saved complete dataset: {complete_data_path}")
                    
                    # Run integrated reporter if data exists
                    try:
                        reporter = IntegratedBatteryReporter(output_dir=str(output_dir))
                        reporter.battery_info = battery_info
                        
                        # Run analysis with complete dataset
                        report_results = reporter.run_comprehensive_analysis(
                            data_path=str(complete_data_path),
                            parallel=True,
                            max_workers=3
                        )
                        
                        result['integrated_report'] = report_results
                        logger.info("Integrated report generated successfully")
                        
                    except Exception as e:
                        logger.error(f"Integrated report generation failed: {e}")
                        result['integrated_report_error'] = str(e)
            
            # Add metadata
            result['battery_info'] = battery_info
            result['output_dir'] = str(output_dir)
            result['is_multi_channel'] = is_multi_channel
            result['analysis_type'] = 'multi_channel'
            
            # Save comprehensive summary
            summary_filename = self.generate_output_filename(
                battery_info, 'multi_channel_summary', 'json'
            )
            summary_path = output_dir / 'reports' / summary_filename
            
            # Create a summary without the large data field for JSON serialization
            summary_result = {k: v for k, v in result.items() if k != 'data'}
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_result, f, indent=2, ensure_ascii=False, default=str)
            
            # Create multi-channel HTML report
            if is_multi_channel:
                html_report_path = self.create_multi_channel_report(result, output_dir)
                if html_report_path:
                    result['html_report'] = html_report_path
            
            logger.info(f"Multi-channel analysis completed for {path}")
            return result
            
        except Exception as e:
            logger.error(f"Error in multi-channel analysis for {path}: {str(e)}")
            return {'error': str(e), 'battery_info': battery_info, 'path': path}
    
    def analyze_multiple_paths(self, paths: List[str]) -> Dict[str, Any]:
        """
        다중 경로 데이터 분석 및 비교
        
        Args:
            paths: 데이터 경로 리스트
            
        Returns:
            분석 결과
        """
        logger.info(f"Analyzing {len(paths)} paths")
        
        results = {}
        manufacturers = {}
        
        # Analyze each path
        for path in paths:
            try:
                # Check path existence first
                if not os.path.exists(path):
                    error_msg = f"WARNING - 경로가 존재하지 않습니다: {path}"
                    logger.warning(error_msg)
                    print(error_msg)
                    results[path] = {'error': error_msg, 'path': path}
                    continue
                
                result = self.analyze_single_path(path)
                battery_info = result.get('battery_info', {})
                manufacturer = battery_info.get('manufacturer', 'Unknown')
                
                # Group by manufacturer
                if manufacturer not in manufacturers:
                    manufacturers[manufacturer] = []
                manufacturers[manufacturer].append(result)
                
                # Store individual result
                result_key = self.generate_output_filename(battery_info, 'result')
                results[result_key] = result
                
            except Exception as e:
                logger.error(f"Failed to analyze {path}: {str(e)}")
                results[path] = {'error': str(e)}
        
        # Create comparison report if multiple manufacturers
        if len(manufacturers) > 1:
            comparison_name = f"Comparison_{'_vs_'.join(manufacturers.keys())}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            comparison_dir = self.output_base_dir / comparison_name
            comparison_dir.mkdir(parents=True, exist_ok=True)
            
            # Create comparison analysis
            comparison_result = self.create_manufacturer_comparison(manufacturers, comparison_dir)
            results[comparison_name] = comparison_result
            
            logger.info(f"Created comparison report: {comparison_name}")
        
        return results
    
    def create_manufacturer_comparison(self, manufacturers: Dict, output_dir: Path) -> Dict:
        """
        제조사별 비교 분석 생성
        
        Args:
            manufacturers: 제조사별 분석 결과
            output_dir: 출력 디렉토리
            
        Returns:
            비교 분석 결과
        """
        comparison = {
            'manufacturers': list(manufacturers.keys()),
            'comparison_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'battery_count': sum(len(v) for v in manufacturers.values()),
            'comparisons': {}
        }
        
        # Extract key metrics for comparison
        for mfr, results in manufacturers.items():
            metrics = []
            for result in results:
                if 'battery_info' in result:
                    info = result['battery_info']
                    metrics.append({
                        'capacity_mah': info.get('capacity_mah'),
                        'model': info.get('model'),
                        'test_condition': info.get('test_condition')
                    })
            comparison['comparisons'][mfr] = metrics
        
        # Save comparison report
        comparison_path = output_dir / 'manufacturer_comparison.json'
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        # Create HTML comparison report
        html_content = self.generate_comparison_html(comparison)
        html_path = output_dir / 'comparison_report.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        comparison['output_dir'] = str(output_dir)
        comparison['report_files'] = {
            'json': str(comparison_path),
            'html': str(html_path)
        }
        
        return comparison
    
    def generate_comparison_html(self, comparison: Dict) -> str:
        """
        HTML 비교 리포트 생성
        
        Args:
            comparison: 비교 데이터
            
        Returns:
            HTML 내용
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Battery Manufacturer Comparison Report</title>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 20px;
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
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                .manufacturer-section {{
                    margin: 20px 0;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    border-left: 4px solid #3498db;
                }}
                .manufacturer-name {{
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #2c3e50;
                    margin-bottom: 10px;
                }}
                .battery-info {{
                    margin: 10px 0;
                    padding: 10px;
                    background-color: white;
                    border-radius: 5px;
                }}
                .comparison-summary {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .summary-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                }}
                .summary-value {{
                    font-size: 2em;
                    font-weight: bold;
                }}
                .summary-label {{
                    font-size: 0.9em;
                    opacity: 0.9;
                    margin-top: 5px;
                }}
                .timestamp {{
                    text-align: center;
                    color: #7f8c8d;
                    margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔋 Battery Manufacturer Comparison Report</h1>
                <p class="timestamp">Generated: {comparison['comparison_date']}</p>
                
                <div class="comparison-summary">
                    <div class="summary-card">
                        <div class="summary-value">{len(comparison['manufacturers'])}</div>
                        <div class="summary-label">Manufacturers</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-value">{comparison['battery_count']}</div>
                        <div class="summary-label">Total Batteries</div>
                    </div>
                </div>
                
                <h2>Manufacturer Details</h2>
        """
        
        for mfr, batteries in comparison['comparisons'].items():
            html += f"""
                <div class="manufacturer-section">
                    <div class="manufacturer-name">{mfr}</div>
                    <div>Number of batteries: {len(batteries)}</div>
            """
            
            for i, battery in enumerate(batteries, 1):
                html += f"""
                    <div class="battery-info">
                        <strong>Battery {i}:</strong>
                        Capacity: {battery.get('capacity_mah', 'N/A')} mAh | 
                        Model: {battery.get('model', 'N/A')} | 
                        Test: {battery.get('test_condition', 'N/A')}
                    </div>
                """
            
            html += "</div>"
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def create_multi_channel_report(self, result: Dict[str, Any], output_dir: Path) -> str:
        """
        다중 채널 HTML 리포트 생성
        
        Args:
            result: 분석 결과
            output_dir: 출력 디렉토리
            
        Returns:
            HTML 리포트 파일 경로
        """
        if not result.get('is_multi_channel', False):
            return ""
        
        channels = result.get('channels', {})
        cross_analysis = result.get('cross_channel_analysis', {})
        summary = result.get('summary', {})
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multi-Channel Battery Analysis Report</title>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    border-bottom: 2px solid #bdc3c7;
                    padding-bottom: 5px;
                }}
                .summary-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .summary-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                }}
                .summary-value {{
                    font-size: 2em;
                    font-weight: bold;
                }}
                .summary-label {{
                    font-size: 0.9em;
                    opacity: 0.9;
                    margin-top: 5px;
                }}
                .channel-section {{
                    margin: 20px 0;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    border-left: 4px solid #3498db;
                }}
                .channel-name {{
                    font-size: 1.3em;
                    font-weight: bold;
                    color: #2c3e50;
                    margin-bottom: 10px;
                }}
                .channel-stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 15px 0;
                }}
                .stat-item {{
                    background: white;
                    padding: 10px;
                    border-radius: 5px;
                    border: 1px solid #e0e0e0;
                }}
                .stat-label {{
                    font-weight: bold;
                    color: #666;
                    font-size: 0.9em;
                }}
                .stat-value {{
                    font-size: 1.2em;
                    color: #2c3e50;
                    margin-top: 5px;
                }}
                .performance-section {{
                    background-color: #e8f5e8;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                .best-channel {{
                    color: #27ae60;
                    font-weight: bold;
                }}
                .worst-channel {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                .timestamp {{
                    text-align: center;
                    color: #7f8c8d;
                    margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔋 Multi-Channel Battery Analysis Report</h1>
                <p class="timestamp">Generated: {summary.get('analysis_timestamp', 'Unknown')}</p>
                
                <div class="summary-grid">
                    <div class="summary-card">
                        <div class="summary-value">{summary.get('total_channels', 0)}</div>
                        <div class="summary-label">Total Channels</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-value">{summary.get('data_shape', [0, 0])[0]:,}</div>
                        <div class="summary-label">Total Data Points</div>
                    </div>
                </div>
                
                <h2>📊 Channel Analysis Results</h2>
        """
        
        # Add channel details
        for channel_id, channel_result in channels.items():
            if 'error' not in channel_result:
                equipment_type = channel_result.get('equipment_type', 'Unknown')
                data_shape = channel_result.get('data_shape', [0, 0])
                data_quality = channel_result.get('data_quality', {})
                
                html_content += f"""
                <div class="channel-section">
                    <div class="channel-name">📈 {channel_id}</div>
                    <div class="channel-stats">
                        <div class="stat-item">
                            <div class="stat-label">Equipment Type</div>
                            <div class="stat-value">{equipment_type}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Data Points</div>
                            <div class="stat-value">{data_shape[0]:,}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Source Files</div>
                            <div class="stat-value">{len(data_quality.get('source_files', []))}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Data Quality</div>
                            <div class="stat-value">{(100 - (data_quality.get('null_count', 0) / max(data_quality.get('total_rows', 1), 1) * 100)):.1f}%</div>
                        </div>
                    </div>
                </div>
                """
            else:
                html_content += f"""
                <div class="channel-section">
                    <div class="channel-name">❌ {channel_id}</div>
                    <div style="color: red;">Error: {channel_result.get('error', 'Unknown error')}</div>
                </div>
                """
        
        # Add cross-channel analysis if available
        if cross_analysis:
            performance_comp = cross_analysis.get('performance_comparison', {})
            if performance_comp:
                best_channel = performance_comp.get('best_performing_channel', 'N/A')
                worst_channel = performance_comp.get('worst_performing_channel', 'N/A')
                performance_spread = performance_comp.get('performance_spread', 0)
                
                html_content += f"""
                <h2>⚡ Cross-Channel Performance Comparison</h2>
                <div class="performance-section">
                    <p><strong>Best Performing Channel:</strong> <span class="best-channel">{best_channel}</span></p>
                    <p><strong>Worst Performing Channel:</strong> <span class="worst-channel">{worst_channel}</span></p>
                    <p><strong>Performance Spread:</strong> {performance_spread:.2f} mAh</p>
                </div>
                """
            
            # Channel statistics table
            channel_stats = cross_analysis.get('channel_statistics', {})
            if channel_stats:
                html_content += """
                <h2>📋 Channel Statistics Summary</h2>
                <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                    <thead>
                        <tr style="background-color: #3498db; color: white;">
                            <th style="padding: 12px; text-align: left;">Channel</th>
                            <th style="padding: 12px; text-align: right;">Capacity (mAh)</th>
                            <th style="padding: 12px; text-align: right;">Cycles</th>
                            <th style="padding: 12px; text-align: right;">Voltage (V)</th>
                            <th style="padding: 12px; text-align: right;">Data Quality</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for ch_id, stats in channel_stats.items():
                    capacity_mean = stats.get('capacity_mean', 0)
                    cycle_count = stats.get('cycle_count', 0)
                    voltage_mean = stats.get('voltage_mean', 0)
                    row_count = stats.get('row_count', 0)
                    
                    html_content += f"""
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 12px;"><strong>{ch_id}</strong></td>
                        <td style="padding: 12px; text-align: right;">{capacity_mean:.2f}</td>
                        <td style="padding: 12px; text-align: right;">{cycle_count}</td>
                        <td style="padding: 12px; text-align: right;">{voltage_mean/1000000 if voltage_mean > 1000 else voltage_mean:.2f}</td>
                        <td style="padding: 12px; text-align: right;">{row_count:,}</td>
                    </tr>
                    """
                
                html_content += """
                    </tbody>
                </table>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        html_path = output_dir / 'multi_channel_report.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Multi-channel HTML report created: {html_path}")
        return str(html_path)
    
    def run(self, paths: List[str]) -> Dict[str, Any]:
        """
        메인 실행 함수
        
        Args:
            paths: 분석할 경로 리스트
            
        Returns:
            분석 결과
        """
        if not paths:
            logger.error("No paths provided for analysis")
            return {'error': 'No paths provided'}
        
        # Single or multiple path analysis
        if len(paths) == 1:
            return self.analyze_single_path(paths[0])
        else:
            return self.analyze_multiple_paths(paths)


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='Battery Data Analyzer - Single/Multiple Path Analysis with Manufacturer Recognition'
    )
    
    parser.add_argument(
        '--path',
        type=str,
        help='Single data path for analysis'
    )
    
    parser.add_argument(
        '--paths',
        nargs='+',
        help='Multiple data paths for comparison analysis'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("🔋 Battery Data Analyzer")
    print("제조사 자동 인식 및 날짜 기반 파일명 생성 시스템")
    print("=" * 80)
    
    # Determine paths to analyze
    paths = []
    
    if args.path:
        paths = [args.path]
    elif args.paths:
        paths = args.paths
    elif args.interactive or len(sys.argv) == 1:
        # Interactive mode
        print("\n대화형 모드로 실행합니다.")
        print("분석할 경로를 입력하세요 (여러 경로는 쉼표로 구분):")
        user_input = input("> ").strip()
        
        if not user_input:
            print("❌ 경로가 입력되지 않았습니다.")
            sys.exit(1)
        
        # Split by comma or semicolon
        paths = [p.strip() for p in re.split('[,;]', user_input) if p.strip()]
    
    if not paths:
        print("❌ 분석할 경로가 없습니다.")
        print("\n사용법:")
        print("  단일 경로: python battery_analyzer_main.py --path 'data/LGES_4352mAh'")
        print("  다중 경로: python battery_analyzer_main.py --paths 'data/LGES' 'data/SDI' 'data/ATL'")
        print("  대화형: python battery_analyzer_main.py --interactive")
        sys.exit(1)
    
    # Create analyzer
    analyzer = BatteryAnalyzerMain()
    
    # Display detected information
    print(f"\n📂 분석할 경로: {len(paths)}개")
    for i, path in enumerate(paths, 1):
        battery_info = analyzer.extract_battery_info(path)
        print(f"\n[경로 {i}] {path}")
        print(f"  • 제조사: {battery_info['manufacturer']}")
        print(f"  • 용량: {battery_info.get('capacity_mah', 'Unknown')} mAh")
        print(f"  • 모델: {battery_info.get('model', 'Unknown')}")
        print(f"  • 테스트: {battery_info['test_condition']}")
    
    # Confirm analysis
    print("\n분석을 시작하시겠습니까? (y/n)")
    confirm = input("> ").strip().lower()
    
    if confirm != 'y':
        print("분석을 취소했습니다.")
        sys.exit(0)
    
    # Run analysis
    print("\n🔍 분석을 시작합니다...")
    results = analyzer.run(paths)
    
    # Display results
    if 'error' in results:
        print(f"\n❌ 분석 실패: {results['error']}")
    else:
        print("\n✅ 분석 완료!")
        
        # Display individual results
        for key, result in results.items():
            if isinstance(result, dict) and 'battery_info' in result:
                info = result['battery_info']
                print(f"\n📊 {info['manufacturer']} - {info.get('model', 'N/A')}")
                print(f"   출력 디렉토리: {result.get('output_dir', 'N/A')}")
        
        # Display comparison if created
        comparison_keys = [k for k in results.keys() if k.startswith('Comparison_')]
        if comparison_keys:
            for comp_key in comparison_keys:
                comp = results[comp_key]
                print(f"\n📈 비교 리포트 생성됨:")
                print(f"   제조사: {', '.join(comp['manufacturers'])}")
                print(f"   배터리 수: {comp['battery_count']}")
                if 'report_files' in comp:
                    print(f"   HTML: {comp['report_files']['html']}")
    
    print("\n" + "=" * 80)
    print("분석이 완료되었습니다. 결과는 analysis_output 폴더를 확인하세요.")
    print("=" * 80)


if __name__ == "__main__":
    main()
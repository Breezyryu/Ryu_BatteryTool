#!/usr/bin/env python3
"""
배터리 패턴 동적 분석기
경로 기반 용량 정보 추출 및 전압/전류 패턴을 통한 동적 사이클 패턴 정의 시스템
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
import logging

# 경고 필터링 및 로깅 설정
warnings.filterwarnings('ignore')

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 콘솔 핸들러
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', 
                                datefmt='%H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class BatteryPatternAnalyzer:
    """동적 배터리 패턴 분석 클래스"""
    
    def __init__(self):
        """초기화"""
        self.capacity_info = None
        self.equipment_type = None
        self.data_path = None
        self.analysis_results = {}
        
        # PNE 46컬럼 정의
        self.pne_columns = [
            'Index', 'default(2)', 'Step_type', 'ChgDchg', 'Current_application', 
            'CCCV', 'EndState', 'Step_count', 'Voltage[uV]', 'Current[uA]', 
            'Chg_Capacity[uAh]', 'Dchg_Capacity[uAh]', 'Chg_Power[mW]', 
            'Dchg_Power[mW]', 'Chg_WattHour[Wh]', 'Dchg_WattHour[Wh]', 
            'repeat_pattern_count', 'StepTime[1/100s]', 'TotTime[day]', 
            'TotTime[1/100s]', 'Impedance', 'Temperature1', 'Temperature2', 
            'Temperature3', 'Temperature4', 'col25', 'Repeat_count', 
            'TotalCycle', 'Current_Cycle', 'Average_Voltage[uV]', 
            'Average_Current[uA]', 'col31', 'col32', 'CV_section', 
            'Date', 'Time', 'col35', 'col36', 'col37', 'col38', 
            'CC_charge', 'CV_section2', 'Discharge', 'col42', 
            'Average_voltage_section', 'Accumulated_step', 'Voltage_max[uV]', 'Voltage_min[uV]'
        ]
        
    def extract_capacity_from_path(self, data_path: str) -> Dict[str, Any]:
        """
        경로에서 용량 정보 및 기타 배터리 정보 추출
        
        Args:
            data_path: 데이터 경로 (예: "D:\\pne\\LGES_G3_MP1_4352mAh_상온수명")
            
        Returns:
            배터리 정보 딕셔너리
        """
        path_str = str(data_path).replace('/', '\\').replace('//', '\\')
        
        # 용량 정보 추출 (mAh)
        capacity_pattern = r'(\d+)mAh'
        capacity_match = re.search(capacity_pattern, path_str)
        capacity_mah = int(capacity_match.group(1)) if capacity_match else 4800
        
        # 장비 타입 판별
        if 'pne' in path_str.lower():
            equipment_type = 'PNE'
        elif 'toyo' in path_str.lower():
            equipment_type = 'Toyo'
        else:
            # 폴더 구조로 판별
            path_obj = Path(path_str)
            if any(p.name.startswith('M01Ch') for p in path_obj.rglob('*')):
                equipment_type = 'PNE'
            elif any(p.name == 'CAPACITY.LOG' for p in path_obj.rglob('*')):
                equipment_type = 'Toyo'
            else:
                equipment_type = 'Unknown'
        
        # 기타 정보 추출
        battery_info = {
            'capacity_mah': capacity_mah,
            'capacity_ah': capacity_mah / 1000,
            'equipment_type': equipment_type,
            'path': path_str
        }
        
        # 추가 정보 추출 시도
        info_patterns = {
            'manufacturer': r'(LGES|LG|Samsung|CATL)',
            'grade': r'(G\d+)',
            'version': r'(MP\d+)',
            'temperature': r'(상온|고온|저온|RT\d+)',
            'test_type': r'(수명|용량|성능|충방전)'
        }
        
        for key, pattern in info_patterns.items():
            match = re.search(pattern, path_str)
            if match:
                battery_info[key] = match.group(1)
        
        return battery_info
    
    def detect_equipment_type(self, data_path: str) -> str:
        """
        데이터 구조를 통한 장비 타입 자동 감지
        
        Args:
            data_path: 데이터 경로
            
        Returns:
            장비 타입 ('PNE' 또는 'Toyo')
        """
        path_obj = Path(data_path)
        
        # PNE 특징: M01Ch###[###]/Restore/ 구조
        pne_indicators = [
            any(p.name.startswith('M01Ch') for p in path_obj.rglob('*')),
            any(p.name == 'Restore' for p in path_obj.rglob('*')),
            any('SaveData' in p.name for p in path_obj.rglob('*.csv')),
            any('savingFileIndex' in p.name for p in path_obj.rglob('*.csv'))
        ]
        
        # Toyo 특징: CAPACITY.LOG와 숫자 파일들
        toyo_indicators = [
            any(p.name == 'CAPACITY.LOG' for p in path_obj.rglob('*')),
            len([p for p in path_obj.rglob('*') if p.name.isdigit()]) > 10,
            any(re.match(r'^\d{6}$', p.name) for p in path_obj.rglob('*'))
        ]
        
        pne_score = sum(pne_indicators)
        toyo_score = sum(toyo_indicators)
        
        if pne_score > toyo_score:
            return 'PNE'
        elif toyo_score > pne_score:
            return 'Toyo'
        else:
            # 경로 문자열로 최종 판별
            path_str = str(data_path).lower()
            if 'pne' in path_str:
                return 'PNE'
            elif 'toyo' in path_str:
                return 'Toyo'
            else:
                return 'Unknown'
    
    def load_and_concatenate_data(self, data_path: str) -> pd.DataFrame:
        """
        데이터 로드 및 파일 연결
        
        Args:
            data_path: 데이터 경로
            
        Returns:
            통합된 데이터프레임
        """
        path_obj = Path(data_path)
        combined_data = pd.DataFrame()
        
        if self.equipment_type == 'PNE':
            # PNE 데이터 처리
            restore_folders = list(path_obj.rglob('Restore'))
            
            for restore_folder in restore_folders:
                # SaveData 파일들 로드
                save_data_files = sorted([f for f in restore_folder.glob("*_SaveData*.csv")])
                
                for file_path in save_data_files:
                    try:
                        # 탭 구분자로 로드, 헤더 없음
                        df = pd.read_csv(file_path, sep='\t', header=None, 
                                       names=self.pne_columns, low_memory=False)
                        
                        # 데이터 변환
                        self._convert_pne_data(df)
                        
                        # 파일 정보 추가
                        df['source_file'] = file_path.name
                        df['channel'] = restore_folder.parent.name
                        
                        combined_data = pd.concat([combined_data, df], ignore_index=True)
                        logger.info(f"로드됨: {file_path.name} - {len(df)} 행")
                        
                    except Exception as e:
                        logger.error(f"파일 로드 실패 {file_path.name}: {e}")
                        
        elif self.equipment_type == 'Toyo':
            # Toyo 데이터 처리
            combined_data = self._load_toyo_data(path_obj, combined_data)
            
        return combined_data
    
    def _convert_pne_data(self, df: pd.DataFrame):
        """PNE 데이터 변환"""
        # 수치 컬럼 변환
        numeric_columns = ['Voltage[uV]', 'Current[uA]', 'Chg_Capacity[uAh]', 'Dchg_Capacity[uAh]']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 단위 변환
        if 'Voltage[uV]' in df.columns:
            df['Voltage[V]'] = df['Voltage[uV]'] / 1000000
        if 'Current[uA]' in df.columns:
            df['Current[A]'] = df['Current[uA]'] / 1000000
        if 'Chg_Capacity[uAh]' in df.columns:
            df['Chg_Capacity[Ah]'] = df['Chg_Capacity[uAh]'] / 1000000
        if 'Dchg_Capacity[uAh]' in df.columns:
            df['Dchg_Capacity[Ah]'] = df['Dchg_Capacity[uAh]'] / 1000000
        
        # 시간 변환
        if 'StepTime[1/100s]' in df.columns:
            df['Time[s]'] = pd.to_numeric(df['StepTime[1/100s]'], errors='coerce') / 100
    
    def _load_toyo_data(self, path_obj: Path, combined_data: pd.DataFrame):
        """Toyo 데이터 로드"""
        # CAPACITY.LOG 로드
        capacity_files = list(path_obj.rglob('CAPACITY.LOG'))
        
        for capacity_file in capacity_files:
            try:
                capacity_df = pd.read_csv(capacity_file)
                capacity_df['source_file'] = 'CAPACITY.LOG'
                capacity_df['equipment_type'] = 'Toyo'
                
                # mAh를 Ah로 변환
                if 'Cap[mAh]' in capacity_df.columns:
                    capacity_df['Cap[Ah]'] = capacity_df['Cap[mAh]'] / 1000
                
                combined_data = pd.concat([combined_data, capacity_df], ignore_index=True)
                logger.info(f"로드됨: CAPACITY.LOG - {len(capacity_df)} 행")
                
            except Exception as e:
                logger.error(f"CAPACITY.LOG 로드 실패: {e}")
        
        # 개별 측정 파일들 로드 (선택적으로 몇 개만)
        measurement_files = sorted([f for f in path_obj.rglob('*') 
                                  if f.is_file() and f.name.isdigit()])[:10]
        
        for file_path in measurement_files:
            try:
                # 헤더 찾기
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                header_line = None
                for i, line in enumerate(lines):
                    if 'Date,Time' in line:
                        header_line = i
                        break
                
                if header_line is not None:
                    df = pd.read_csv(file_path, skiprows=header_line)
                    
                    # 전압/전류 변환
                    if 'Voltage[V]' in df.columns:
                        df['Voltage[V]'] = pd.to_numeric(df['Voltage[V]'], errors='coerce')
                    if 'Current[mA]' in df.columns:
                        df['Current[mA]'] = pd.to_numeric(df['Current[mA]'], errors='coerce')
                        df['Current[A]'] = df['Current[mA]'] / 1000
                    
                    df['source_file'] = file_path.name
                    combined_data = pd.concat([combined_data, df], ignore_index=True)
                    
            except Exception as e:
                print(f"측정 파일 로드 실패 {file_path.name}: {e}")
                
        return combined_data
    
    def load_and_concatenate_multi_paths(self, data_paths: List[str]) -> pd.DataFrame:
        """
        다중 경로에서 데이터 로드 및 채널별 연결
        
        Args:
            data_paths: 데이터 경로 리스트
            
        Returns:
            통합된 데이터프레임 (채널별로 정렬)
        """
        all_data = []
        path_metadata = []
        
        for path_idx, data_path in enumerate(data_paths):
            print(f"경로 {path_idx+1}/{len(data_paths)} 로딩: {data_path}")
            
            # 단일 경로 데이터 로드
            path_data = self.load_and_concatenate_data(data_path)
            
            if len(path_data) > 0:
                # 경로 정보 추가
                path_data['path_index'] = path_idx
                path_data['path_source'] = data_path
                
                # 각 경로에서 채널별 사이클 범위 기록
                if 'TotalCycle' in path_data.columns and 'channel' in path_data.columns:
                    for channel in path_data['channel'].unique():
                        channel_data = path_data[path_data['channel'] == channel]
                        if len(channel_data) > 0:
                            cycles = channel_data['TotalCycle'].dropna()
                            if len(cycles) > 0:
                                path_metadata.append({
                                    'path_index': path_idx,
                                    'path_source': data_path,
                                    'channel': channel,
                                    'cycle_start': int(cycles.min()),
                                    'cycle_end': int(cycles.max()),
                                    'data_points': len(channel_data)
                                })
                
                all_data.append(path_data)
                print(f"  로드됨: {len(path_data):,} 행")
            else:
                print(f"  경고: 데이터가 없습니다.")
        
        if not all_data:
            print("모든 경로에서 데이터 로드 실패")
            return pd.DataFrame()
        
        # 모든 데이터 결합
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # 메타데이터 저장 (디버깅 및 분석용)
        self.path_metadata = path_metadata
        
        # 채널별로 정렬 (연속성 확보)
        if 'channel' in combined_data.columns:
            combined_data = self._sort_data_by_channel_continuity(combined_data)
        
        print(f"다중 경로 통합 완료: {len(combined_data):,} 행")
        
        return combined_data
    
    def _sort_data_by_channel_continuity(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        채널별 데이터 연속성을 고려한 정렬
        
        Args:
            data: 통합 데이터
            
        Returns:
            정렬된 데이터프레임
        """
        if 'channel' not in data.columns:
            return data
        
        sorted_data = []
        
        # 채널별로 처리
        for channel in sorted(data['channel'].unique()):
            channel_data = data[data['channel'] == channel].copy()
            
            # 경로별, 사이클별, 시간별 정렬
            sort_columns = []
            if 'path_index' in channel_data.columns:
                sort_columns.append('path_index')
            if 'TotalCycle' in channel_data.columns:
                sort_columns.append('TotalCycle')
            if 'Time[s]' in channel_data.columns:
                sort_columns.append('Time[s]')
            elif 'StepTime[1/100s]' in channel_data.columns:
                sort_columns.append('StepTime[1/100s]')
            
            if sort_columns:
                channel_data = channel_data.sort_values(sort_columns)
            
            sorted_data.append(channel_data)
        
        return pd.concat(sorted_data, ignore_index=True)
    
    def verify_channel_continuity(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        채널별 데이터 연속성 검증
        
        Args:
            data: 통합 데이터
            
        Returns:
            채널별 연속성 정보
        """
        continuity_info = {}
        
        if 'channel' not in data.columns:
            return continuity_info
        
        for channel in data['channel'].unique():
            channel_data = data[data['channel'] == channel]
            
            info = {
                'channel': channel,
                'total_data_points': len(channel_data),
                'segments': 0,
                'total_cycles': 0,
                'cycle_gaps': [],
                'path_segments': []
            }
            
            if 'TotalCycle' in channel_data.columns:
                cycles = channel_data['TotalCycle'].dropna()
                if len(cycles) > 0:
                    info['total_cycles'] = int(cycles.max())
                    
                    # 경로별 사이클 구간 분석
                    if 'path_index' in channel_data.columns:
                        for path_idx in sorted(channel_data['path_index'].unique()):
                            path_data = channel_data[channel_data['path_index'] == path_idx]
                            path_cycles = path_data['TotalCycle'].dropna()
                            
                            if len(path_cycles) > 0:
                                segment_info = {
                                    'path_index': int(path_idx),
                                    'cycle_start': int(path_cycles.min()),
                                    'cycle_end': int(path_cycles.max()),
                                    'data_points': len(path_data)
                                }
                                info['path_segments'].append(segment_info)
                        
                        info['segments'] = len(info['path_segments'])
                        
                        # 사이클 연속성 검사 (갭 찾기)
                        if len(info['path_segments']) > 1:
                            for i in range(len(info['path_segments']) - 1):
                                current_end = info['path_segments'][i]['cycle_end']
                                next_start = info['path_segments'][i+1]['cycle_start']
                                
                                if next_start > current_end + 1:
                                    gap = {
                                        'after_path': info['path_segments'][i]['path_index'],
                                        'before_path': info['path_segments'][i+1]['path_index'],
                                        'gap_start': current_end + 1,
                                        'gap_end': next_start - 1,
                                        'gap_size': next_start - current_end - 1
                                    }
                                    info['cycle_gaps'].append(gap)
            
            continuity_info[channel] = info
        
        return continuity_info
    
    def _generate_multi_path_report(self, report_path: str, patterns: Dict[str, Any], continuity_info: Dict[str, Dict[str, Any]]):
        """
        다중 경로 분석 리포트 생성
        
        Args:
            report_path: 리포트 파일 경로
            patterns: 패턴 분석 결과
            continuity_info: 연속성 정보
        """
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("배터리 다중 경로 패턴 분석 보고서\n")
            f.write("=" * 80 + "\n")
            f.write(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # 배터리 정보
            if self.capacity_info:
                f.write(f"배터리 용량: {self.capacity_info.get('capacity_mah', 'Unknown')}mAh\n")
                f.write(f"장비 타입: {self.capacity_info.get('equipment_type', 'Unknown')}\n")
            
            f.write("\n1. 다중 경로 정보\n")
            f.write("-" * 40 + "\n")
            if hasattr(self, 'path_metadata'):
                f.write(f"  총 경로 수: {len(set(m['path_index'] for m in self.path_metadata))}\n")
                for path_info in self.path_metadata:
                    f.write(f"  경로 {path_info['path_index']+1}: {path_info['channel']} "
                           f"(사이클 {path_info['cycle_start']}-{path_info['cycle_end']}, "
                           f"{path_info['data_points']:,}개 데이터)\n")
            
            f.write("\n2. 채널별 연속성 분석\n")
            f.write("-" * 40 + "\n")
            for channel, info in continuity_info.items():
                f.write(f"  채널 {channel}:\n")
                f.write(f"    - 총 구간 수: {info['segments']}\n")
                f.write(f"    - 총 사이클 수: {info['total_cycles']}\n")
                f.write(f"    - 총 데이터 포인트: {info['total_data_points']:,}\n")
                
                if info['cycle_gaps']:
                    f.write(f"    - 사이클 갭: {len(info['cycle_gaps'])}개\n")
                    for gap in info['cycle_gaps']:
                        f.write(f"      경로 {gap['after_path']+1} → {gap['before_path']+1}: "
                               f"사이클 {gap['gap_start']}-{gap['gap_end']} ({gap['gap_size']}개 사이클 누락)\n")
                else:
                    f.write(f"    - 사이클 연속성: 양호\n")
                
                f.write(f"    - 경로별 구간:\n")
                for segment in info['path_segments']:
                    f.write(f"      경로 {segment['path_index']+1}: "
                           f"사이클 {segment['cycle_start']}-{segment['cycle_end']} "
                           f"({segment['data_points']:,}개 데이터)\n")
                f.write("\n")
            
            # 기존 패턴 분석 결과 추가
            self._write_pattern_analysis_to_report(f, patterns)
    
    def _write_pattern_analysis_to_report(self, f, patterns: Dict[str, Any]):
        """패턴 분석 결과를 리포트에 작성"""
        f.write("\n3. 패턴 분석 결과\n")
        f.write("-" * 40 + "\n")
        
        # 사이클 정보
        if 'cycles_detected' in patterns:
            cycles_info = patterns['cycles_detected']
            f.write(f"  총 사이클 수: {cycles_info.get('total_cycles', 0)}\n")
            if 'cycle_range' in cycles_info:
                cycle_range = cycles_info['cycle_range']
                f.write(f"  사이클 범위: {cycle_range[0]} ~ {cycle_range[1]}\n")
        
        # 패턴 통계
        pattern_stats = patterns.get('pattern_statistics', {})
        
        if 'charge_statistics' in pattern_stats:
            charge_stats = pattern_stats['charge_statistics']
            f.write(f"\n  충전 패턴 통계:\n")
            f.write(f"    - 패턴 수: {charge_stats.get('pattern_count', 0)}\n")
            f.write(f"    - 평균 C-rate: {charge_stats.get('avg_c_rate', 0):.3f}C\n")
            f.write(f"    - 최대 C-rate: {charge_stats.get('max_c_rate', 0):.3f}C\n")
            f.write(f"    - 평균 충전 시간: {charge_stats.get('avg_duration', 0)/60:.1f}분\n")
        
        if 'discharge_statistics' in pattern_stats:
            discharge_stats = pattern_stats['discharge_statistics']
            f.write(f"\n  방전 패턴 통계:\n")
            f.write(f"    - 패턴 수: {discharge_stats.get('pattern_count', 0)}\n")
            f.write(f"    - 평균 C-rate: {discharge_stats.get('avg_c_rate', 0):.3f}C\n")
            f.write(f"    - 최대 C-rate: {discharge_stats.get('max_c_rate', 0):.3f}C\n")
            f.write(f"    - 평균 방전 시간: {discharge_stats.get('avg_duration', 0)/60:.1f}분\n")
            f.write(f"    - 평균 방전 용량: {discharge_stats.get('avg_capacity', 0):.3f}Ah\n")
        
        # 용량 분석 (Toyo 데이터)
        if 'capacity_data' in patterns and patterns['capacity_data']:
            f.write(f"\n  용량 분석:\n")
            f.write(f"    - 초기 용량: {pattern_stats.get('initial_capacity', 0):.3f}Ah\n")
            f.write(f"    - 최종 용량: {pattern_stats.get('final_capacity', 0):.3f}Ah\n")
            f.write(f"    - 용량 감소율: {pattern_stats.get('capacity_fade', 0):.2f}%\n")
    
    def _get_essential_columns(self) -> List[str]:
        """필수 컬럼 리스트 반환"""
        if self.equipment_type == 'PNE':
            return [
                'TotalCycle', 'Current_Cycle', 'Step_type', 'Voltage[V]', 'Current[A]',
                'Chg_Capacity[Ah]', 'Dchg_Capacity[Ah]', 'Time[s]', 'source_file', 'channel',
                'path_index', 'path_source'
            ]
        else:  # Toyo
            return [
                'Date', 'Time', 'Cycle', 'TotlCycle', 'Voltage[V]', 'Current[A]', 
                'Cap[Ah]', 'Condition', 'Mode', 'source_file', 'equipment_type',
                'path_index', 'path_source'
            ]
    
    def analyze_cycle_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        동적 사이클 패턴 분석
        
        Args:
            data: 통합 데이터프레임
            
        Returns:
            패턴 분석 결과
        """
        pattern_analysis = {
            'cycles_detected': {},
            'charge_patterns': [],
            'discharge_patterns': [],
            'voltage_profiles': {},
            'current_profiles': {},
            'pattern_statistics': {}
        }
        
        if self.equipment_type == 'PNE':
            pattern_analysis = self._analyze_pne_patterns(data)
        elif self.equipment_type == 'Toyo':
            pattern_analysis = self._analyze_toyo_patterns(data)
        
        return pattern_analysis
    
    def _analyze_pne_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """PNE 데이터 패턴 분석"""
        patterns = {
            'cycles_detected': {},
            'charge_patterns': [],
            'discharge_patterns': [],
            'rest_patterns': [],
            'voltage_profiles': {},
            'current_profiles': {},
            'pattern_statistics': {}
        }
        
        if 'TotalCycle' not in data.columns:
            print("TotalCycle 컬럼을 찾을 수 없습니다.")
            return patterns
        
        # 사이클별 분석
        unique_cycles = sorted(data['TotalCycle'].unique())
        patterns['cycles_detected'] = {
            'total_cycles': len(unique_cycles),
            'cycle_range': [int(unique_cycles[0]), int(unique_cycles[-1])],
            'cycles_list': [int(c) for c in unique_cycles]
        }
        
        # 각 사이클별 패턴 분석
        for cycle in unique_cycles[:10]:  # 처음 10개 사이클만 상세 분석
            cycle_data = data[data['TotalCycle'] == cycle].copy()
            
            if len(cycle_data) == 0:
                continue
            
            # 충전 패턴 추출
            if 'Step_type' in cycle_data.columns:
                charge_steps = cycle_data[cycle_data['Step_type'] == 1]
                discharge_steps = cycle_data[cycle_data['Step_type'] == 2]
                rest_steps = cycle_data[cycle_data['Step_type'] == 3]
                
                if len(charge_steps) > 0:
                    charge_pattern = self._extract_charge_pattern(charge_steps)
                    charge_pattern['cycle'] = int(cycle)
                    patterns['charge_patterns'].append(charge_pattern)
                
                if len(discharge_steps) > 0:
                    discharge_pattern = self._extract_discharge_pattern(discharge_steps)
                    discharge_pattern['cycle'] = int(cycle)
                    patterns['discharge_patterns'].append(discharge_pattern)
                
                if len(rest_steps) > 0:
                    rest_pattern = self._extract_rest_pattern(rest_steps)
                    rest_pattern['cycle'] = int(cycle)
                    patterns['rest_patterns'].append(rest_pattern)
        
        # 패턴 통계
        patterns['pattern_statistics'] = self._calculate_pattern_statistics(patterns)
        
        return patterns
    
    def _extract_charge_pattern(self, charge_data: pd.DataFrame) -> Dict[str, Any]:
        """충전 패턴 추출"""
        pattern = {}
        
        if len(charge_data) == 0:
            return pattern
        
        # 전압 프로파일
        if 'Voltage[V]' in charge_data.columns:
            voltages = charge_data['Voltage[V]'].dropna()
            pattern['voltage_start'] = float(voltages.min())
            pattern['voltage_end'] = float(voltages.max())
            pattern['voltage_profile'] = voltages.tolist()
        
        # 전류 프로파일
        if 'Current[A]' in charge_data.columns:
            currents = charge_data['Current[A]'].dropna()
            pattern['current_max'] = float(currents.max())
            pattern['current_min'] = float(currents.min())
            pattern['current_profile'] = currents.tolist()
            
            # C-rate 추정 (배터리 용량 기준)
            if self.capacity_info and 'capacity_ah' in self.capacity_info:
                max_current = currents.max()
                c_rate = abs(max_current) / self.capacity_info['capacity_ah']
                pattern['estimated_c_rate'] = float(c_rate)
        
        # 충전 단계 감지 (CC/CV 구분)
        pattern['charge_phases'] = self._detect_charge_phases(charge_data)
        
        # 시간 정보
        if 'Time[s]' in charge_data.columns:
            times = charge_data['Time[s]'].dropna()
            pattern['duration'] = float(times.max() - times.min())
        
        return pattern
    
    def _extract_discharge_pattern(self, discharge_data: pd.DataFrame) -> Dict[str, Any]:
        """방전 패턴 추출"""
        pattern = {}
        
        if len(discharge_data) == 0:
            return pattern
        
        # 전압 프로파일
        if 'Voltage[V]' in discharge_data.columns:
            voltages = discharge_data['Voltage[V]'].dropna()
            pattern['voltage_start'] = float(voltages.max())
            pattern['voltage_end'] = float(voltages.min())
            pattern['voltage_profile'] = voltages.tolist()
        
        # 전류 프로파일
        if 'Current[A]' in discharge_data.columns:
            currents = discharge_data['Current[A]'].dropna()
            pattern['current_max'] = float(abs(currents.min()))  # 방전은 음수
            pattern['current_profile'] = currents.tolist()
            
            # C-rate 추정
            if self.capacity_info and 'capacity_ah' in self.capacity_info:
                max_discharge_current = abs(currents.min())
                c_rate = max_discharge_current / self.capacity_info['capacity_ah']
                pattern['estimated_c_rate'] = float(c_rate)
        
        # 방전 단계 감지 (CC 단계별)
        pattern['discharge_phases'] = self._detect_discharge_phases(discharge_data)
        
        # 용량 정보
        if 'Dchg_Capacity[Ah]' in discharge_data.columns:
            capacities = discharge_data['Dchg_Capacity[Ah]'].dropna()
            if len(capacities) > 0:
                pattern['capacity_discharged'] = float(capacities.max())
        
        # 시간 정보
        if 'Time[s]' in discharge_data.columns:
            times = discharge_data['Time[s]'].dropna()
            pattern['duration'] = float(times.max() - times.min())
        
        return pattern
    
    def _extract_rest_pattern(self, rest_data: pd.DataFrame) -> Dict[str, Any]:
        """휴지 패턴 추출"""
        pattern = {}
        
        if len(rest_data) == 0:
            return pattern
        
        # 휴지 시간
        if 'Time[s]' in rest_data.columns:
            times = rest_data['Time[s]'].dropna()
            pattern['duration'] = float(times.max() - times.min())
        
        # 휴지 중 전압 변화
        if 'Voltage[V]' in rest_data.columns:
            voltages = rest_data['Voltage[V]'].dropna()
            pattern['voltage_start'] = float(voltages.iloc[0]) if len(voltages) > 0 else 0
            pattern['voltage_end'] = float(voltages.iloc[-1]) if len(voltages) > 0 else 0
            pattern['voltage_drop'] = pattern['voltage_start'] - pattern['voltage_end']
        
        return pattern
    
    def _detect_charge_phases(self, charge_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """충전 단계 감지 (CC/CV)"""
        phases = []
        
        if 'Voltage[V]' not in charge_data.columns or 'Current[A]' not in charge_data.columns:
            return phases
        
        voltages = charge_data['Voltage[V]'].values
        currents = charge_data['Current[A]'].values
        
        # 전류 변화율 계산으로 CC/CV 구분
        current_diff = np.diff(currents)
        
        # CC 구간: 전류가 거의 일정 (변화율 < 임계값)
        # CV 구간: 전류가 감소 (변화율 < -임계값)
        cc_threshold = 0.01  # A/step
        
        current_phase = None
        phase_start = 0
        
        for i, diff in enumerate(current_diff):
            if abs(diff) < cc_threshold:
                # CC 구간
                if current_phase != 'CC':
                    if current_phase is not None:
                        # 이전 단계 종료
                        phases.append({
                            'type': current_phase,
                            'start_idx': phase_start,
                            'end_idx': i,
                            'voltage_range': [voltages[phase_start], voltages[i]],
                            'current_range': [currents[phase_start], currents[i]]
                        })
                    current_phase = 'CC'
                    phase_start = i
            else:
                # CV 구간으로 추정
                if current_phase != 'CV':
                    if current_phase is not None:
                        phases.append({
                            'type': current_phase,
                            'start_idx': phase_start,
                            'end_idx': i,
                            'voltage_range': [voltages[phase_start], voltages[i]],
                            'current_range': [currents[phase_start], currents[i]]
                        })
                    current_phase = 'CV'
                    phase_start = i
        
        # 마지막 단계 추가
        if current_phase is not None:
            phases.append({
                'type': current_phase,
                'start_idx': phase_start,
                'end_idx': len(voltages) - 1,
                'voltage_range': [voltages[phase_start], voltages[-1]],
                'current_range': [currents[phase_start], currents[-1]]
            })
        
        return phases
    
    def _detect_discharge_phases(self, discharge_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """방전 단계 감지"""
        phases = []
        
        if 'Voltage[V]' not in discharge_data.columns or 'Current[A]' not in discharge_data.columns:
            return phases
        
        voltages = discharge_data['Voltage[V]'].values
        currents = discharge_data['Current[A]'].values
        
        # 전류 변화로 방전 단계 구분
        current_diff = np.diff(np.abs(currents))
        
        # DBSCAN으로 전류 클러스터링
        if len(np.abs(currents)) > 10:
            scaler = StandardScaler()
            current_scaled = scaler.fit_transform(np.abs(currents).reshape(-1, 1))
            
            clustering = DBSCAN(eps=0.3, min_samples=5).fit(current_scaled)
            labels = clustering.labels_
            
            # 클러스터별로 단계 구분
            unique_labels = set(labels)
            for label in unique_labels:
                if label != -1:  # 노이즈 제외
                    indices = np.where(labels == label)[0]
                    phases.append({
                        'type': 'CC',
                        'start_idx': int(indices.min()),
                        'end_idx': int(indices.max()),
                        'voltage_range': [voltages[indices.min()], voltages[indices.max()]],
                        'current_range': [currents[indices.min()], currents[indices.max()]],
                        'avg_current': float(np.mean(np.abs(currents[indices])))
                    })
        
        return phases
    
    def _analyze_toyo_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Toyo 데이터 패턴 분석"""
        patterns = {
            'cycles_detected': {},
            'capacity_data': [],
            'cycle_patterns': {},
            'pattern_statistics': {}
        }
        
        # CAPACITY.LOG 분석
        capacity_data = data[data['source_file'] == 'CAPACITY.LOG']
        
        if len(capacity_data) > 0:
            patterns['cycles_detected'] = {
                'total_cycles': len(capacity_data),
                'cycle_range': [1, len(capacity_data)]
            }
            
            # 용량 트렌드 분석
            if 'Cap[Ah]' in capacity_data.columns:
                capacities = capacity_data['Cap[Ah]'].dropna()
                patterns['capacity_data'] = capacities.tolist()
                
                patterns['pattern_statistics'] = {
                    'initial_capacity': float(capacities.iloc[0]) if len(capacities) > 0 else 0,
                    'final_capacity': float(capacities.iloc[-1]) if len(capacities) > 0 else 0,
                    'capacity_fade': float((capacities.iloc[0] - capacities.iloc[-1]) / capacities.iloc[0] * 100) if len(capacities) > 0 else 0,
                    'avg_capacity': float(capacities.mean())
                }
        
        return patterns
    
    def _calculate_pattern_statistics(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """패턴 통계 계산"""
        stats = {}
        
        # 충전 패턴 통계
        if patterns['charge_patterns']:
            charge_c_rates = [p.get('estimated_c_rate', 0) for p in patterns['charge_patterns']]
            charge_durations = [p.get('duration', 0) for p in patterns['charge_patterns']]
            
            stats['charge_statistics'] = {
                'avg_c_rate': np.mean(charge_c_rates),
                'max_c_rate': np.max(charge_c_rates),
                'avg_duration': np.mean(charge_durations),
                'pattern_count': len(patterns['charge_patterns'])
            }
        
        # 방전 패턴 통계
        if patterns['discharge_patterns']:
            discharge_c_rates = [p.get('estimated_c_rate', 0) for p in patterns['discharge_patterns']]
            discharge_durations = [p.get('duration', 0) for p in patterns['discharge_patterns']]
            discharge_capacities = [p.get('capacity_discharged', 0) for p in patterns['discharge_patterns']]
            
            stats['discharge_statistics'] = {
                'avg_c_rate': np.mean(discharge_c_rates),
                'max_c_rate': np.max(discharge_c_rates),
                'avg_duration': np.mean(discharge_durations),
                'avg_capacity': np.mean([c for c in discharge_capacities if c > 0]),
                'pattern_count': len(patterns['discharge_patterns'])
            }
        
        return stats
    
    def generate_processed_csv(self, data: pd.DataFrame, output_path: str):
        """전처리된 데이터를 CSV로 출력"""
        processed_data = data.copy()
        
        # 필수 컬럼만 선택하여 정리
        if self.equipment_type == 'PNE':
            essential_columns = [
                'TotalCycle', 'Current_Cycle', 'Step_type', 'Voltage[V]', 'Current[A]',
                'Chg_Capacity[Ah]', 'Dchg_Capacity[Ah]', 'Time[s]', 'source_file', 'channel'
            ]
        else:  # Toyo
            essential_columns = [
                'Cycle', 'TotlCycle', 'Mode', 'Voltage[V]', 'Current[A]', 
                'Cap[Ah]', 'PassTime[Sec]', 'source_file'
            ]
        
        # 존재하는 컬럼만 선택
        available_columns = [col for col in essential_columns if col in processed_data.columns]
        processed_data = processed_data[available_columns]
        
        # 결측치 처리
        processed_data = processed_data.dropna(subset=['Voltage[V]'] if 'Voltage[V]' in processed_data.columns else [])
        
        # CSV 저장
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        processed_data.to_csv(output_file, index=False)
        print(f"전처리된 데이터 저장: {output_file} ({len(processed_data):,} 행)")
        
        return str(output_file)
    
    def generate_analysis_report(self, patterns: Dict[str, Any], output_path: str):
        """분석 보고서 생성"""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("배터리 패턴 동적 분석 보고서")
        report_lines.append("=" * 80)
        report_lines.append(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.capacity_info:
            report_lines.append(f"데이터 경로: {self.capacity_info['path']}")
            report_lines.append(f"배터리 용량: {self.capacity_info['capacity_mah']}mAh")
            report_lines.append(f"장비 타입: {self.capacity_info['equipment_type']}")
        
        report_lines.append("")
        
        # 사이클 정보
        if 'cycles_detected' in patterns:
            cycles_info = patterns['cycles_detected']
            report_lines.append("1. 사이클 분석 결과")
            report_lines.append("-" * 40)
            report_lines.append(f"  총 사이클 수: {cycles_info.get('total_cycles', 0):,}")
            if 'cycle_range' in cycles_info:
                report_lines.append(f"  사이클 범위: {cycles_info['cycle_range'][0]} ~ {cycles_info['cycle_range'][1]}")
        
        # 충전 패턴 분석
        if patterns.get('charge_patterns'):
            report_lines.append("\n2. 충전 패턴 분석")
            report_lines.append("-" * 40)
            
            charge_stats = patterns.get('pattern_statistics', {}).get('charge_statistics', {})
            if charge_stats:
                report_lines.append(f"  평균 C-rate: {charge_stats.get('avg_c_rate', 0):.3f}C")
                report_lines.append(f"  최대 C-rate: {charge_stats.get('max_c_rate', 0):.3f}C")
                report_lines.append(f"  평균 충전 시간: {charge_stats.get('avg_duration', 0):.0f}초")
                report_lines.append(f"  충전 패턴 수: {charge_stats.get('pattern_count', 0)}")
            
            # 첫 번째 충전 패턴 예시
            if len(patterns['charge_patterns']) > 0:
                first_pattern = patterns['charge_patterns'][0]
                report_lines.append("\n  충전 패턴 예시 (사이클 1):")
                if 'voltage_start' in first_pattern and 'voltage_end' in first_pattern:
                    report_lines.append(f"    전압 범위: {first_pattern['voltage_start']:.3f}V ~ {first_pattern['voltage_end']:.3f}V")
                if 'estimated_c_rate' in first_pattern:
                    report_lines.append(f"    추정 C-rate: {first_pattern['estimated_c_rate']:.3f}C")
                if 'charge_phases' in first_pattern:
                    phases = first_pattern['charge_phases']
                    report_lines.append(f"    충전 단계: {len(phases)}개 단계 감지")
                    for i, phase in enumerate(phases):
                        report_lines.append(f"      단계 {i+1}: {phase['type']} ({phase['voltage_range'][0]:.3f}V ~ {phase['voltage_range'][1]:.3f}V)")
        
        # 방전 패턴 분석
        if patterns.get('discharge_patterns'):
            report_lines.append("\n3. 방전 패턴 분석")
            report_lines.append("-" * 40)
            
            discharge_stats = patterns.get('pattern_statistics', {}).get('discharge_statistics', {})
            if discharge_stats:
                report_lines.append(f"  평균 C-rate: {discharge_stats.get('avg_c_rate', 0):.3f}C")
                report_lines.append(f"  최대 C-rate: {discharge_stats.get('max_c_rate', 0):.3f}C")
                report_lines.append(f"  평균 방전 시간: {discharge_stats.get('avg_duration', 0):.0f}초")
                report_lines.append(f"  평균 방전 용량: {discharge_stats.get('avg_capacity', 0):.3f}Ah")
                report_lines.append(f"  방전 패턴 수: {discharge_stats.get('pattern_count', 0)}")
        
        # 용량 분석 (Toyo)
        if 'capacity_data' in patterns and patterns['capacity_data']:
            report_lines.append("\n4. 용량 분석")
            report_lines.append("-" * 40)
            
            capacity_stats = patterns.get('pattern_statistics', {})
            if capacity_stats:
                report_lines.append(f"  초기 용량: {capacity_stats.get('initial_capacity', 0):.3f}Ah")
                report_lines.append(f"  최종 용량: {capacity_stats.get('final_capacity', 0):.3f}Ah")
                report_lines.append(f"  용량 감소율: {capacity_stats.get('capacity_fade', 0):.2f}%")
                report_lines.append(f"  평균 용량: {capacity_stats.get('avg_capacity', 0):.3f}Ah")
        
        # 권장사항
        report_lines.append("\n5. 권장사항")
        report_lines.append("-" * 40)
        report_lines.append("  - 추가 사이클 데이터 분석을 통한 장기 트렌드 파악 권장")
        report_lines.append("  - 온도별, C-rate별 성능 비교 분석 권장")
        report_lines.append("  - 패턴 변화 시점 분석을 통한 배터리 상태 모니터링 권장")
        
        # 보고서 저장
        report_content = "\n".join(report_lines)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"분석 보고서 저장: {output_file}")
        
        return report_content
    
    def create_visualizations(self, data: pd.DataFrame, patterns: Dict[str, Any], output_dir: str):
        """시각화 생성"""
        self._create_visualizations(data, patterns, output_dir, multi_path=False)
    
    def _create_visualizations(self, data: pd.DataFrame, patterns: Dict[str, Any], output_dir: str, multi_path: bool = False):
        """내부 시각화 생성 메서드"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 용량 트렌드 (Toyo 데이터인 경우)
        if 'capacity_data' in patterns and patterns['capacity_data']:
            plt.figure(figsize=(14, 8))
            capacity_data = patterns['capacity_data']
            cycles = range(1, len(capacity_data) + 1)
            
            if multi_path and 'path_index' in data.columns and hasattr(self, 'path_metadata'):
                # 다중 경로의 경우 경로별로 색상 구분
                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
                toyo_data = data[data['source_file'] == 'CAPACITY.LOG'].copy()
                
                if not toyo_data.empty:
                    # 경로별로 그래프 그리기
                    for path_idx in sorted(toyo_data['path_index'].unique()):
                        path_data = toyo_data[toyo_data['path_index'] == path_idx]
                        if 'Cap[Ah]' in path_data.columns and len(path_data) > 0:
                            color = colors[path_idx % len(colors)]
                            label = f'경로 {path_idx+1}'
                            
                            # 사이클별 정렬
                            if 'Cycle' in path_data.columns:
                                path_data = path_data.sort_values('Cycle')
                                plt.plot(path_data['Cycle'], path_data['Cap[Ah]'], 
                                        'o-', color=color, linewidth=2, markersize=4, label=label)
                    
                    plt.legend()
                    plt.title('다중 경로 사이클별 용량 변화')
                else:
                    # 기본 플롯
                    plt.plot(cycles, capacity_data, 'o-', linewidth=2, markersize=4)
                    plt.title('사이클별 용량 변화')
            else:
                # 단일 경로의 경우
                plt.plot(cycles, capacity_data, 'o-', linewidth=2, markersize=4)
                plt.title('사이클별 용량 변화')
            
            plt.xlabel('Cycle Number')
            plt.ylabel('Capacity (Ah)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / '01_capacity_trend.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. 전압-전류 프로파일 (첫 번째 사이클)
        if self.equipment_type == 'PNE' and 'TotalCycle' in data.columns:
            first_cycle_data = data[data['TotalCycle'] == data['TotalCycle'].iloc[0]]
            
            if len(first_cycle_data) > 0 and 'Voltage[V]' in first_cycle_data.columns:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                
                # 시간 축 (또는 인덱스)
                if 'Time[s]' in first_cycle_data.columns:
                    time_axis = first_cycle_data['Time[s]']
                    xlabel = 'Time (s)'
                else:
                    time_axis = range(len(first_cycle_data))
                    xlabel = 'Data Point'
                
                # 전압 그래프
                ax1.plot(time_axis, first_cycle_data['Voltage[V]'], 'b-', linewidth=1)
                ax1.set_ylabel('Voltage (V)')
                ax1.set_title('첫 번째 사이클 전압-전류 프로파일')
                ax1.grid(True, alpha=0.3)
                
                # 전류 그래프
                if 'Current[A]' in first_cycle_data.columns:
                    ax2.plot(time_axis, first_cycle_data['Current[A]'], 'r-', linewidth=1)
                    ax2.set_ylabel('Current (A)')
                    ax2.set_xlabel(xlabel)
                    ax2.grid(True, alpha=0.3)
                    
                    # 충전/방전 구간 색상 구분
                    positive_mask = first_cycle_data['Current[A]'] > 0
                    negative_mask = first_cycle_data['Current[A]'] < 0
                    
                    ax2.fill_between(time_axis, first_cycle_data['Current[A]'], 
                                   where=positive_mask, alpha=0.3, color='red', label='충전')
                    ax2.fill_between(time_axis, first_cycle_data['Current[A]'], 
                                   where=negative_mask, alpha=0.3, color='blue', label='방전')
                    ax2.legend()
                
                plt.tight_layout()
                plt.savefig(output_path / '02_voltage_current_profile.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. 패턴 통계 시각화
        if 'pattern_statistics' in patterns:
            stats = patterns['pattern_statistics']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 충전 패턴 통계
            if 'charge_statistics' in stats:
                charge_stats = stats['charge_statistics']
                
                ax = axes[0, 0]
                metrics = ['Avg C-rate', 'Max C-rate', 'Avg Duration (min)']
                values = [
                    charge_stats.get('avg_c_rate', 0),
                    charge_stats.get('max_c_rate', 0),
                    charge_stats.get('avg_duration', 0) / 60  # 초를 분으로 변환
                ]
                
                bars = ax.bar(metrics, values, color=['red', 'orange', 'green'])
                ax.set_title('충전 패턴 통계')
                ax.set_ylabel('Values')
                
                # 값 표시
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.2f}', ha='center', va='bottom')
            
            # 방전 패턴 통계
            if 'discharge_statistics' in stats:
                discharge_stats = stats['discharge_statistics']
                
                ax = axes[0, 1]
                metrics = ['Avg C-rate', 'Max C-rate', 'Avg Capacity (Ah)']
                values = [
                    discharge_stats.get('avg_c_rate', 0),
                    discharge_stats.get('max_c_rate', 0),
                    discharge_stats.get('avg_capacity', 0)
                ]
                
                bars = ax.bar(metrics, values, color=['blue', 'navy', 'cyan'])
                ax.set_title('방전 패턴 통계')
                ax.set_ylabel('Values')
                
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.2f}', ha='center', va='bottom')
            
            # 나머지 subplot은 비워두거나 다른 정보 표시
            axes[1, 0].text(0.5, 0.5, '추가 분석 정보', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=16)
            axes[1, 1].text(0.5, 0.5, f'총 {patterns["cycles_detected"].get("total_cycles", 0)} 사이클 분석', 
                           ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=16)
            
            plt.tight_layout()
            plt.savefig(output_path / '03_pattern_statistics.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"시각화 파일 생성 완료: {output_path}")
    
    def _run_multi_path_analysis(self, data_paths: List[str], output_dir: str = "analysis_output") -> Dict[str, Any]:
        """
        다중 경로 분석 실행 (수명시험 중단 케이스 처리)
        
        Args:
            data_paths: 데이터 경로 리스트
            output_dir: 출력 디렉토리
            
        Returns:
            분석 결과
        """
        print(f"다중 경로 배터리 패턴 분석 시작: {len(data_paths)}개 경로")
        for i, path in enumerate(data_paths, 1):
            print(f"  {i}. {path}")
        
        # 1. 첫 번째 경로에서 배터리 정보 추출 (기준 정보로 사용)
        self.capacity_info = self.extract_capacity_from_path(data_paths[0])
        print(f"배터리 정보: {self.capacity_info}")
        
        # 2. 장비 타입 감지 (첫 번째 경로 기준)
        self.equipment_type = self.detect_equipment_type(data_paths[0])
        self.capacity_info['equipment_type'] = self.equipment_type
        print(f"감지된 장비 타입: {self.equipment_type}")
        
        # 3. 다중 경로 데이터 로드 및 채널별 연결
        print("\n다중 경로 데이터 로딩 및 연결 중...")
        combined_data = self.load_and_concatenate_multi_paths(data_paths)
        
        if len(combined_data) == 0:
            print("로드된 데이터가 없습니다.")
            return {}
        
        print(f"총 {len(combined_data):,}개의 데이터가 로드되었습니다.")
        
        # 4. 채널별 데이터 연속성 검증
        print("\n채널별 데이터 연속성 검증 중...")
        continuity_info = self.verify_channel_continuity(combined_data)
        print(f"채널별 연속성 정보:")
        for channel, info in continuity_info.items():
            print(f"  {channel}: {info['segments']}개 구간, 총 {info['total_cycles']}사이클")
        
        # 5. 패턴 분석
        print("\n패턴 분석 중...")
        patterns = self.analyze_cycle_patterns(combined_data)
        
        # 6. 결과 저장
        print("\n결과 저장 중...")
        os.makedirs(output_dir, exist_ok=True)
        
        # 처리된 데이터 저장
        processed_csv = os.path.join(output_dir, "processed_data.csv")
        essential_columns = self._get_essential_columns()
        available_columns = [col for col in essential_columns if col in combined_data.columns]
        
        if available_columns:
            combined_data[available_columns].to_csv(processed_csv, index=False)
            print(f"처리된 데이터 저장: {processed_csv} ({len(combined_data)} 행)")
        
        # 분석 리포트 생성
        report_path = os.path.join(output_dir, "analysis_report.txt")
        self._generate_multi_path_report(report_path, patterns, continuity_info)
        print(f"분석 리포트 저장: {report_path}")
        
        # 시각화 생성
        viz_dir = os.path.join(output_dir, "visualizations")
        self._create_visualizations(combined_data, patterns, viz_dir, multi_path=True)
        print(f"시각화 파일 생성 완료: {viz_dir}")
        
        print(f"\n분석 완료! 결과는 '{output_dir}' 폴더에 저장되었습니다.")
        
        # 결과 반환
        return {
            'battery_info': self.capacity_info,
            'patterns': patterns,
            'continuity_info': continuity_info,
            'data_summary': {
                'total_rows': len(combined_data),
                'total_paths': len(data_paths),
                'equipment_type': self.equipment_type,
                'channels': list(continuity_info.keys()),
                'output_files': {
                    'csv': processed_csv,
                    'report': report_path,
                    'visualizations': viz_dir
                }
            }
        }
    
    def run_analysis(self, data_path, output_dir: str = "analysis_output") -> Dict[str, Any]:
        """
        범용 배터리 패턴 분석 실행 (단일/다중 경로 자동 감지)
        
        Args:
            data_path: 데이터 경로 (문자열: 단일 경로, 리스트: 다중 경로)
            output_dir: 출력 디렉토리
            
        Returns:
            분석 결과
        """
        # 입력 타입 감지 및 적절한 분석 메서드 호출
        if isinstance(data_path, list):
            # 다중 경로 처리
            if len(data_path) == 0:
                print("[ERROR] 빈 경로 리스트입니다.")
                return {}
            elif len(data_path) == 1:
                print("[INFO] 리스트에 경로가 1개만 있어 단일 경로로 처리합니다.")
                return self._run_single_path_analysis(data_path[0], output_dir)
            else:
                print(f"[MULTI] 다중 경로 분석 모드 ({len(data_path)}개 경로)")
                return self._run_multi_path_analysis(data_path, output_dir)
        
        elif isinstance(data_path, str):
            # 단일 경로 처리
            print("[SINGLE] 단일 경로 분석 모드")
            return self._run_single_path_analysis(data_path, output_dir)
        
        else:
            print(f"[ERROR] 지원되지 않는 입력 타입: {type(data_path)}")
            print("   문자열(단일 경로) 또는 리스트(다중 경로)를 입력하세요.")
            return {}

    def run_multi_path_analysis(self, data_paths: List[str], output_dir: str = "analysis_output") -> Dict[str, Any]:
        """
        다중 경로 분석 실행 (기존 호환성을 위한 공용 메서드)
        
        Args:
            data_paths: 데이터 경로 리스트
            output_dir: 출력 디렉토리
            
        Returns:
            분석 결과
        """
        print("[INFO] run_multi_path_analysis 호출 -> 통합 run_analysis로 리다이렉트")
        return self.run_analysis(data_paths, output_dir)

    def _run_single_path_analysis(self, data_path: str, output_dir: str = "analysis_output") -> Dict[str, Any]:
        """
        단일 경로 분석 실행
        
        Args:
            data_path: 데이터 경로
            output_dir: 출력 디렉토리
            
        Returns:
            분석 결과
        """
        print(f"배터리 패턴 분석 시작: {data_path}")
        
        # 1. 경로에서 배터리 정보 추출
        self.capacity_info = self.extract_capacity_from_path(data_path)
        print(f"배터리 정보: {self.capacity_info}")
        
        # 2. 장비 타입 감지
        self.equipment_type = self.detect_equipment_type(data_path)
        self.capacity_info['equipment_type'] = self.equipment_type
        print(f"감지된 장비 타입: {self.equipment_type}")
        
        # 3. 데이터 로드 및 연결
        print("\n데이터 로딩 중...")
        combined_data = self.load_and_concatenate_data(data_path)
        
        if len(combined_data) == 0:
            print("로드된 데이터가 없습니다.")
            return {}
        
        print(f"총 {len(combined_data):,}행의 데이터가 로드되었습니다.")
        
        # 4. 패턴 분석
        print("\n패턴 분석 중...")
        patterns = self.analyze_cycle_patterns(combined_data)
        
        # 5. 결과 출력
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # CSV 출력
        csv_file = self.generate_processed_csv(
            combined_data, 
            str(output_path / "processed_data.csv")
        )
        
        # 보고서 생성
        report_content = self.generate_analysis_report(
            patterns, 
            str(output_path / "analysis_report.txt")
        )
        
        # 시각화 생성
        self.create_visualizations(combined_data, patterns, str(output_path / "visualizations"))
        
        # 결과 반환
        result = {
            'battery_info': self.capacity_info,
            'patterns': patterns,
            'data_summary': {
                'total_rows': len(combined_data),
                'equipment_type': self.equipment_type,
                'output_files': {
                    'csv': csv_file,
                    'report': str(output_path / "analysis_report.txt"),
                    'visualizations': str(output_path / "visualizations")
                }
            }
        }
        
        print(f"\n분석 완료! 결과는 '{output_path}' 폴더에 저장되었습니다.")
        
        return result

def main():
    """메인 실행 함수"""
    print("배터리 패턴 동적 분석기")
    print("=" * 50)
    
    # 사용자 입력 받기
    data_path = input("데이터 경로를 입력하세요: ").strip()
    
    if not data_path:
        print("경로가 입력되지 않았습니다. 예시 경로로 실행합니다.")
        data_path = "data/PNE_generated"
    
    # 분석 실행
    analyzer = BatteryPatternAnalyzer()
    
    try:
        result = analyzer.run_analysis(data_path)
        
        if result:
            print("\n=== 분석 요약 ===")
            print(f"배터리 용량: {result['battery_info']['capacity_mah']}mAh")
            print(f"장비 타입: {result['battery_info']['equipment_type']}")
            print(f"총 데이터 행 수: {result['data_summary']['total_rows']:,}")
            
            if 'cycles_detected' in result['patterns']:
                cycles_info = result['patterns']['cycles_detected']
                print(f"감지된 사이클 수: {cycles_info.get('total_cycles', 0)}")
            
            print(f"\n출력 파일:")
            for file_type, file_path in result['data_summary']['output_files'].items():
                print(f"  {file_type}: {file_path}")
        
    except Exception as e:
        print(f"분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
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
        self.channels = {}  # 채널 정보 저장
        self.is_multi_channel = False  # 다중 채널 여부
        
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
        
        # Manufacturer mapping
        self.manufacturer_mapping = {
            'SDI': ['SDI', 'Samsung', 'SAMSUNG', 'samsung_sdi', 'SamsungSDI', '삼성', '삼성SDI'],
            'ATL': ['ATL', 'Amperex', 'CATL', 'amperex', 'AMPEREX'],
            'LGES': ['LGES', 'LG', 'LGChem', 'LG_Energy', 'LGEnergy', 'LG화학', 'LG에너지솔루션'],
            'COSMX': ['COSMX', 'Cosmo', 'cosmo', '코스모', 'cosmos', '코스모신소재'],
            'BYD': ['BYD', 'byd', 'Build_Your_Dreams', 'Blade'],
            'Panasonic': ['Panasonic', 'PANA', 'Tesla_Panasonic', '파나소닉'],
            'SK': ['SK', 'SKI', 'SK_Innovation', 'SK이노베이션', 'SKInnovation'],
            'EVE': ['EVE', 'eve', 'EVE_Energy', 'EVEEnergy'],
            'Northvolt': ['Northvolt', 'NORTH', 'northvolt'],
            'SVOLT': ['SVOLT', 'svolt', '스볼트', 'Svolt'],
        }
    
    def extract_manufacturer_from_path(self, path: str) -> str:
        """
        경로에서 제조사 정보 추출
        
        Args:
            path: 데이터 경로
            
        Returns:
            제조사명 (표준화된 이름)
        """
        path_upper = str(path).upper()
        
        for standard_name, variations in self.manufacturer_mapping.items():
            for variation in variations:
                if variation.upper() in path_upper:
                    logger.info(f"Detected manufacturer: {standard_name} from '{variation}'")
                    return standard_name
        
        logger.warning(f"Unknown manufacturer in path: {path}")
        return "Unknown"
        
    def extract_capacity_from_path(self, data_path: str) -> Dict[str, Any]:
        """
        경로에서 용량 정보 및 기타 배터리 정보 추출
        
        Args:
            data_path: 데이터 경로 (예: "D:\\pne\\LGES_G3_MP1_4352mAh_상온수명")
            
        Returns:
            배터리 정보 딕셔너리
        """
        path_str = str(data_path).replace('/', '\\').replace('//', '\\')
        
        # 경로 존재 여부 확인
        path_obj = Path(path_str)
        if not path_obj.exists():
            logger.warning(f"경로가 존재하지 않습니다: {data_path}")
        else:
            logger.info(f"경로 확인됨: {data_path}")
        
        # 용량 정보 추출 (mAh)
        capacity_pattern = r'(\d+)mAh'
        capacity_match = re.search(capacity_pattern, path_str)
        capacity_mah = int(capacity_match.group(1)) if capacity_match else 4800
        
        # 개선된 장비 타입 판별
        equipment_type = self._determine_equipment_type_enhanced(data_path)
        
        # 기타 정보 추출
        battery_info = {
            'capacity_mah': capacity_mah,
            'capacity_ah': capacity_mah / 1000,
            'equipment_type': equipment_type,
            'path': path_str
        }
        
        # 제조사 정보 추출 (battery_analyzer_main의 로직과 통합)
        manufacturer = self.extract_manufacturer_from_path(path_str)
        battery_info['manufacturer'] = manufacturer
        
        # 추가 정보 추출 시도
        info_patterns = {
            'grade': r'(G\d+)',
            'version': r'(MP\d+)',
            'model': r'(NCM\d+|NCA|LFP|LCO)',
            'temperature': r'(상온|고온|저온|RT\d+)',
            'test_type': r'(수명|용량|성능|충방전)'
        }
        
        for key, pattern in info_patterns.items():
            match = re.search(pattern, path_str, re.IGNORECASE)
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
    
    def detect_channels(self, data_path: str) -> Dict[str, Dict]:
        """
        채널 구조 감지 및 정보 추출
        
        Args:
            data_path: 데이터 경로
            
        Returns:
            채널 정보 딕셔너리
        """
        path_obj = Path(data_path)
        channels = {}
        
        # 장비 타입 감지
        equipment_type = self.detect_equipment_type(data_path)
        
        if equipment_type == 'PNE':
            channels = self._detect_pne_channels(path_obj)
        elif equipment_type == 'Toyo':
            channels = self._detect_toyo_channels(path_obj)
        else:
            logger.warning(f"Unknown equipment type for path: {data_path}")
            return {}
        
        # 다중 채널 여부 판단
        self.is_multi_channel = len(channels) > 1
        
        logger.info(f"감지된 채널: {len(channels)}개, 장비타입: {equipment_type}")
        for ch_id, ch_info in channels.items():
            logger.info(f"  채널 {ch_id}: {ch_info['path']} ({ch_info['file_count']}개 파일)")
        
        self.channels = channels
        return channels
    
    def _detect_pne_channels(self, path_obj: Path) -> Dict[str, Dict]:
        """
        PNE 시스템 채널 감지
        
        Args:
            path_obj: 경로 객체
            
        Returns:
            PNE 채널 정보
        """
        channels = {}
        
        # M01Ch###[###] 패턴 찾기
        channel_dirs = []
        for item in path_obj.rglob('M01Ch*'):
            if item.is_dir():
                channel_dirs.append(item)
        
        for ch_dir in sorted(channel_dirs):
            # 채널 번호 추출
            match = re.search(r'M01Ch(\d+)', ch_dir.name)
            if match:
                channel_id = f"Ch{match.group(1)}"
                
                # Restore 폴더 확인
                restore_dir = ch_dir / 'Restore'
                if restore_dir.exists():
                    data_files = list(restore_dir.glob('ch*_SaveData*.csv'))
                    index_files = list(restore_dir.glob('savingFileIndex*.csv'))
                    
                    channels[channel_id] = {
                        'path': str(ch_dir),
                        'restore_path': str(restore_dir),
                        'data_files': [str(f) for f in data_files],
                        'index_files': [str(f) for f in index_files],
                        'file_count': len(data_files) + len(index_files),
                        'equipment_type': 'PNE'
                    }
                    
                    logger.debug(f"PNE 채널 감지: {channel_id} - {len(data_files)}개 데이터 파일")
        
        return channels
    
    def _detect_toyo_channels(self, path_obj: Path) -> Dict[str, Dict]:
        """
        Toyo 시스템 채널 감지 (개선된 단일/다중 채널 구분 로직)
        
        Args:
            path_obj: 경로 객체
            
        Returns:
            Toyo 채널 정보
        """
        channels = {}
        
        # 먼저 루트 레벨에 직접 데이터 파일이 있는지 확인 (단일 채널 우선 검사)
        root_measurement_files = []
        root_capacity_files = []
        numbered_dirs = []
        
        # 루트 레벨 파일들 확인
        for item in path_obj.iterdir():
            if item.is_file():
                if item.name.isdigit():
                    root_measurement_files.append(str(item))
                elif item.name == 'CAPACITY.LOG':
                    root_capacity_files.append(str(item))
            elif item.is_dir() and (item.name.isdigit() or re.match(r'^\d+$', item.name)):
                numbered_dirs.append(item)
        
        # 루트 레벨에 데이터 파일이 있으면 단일 채널로 처리
        if root_measurement_files or root_capacity_files:
            logger.debug(f"Toyo 단일 채널 감지: 루트 레벨에 {len(root_measurement_files)}개 측정파일, {len(root_capacity_files)}개 용량파일")
            
            # 하위 디렉토리의 파일들도 포함
            for file_path in path_obj.rglob('*'):
                if file_path.is_file() and file_path.parent != path_obj:
                    if file_path.name.isdigit():
                        root_measurement_files.append(str(file_path))
                    elif file_path.name == 'CAPACITY.LOG':
                        root_capacity_files.append(str(file_path))
            
            channels['Ch_Root'] = {
                'path': str(path_obj),
                'measurement_files': root_measurement_files,
                'capacity_files': root_capacity_files,
                'file_count': len(root_measurement_files) + len(root_capacity_files),
                'equipment_type': 'Toyo'
            }
            
        # 루트 레벨에 데이터 파일이 없고, 숫자 디렉토리들만 있는 경우 다중 채널로 처리
        elif numbered_dirs:
            logger.debug(f"Toyo 다중 채널 구조 감지: {len(numbered_dirs)}개 숫자 디렉토리")
            
            for item in numbered_dirs:
                channel_id = f"Ch{item.name}"
                
                # 채널 내 파일들 수집
                measurement_files = []
                capacity_files = []
                
                # 숫자 파일들 (측정 데이터)
                for file_path in item.rglob('*'):
                    if file_path.is_file():
                        if file_path.name.isdigit():
                            measurement_files.append(str(file_path))
                        elif file_path.name == 'CAPACITY.LOG':
                            capacity_files.append(str(file_path))
                
                if measurement_files or capacity_files:
                    channels[channel_id] = {
                        'path': str(item),
                        'measurement_files': measurement_files,
                        'capacity_files': capacity_files,
                        'file_count': len(measurement_files) + len(capacity_files),
                        'equipment_type': 'Toyo'
                    }
                    
                    logger.debug(f"Toyo 채널 감지: {channel_id} - {len(measurement_files)}개 측정파일, {len(capacity_files)}개 용량파일")
        
        # 아무것도 찾지 못한 경우 전체 하위 구조 검색
        if not channels:
            logger.debug("Toyo 표준 구조 미발견, 전체 하위 구조 검색 중...")
            all_measurement_files = []
            all_capacity_files = []
            
            for file_path in path_obj.rglob('*'):
                if file_path.is_file():
                    if file_path.name.isdigit():
                        all_measurement_files.append(str(file_path))
                    elif file_path.name == 'CAPACITY.LOG':
                        all_capacity_files.append(str(file_path))
            
            if all_measurement_files or all_capacity_files:
                channels['Ch_Unknown'] = {
                    'path': str(path_obj),
                    'measurement_files': all_measurement_files,
                    'capacity_files': all_capacity_files,
                    'file_count': len(all_measurement_files) + len(all_capacity_files),
                    'equipment_type': 'Toyo'
                }
                
                logger.debug(f"Toyo 비표준 구조 채널 감지: {len(all_measurement_files)}개 측정파일, {len(all_capacity_files)}개 용량파일")
        
        return channels
    
    def _determine_equipment_type_enhanced(self, data_path: str) -> str:
        """
        개선된 장비 타입 감지
        
        Args:
            data_path: 데이터 경로
            
        Returns:
            장비 타입 ('PNE' 또는 'Toyo')
        """
        path_str = str(data_path).lower()
        path_obj = Path(data_path)
        
        logger.info(f"장비 타입 감지 시작: {data_path}")
        
        # 1단계: 경로명으로 판별
        if any(keyword in path_str for keyword in ['pne', 'lges', 'samsung']):
            logger.info("경로명 기반으로 PNE 감지")
            return 'PNE'
        elif any(keyword in path_str for keyword in ['toyo', 'toyo_sic']):
            logger.info("경로명 기반으로 Toyo 감지")
            return 'Toyo'
        
        # 2단계: 실제 파일 구조 검사
        if path_obj.exists():
            if self._has_pne_structure(path_obj):
                logger.info("파일 구조 기반으로 PNE 감지")
                return 'PNE'
            elif self._has_toyo_structure(path_obj):
                logger.info("파일 구조 기반으로 Toyo 감지")
                return 'Toyo'
        else:
            logger.warning(f"경로가 존재하지 않습니다: {data_path}")
        
        # 3단계: 기본값 (PNE로 시도)
        logger.warning(f"장비 타입을 확정할 수 없어 PNE로 시도합니다: {data_path}")
        return 'PNE'
    
    def _has_pne_structure(self, path_obj: Path) -> bool:
        """
        PNE 구조 확인
        
        Args:
            path_obj: 경로 객체
            
        Returns:
            PNE 구조 여부
        """
        try:
            # PNE 특징 확인
            pne_indicators = []
            
            # M01Ch### 폴더 존재 확인
            m01ch_folders = list(path_obj.rglob('M01Ch*'))
            if m01ch_folders:
                pne_indicators.append(True)
                logger.debug(f"M01Ch 폴더 발견: {len(m01ch_folders)}개")
            
            # Restore 폴더 존재 확인
            restore_folders = list(path_obj.rglob('Restore'))
            if restore_folders:
                pne_indicators.append(True)
                logger.debug(f"Restore 폴더 발견: {len(restore_folders)}개")
            
            # SaveData 파일 존재 확인
            savedata_files = list(path_obj.rglob('*SaveData*.csv'))
            if savedata_files:
                pne_indicators.append(True)
                logger.debug(f"SaveData 파일 발견: {len(savedata_files)}개")
            
            # 최소 2개 이상의 PNE 특징이 있어야 함
            return len(pne_indicators) >= 2
            
        except Exception as e:
            logger.error(f"PNE 구조 확인 중 오류: {e}")
            return False
    
    def _has_toyo_structure(self, path_obj: Path) -> bool:
        """
        Toyo 구조 확인
        
        Args:
            path_obj: 경로 객체
            
        Returns:
            Toyo 구조 여부
        """
        try:
            # Toyo 특징 확인
            toyo_indicators = []
            
            # CAPACITY.LOG 파일 존재 확인
            capacity_logs = list(path_obj.rglob('CAPACITY.LOG'))
            if capacity_logs:
                toyo_indicators.append(True)
                logger.debug(f"CAPACITY.LOG 파일 발견: {len(capacity_logs)}개")
            
            # 숫자 파일들 존재 확인 (000001, 000002 등)
            numeric_files = [f for f in path_obj.rglob('*') if f.is_file() and f.name.isdigit()]
            if len(numeric_files) > 5:  # 최소 5개 이상
                toyo_indicators.append(True)
                logger.debug(f"숫자 파일 발견: {len(numeric_files)}개")
            
            # 6자리 숫자 파일 확인
            six_digit_files = [f for f in path_obj.rglob('*') if f.is_file() and re.match(r'^\d{6}$', f.name)]
            if six_digit_files:
                toyo_indicators.append(True)
                logger.debug(f"6자리 숫자 파일 발견: {len(six_digit_files)}개")
            
            # 최소 2개 이상의 Toyo 특징이 있어야 함
            return len(toyo_indicators) >= 2
            
        except Exception as e:
            logger.error(f"Toyo 구조 확인 중 오류: {e}")
            return False
    
    def load_and_concatenate_data(self, data_path: str) -> pd.DataFrame:
        """
        다중 채널 데이터 로드 및 파일 연결
        
        Args:
            data_path: 데이터 경로
            
        Returns:
            통합된 데이터프레임 (채널별 구분 포함)
        """
        logger.info(f"다중 채널 데이터 로드 시작: {data_path}")
        
        # 채널 감지
        channels = self.detect_channels(data_path)
        
        if not channels:
            logger.error("채널을 찾을 수 없습니다.")
            return pd.DataFrame()
        
        # 각 채널별로 데이터 로드
        all_channel_data = pd.DataFrame()
        
        for channel_id, channel_info in channels.items():
            logger.info(f"채널 {channel_id} 데이터 로드 중...")
            
            channel_data = self._load_single_channel_data(channel_id, channel_info)
            
            if not channel_data.empty:
                # 채널 정보 추가
                channel_data['channel_id'] = channel_id
                channel_data['channel_path'] = channel_info['path']
                
                # 전체 데이터에 추가
                all_channel_data = pd.concat([all_channel_data, channel_data], ignore_index=True)
                
                logger.info(f"채널 {channel_id}: {len(channel_data):,} 행 로드됨")
            else:
                logger.warning(f"채널 {channel_id}: 데이터 로드 실패")
        
        logger.info(f"전체 데이터 로드 완료: {len(channels)}개 채널, {len(all_channel_data):,} 행")
        return all_channel_data
    
    def _load_single_channel_data(self, channel_id: str, channel_info: Dict) -> pd.DataFrame:
        """
        단일 채널 데이터 로드
        
        Args:
            channel_id: 채널 ID
            channel_info: 채널 정보
            
        Returns:
            채널 데이터프레임
        """
        equipment_type = channel_info['equipment_type']
        channel_data = pd.DataFrame()
        
        try:
            if equipment_type == 'PNE':
                channel_data = self._load_pne_channel_data(channel_info)
            elif equipment_type == 'Toyo':
                channel_data = self._load_toyo_channel_data(channel_info)
            else:
                logger.error(f"알 수 없는 장비 타입: {equipment_type}")
                
        except Exception as e:
            logger.error(f"채널 {channel_id} 데이터 로드 중 오류: {e}")
            
        return channel_data
    
    def _load_pne_channel_data(self, channel_info: Dict) -> pd.DataFrame:
        """
        PNE 채널 데이터 로드
        
        Args:
            channel_info: PNE 채널 정보
            
        Returns:
            PNE 채널 데이터
        """
        channel_data = pd.DataFrame()
        
        # 데이터 파일들 로드
        for data_file in channel_info['data_files']:
            try:
                df = pd.read_csv(data_file, delimiter='\t', header=None, 
                               names=self.pne_columns, encoding='utf-8')
                
                if not df.empty:
                    df['source_file'] = Path(data_file).name
                    df['equipment_type'] = 'PNE'
                    channel_data = pd.concat([channel_data, df], ignore_index=True)
                    
                    logger.debug(f"PNE 파일 로드: {Path(data_file).name} - {len(df):,} 행")
                    
            except Exception as e:
                logger.error(f"PNE 파일 로드 실패 {data_file}: {e}")
                
        return channel_data
    
    def _load_toyo_channel_data(self, channel_info: Dict) -> pd.DataFrame:
        """
        Toyo 채널 데이터 로드
        
        Args:
            channel_info: Toyo 채널 정보
            
        Returns:
            Toyo 채널 데이터
        """
        channel_data = pd.DataFrame()
        
        # CAPACITY.LOG 파일들 로드
        for capacity_file in channel_info['capacity_files']:
            try:
                df = self._try_read_csv_multiple_separators(Path(capacity_file))
                
                if df is not None and not df.empty:
                    # 데이터 검증 및 정리
                    df = self._clean_toyo_capacity_data(df)
                    
                    df['source_file'] = 'CAPACITY.LOG'
                    df['equipment_type'] = 'Toyo'
                    channel_data = pd.concat([channel_data, df], ignore_index=True)
                    
                    logger.debug(f"Toyo CAPACITY.LOG 로드: {len(df):,} 행")
                    
            except Exception as e:
                logger.error(f"Toyo CAPACITY.LOG 로드 실패 {capacity_file}: {e}")
        
        # 측정 파일들 로드 (필요한 경우)
        measurement_count = 0
        for measurement_file in channel_info['measurement_files']:
            if measurement_count >= 10:  # 최대 10개만 로드
                break
                
            try:
                df = self._try_read_csv_multiple_separators(Path(measurement_file))
                
                if df is not None and not df.empty:
                    # 데이터 검증 및 정리
                    df = self._clean_toyo_measurement_data(df)
                    
                    df['source_file'] = Path(measurement_file).name
                    df['equipment_type'] = 'Toyo'
                    channel_data = pd.concat([channel_data, df], ignore_index=True)
                    
                    measurement_count += 1
                    logger.debug(f"Toyo 측정파일 로드: {Path(measurement_file).name} - {len(df):,} 행")
                    
            except Exception as e:
                logger.error(f"Toyo 측정파일 로드 실패 {measurement_file}: {e}")
                
        return channel_data
    
    def _clean_toyo_capacity_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Toyo CAPACITY.LOG 데이터 정리
        
        Args:
            df: 원본 데이터프레임
            
        Returns:
            정리된 데이터프레임
        """
        try:
            # 예상 컬럼들
            expected_columns = [
                'Date', 'Time', 'Condition', 'Mode', 'Cycle', 'TotlCycle', 
                'Cap[mAh]', 'PassTime', 'TotlPassTime', 'Pow[mWh]', 
                'AveVolt[V]', 'PeakVolt[V]', 'col12', 'PeakTemp[Deg]', 
                'Ocv', 'col15', 'Finish', 'DchCycle', 'PassedDate'
            ]
            
            # 컬럼 수가 예상과 다르면 조정
            if len(df.columns) > len(expected_columns):
                # 추가 컬럼들 제거
                df = df.iloc[:, :len(expected_columns)]
                logger.warning(f"CAPACITY.LOG에서 추가 컬럼 제거됨: {len(df.columns) - len(expected_columns)}개")
            elif len(df.columns) < len(expected_columns):
                # 부족한 컬럼들 채우기
                for i in range(len(df.columns), len(expected_columns)):
                    df[f'col{i}'] = np.nan
            
            # 컬럼명 설정
            df.columns = expected_columns[:len(df.columns)]
            
            # 데이터 타입 변환
            numeric_columns = ['Cycle', 'TotlCycle', 'Cap[mAh]', 'Pow[mWh]', 
                             'AveVolt[V]', 'PeakVolt[V]', 'PeakTemp[Deg]', 'Ocv']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # mAh를 Ah로 변환
            if 'Cap[mAh]' in df.columns:
                df['Chg_Capacity[Ah]'] = df['Cap[mAh]'] / 1000
                df['Dchg_Capacity[Ah]'] = df['Cap[mAh]'] / 1000
            
            return df
            
        except Exception as e:
            logger.error(f"Toyo CAPACITY 데이터 정리 중 오류: {e}")
            return df
    
    def _clean_toyo_measurement_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Toyo 측정 데이터 정리
        
        Args:
            df: 원본 데이터프레임
            
        Returns:
            정리된 데이터프레임
        """
        try:
            # 첫 번째 행이 헤더인지 확인
            if len(df) > 0:
                first_row = df.iloc[0]
                if any('Date' in str(val) or 'Time' in str(val) for val in first_row if pd.notna(val)):
                    # 첫 번째 행을 헤더로 사용
                    df.columns = df.iloc[0]
                    df = df[1:].reset_index(drop=True)
            
            # 예상 컬럼들
            expected_columns = [
                'Date', 'Time', 'PassTime[Sec]', 'Voltage[V]', 'Current[mA]',
                'col5', 'col6', 'Temp1[Deg]', 'col8', 'col9', 'col10', 
                'Condition', 'Mode', 'Cycle', 'TotlCycle', 'PassedDate', 'Temp1[Deg]'
            ]
            
            # 컬럼 수 조정
            if len(df.columns) > len(expected_columns):
                df = df.iloc[:, :len(expected_columns)]
                logger.warning(f"측정 파일에서 추가 컬럼 제거됨: {len(df.columns) - len(expected_columns)}개")
            elif len(df.columns) < len(expected_columns):
                for i in range(len(df.columns), len(expected_columns)):
                    df[f'col{i}'] = np.nan
            
            # 컬럼명 설정
            df.columns = expected_columns[:len(df.columns)]
            
            # 데이터 타입 변환
            numeric_columns = ['PassTime[Sec]', 'Voltage[V]', 'Current[mA]', 
                             'Temp1[Deg]', 'Cycle', 'TotlCycle', 'PassedDate']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 전압을 uV로 변환 (PNE와 일치)
            if 'Voltage[V]' in df.columns:
                df['Voltage[uV]'] = df['Voltage[V]'] * 1000000
            
            # 전류를 uA로 변환 (PNE와 일치)
            if 'Current[mA]' in df.columns:
                df['Current[uA]'] = df['Current[mA]'] * 1000
            
            return df
            
        except Exception as e:
            logger.error(f"Toyo 측정 데이터 정리 중 오류: {e}")
            return df
        logger.info(f"총 파일/폴더 수: {len(all_files)}")
        
        if self.equipment_type == 'PNE':
            combined_data = self._load_pne_data_enhanced(path_obj, combined_data)
                        
        elif self.equipment_type == 'Toyo':
            combined_data = self._load_toyo_data_enhanced(path_obj, combined_data)
        
        else:  # equipment_type == 'Unknown' 또는 기타
            logger.warning(f"알 수 없는 장비 타입: {self.equipment_type}")
            # Fallback: 두 방식 모두 시도
            logger.info("PNE 형식으로 먼저 시도합니다...")
            combined_data = self._load_pne_data_enhanced(path_obj, combined_data)
            
            if combined_data.empty:
                logger.info("PNE 형식 실패, Toyo 형식으로 시도합니다...")
                combined_data = self._load_toyo_data_enhanced(path_obj, combined_data)
        
        if combined_data.empty:
            logger.error("어떤 데이터도 로드되지 않았습니다")
        else:
            logger.info(f"최종 로드된 데이터: {len(combined_data):,} 행, {len(combined_data.columns)} 열")
            
        return combined_data
    
    def _load_pne_data_enhanced(self, path_obj: Path, combined_data: pd.DataFrame) -> pd.DataFrame:
        """
        향상된 PNE 데이터 로드
        
        Args:
            path_obj: 경로 객체
            combined_data: 기존 데이터프레임
            
        Returns:
            로드된 데이터프레임
        """
        logger.info("PNE 데이터 로드 시도...")
        
        # 1단계: Restore 폴더에서 SaveData 파일 찾기
        restore_folders = list(path_obj.rglob('Restore'))
        logger.info(f"Restore 폴더 발견: {len(restore_folders)}개")
        
        files_loaded = 0
        
        for restore_folder in restore_folders:
            logger.debug(f"Restore 폴더 처리 중: {restore_folder}")
            
            # SaveData 파일들 찾기
            save_data_files = sorted([f for f in restore_folder.glob("*_SaveData*.csv")])
            logger.info(f"{restore_folder.name}에서 SaveData 파일 발견: {len(save_data_files)}개")
            
            for file_path in save_data_files[:50]:  # 최대 50개 파일만 로드
                try:
                    # 다양한 구분자로 시도
                    df = self._try_read_csv_multiple_separators(file_path, self.pne_columns)
                    
                    if df is not None and not df.empty:
                        # 데이터 변환
                        self._convert_pne_data(df)
                        
                        # 파일 정보 추가
                        df['source_file'] = file_path.name
                        df['channel'] = restore_folder.parent.name
                        df['equipment_type'] = 'PNE'
                        
                        combined_data = pd.concat([combined_data, df], ignore_index=True)
                        files_loaded += 1
                        logger.info(f"로드됨: {file_path.name} - {len(df):,} 행")
                    
                except Exception as e:
                    logger.error(f"파일 로드 실패 {file_path.name}: {e}")
        
        # 2단계: Restore 폴더가 없거나 데이터가 없으면 직접 CSV 파일 찾기
        if combined_data.empty:
            logger.info("Restore 폴더에서 데이터를 찾을 수 없어 직접 CSV 파일을 찾습니다...")
            csv_files = list(path_obj.rglob('*.csv'))[:20]  # 최대 20개 파일
            logger.info(f"CSV 파일 발견: {len(csv_files)}개")
            
            for file_path in csv_files:
                try:
                    df = self._try_read_csv_multiple_separators(file_path, self.pne_columns)
                    
                    if df is not None and not df.empty and len(df.columns) >= 10:
                        self._convert_pne_data(df)
                        df['source_file'] = file_path.name
                        df['equipment_type'] = 'PNE'
                        
                        combined_data = pd.concat([combined_data, df], ignore_index=True)
                        files_loaded += 1
                        logger.info(f"직접 로드됨: {file_path.name} - {len(df):,} 행")
                        
                        if files_loaded >= 10:  # 최대 10개 파일
                            break
                
                except Exception as e:
                    logger.error(f"직접 파일 로드 실패 {file_path.name}: {e}")
        
        logger.info(f"PNE 데이터 로드 완료: {files_loaded}개 파일, {len(combined_data):,} 행")
        return combined_data
    
    def _load_toyo_data_enhanced(self, path_obj: Path, combined_data: pd.DataFrame) -> pd.DataFrame:
        """
        향상된 Toyo 데이터 로드
        
        Args:
            path_obj: 경로 객체
            combined_data: 기존 데이터프레임
            
        Returns:
            로드된 데이터프레임
        """
        logger.info("Toyo 데이터 로드 시도...")
        
        # 1단계: CAPACITY.LOG 파일 로드
        capacity_files = list(path_obj.rglob('CAPACITY.LOG'))
        logger.info(f"CAPACITY.LOG 파일 발견: {len(capacity_files)}개")
        
        files_loaded = 0
        
        for capacity_file in capacity_files:
            try:
                df = self._try_read_csv_multiple_separators(capacity_file)
                
                if df is not None and not df.empty:
                    df['source_file'] = 'CAPACITY.LOG'
                    df['equipment_type'] = 'Toyo'
                    
                    # mAh를 Ah로 변환
                    if 'Cap[mAh]' in df.columns:
                        df['Chg_Capacity[Ah]'] = df['Cap[mAh]'] / 1000
                        df['Dchg_Capacity[Ah]'] = df['Cap[mAh]'] / 1000
                    
                    combined_data = pd.concat([combined_data, df], ignore_index=True)
                    files_loaded += 1
                    logger.info(f"로드됨: CAPACITY.LOG - {len(df):,} 행")
                    
            except Exception as e:
                logger.error(f"CAPACITY.LOG 로드 실패: {e}")
        
        # 2단계: 숫자 파일들 로드 (최대 10개)
        measurement_files = sorted([f for f in path_obj.rglob('*') 
                                  if f.is_file() and f.name.isdigit()])[:10]
        logger.info(f"숫자 측정 파일 발견: {len(measurement_files)}개")
        
        for file_path in measurement_files:
            try:
                df = self._try_read_csv_multiple_separators(file_path)
                
                if df is not None and not df.empty:
                    df['source_file'] = file_path.name
                    df['equipment_type'] = 'Toyo'
                    
                    combined_data = pd.concat([combined_data, df], ignore_index=True)
                    files_loaded += 1
                    logger.info(f"로드됨: {file_path.name} - {len(df):,} 행")
                    
            except Exception as e:
                logger.error(f"숫자 파일 로드 실패 {file_path.name}: {e}")
        
        logger.info(f"Toyo 데이터 로드 완료: {files_loaded}개 파일, {len(combined_data):,} 행")
        return combined_data
    
    def _try_read_csv_multiple_separators(self, file_path: Path, columns: List[str] = None) -> Optional[pd.DataFrame]:
        """
        다양한 구분자로 CSV 파일 읽기 시도
        
        Args:
            file_path: 파일 경로
            columns: 컬럼명 리스트 (옵션)
            
        Returns:
            데이터프레임 또는 None
        """
        separators = ['\t', ',', ';', ' ']  # 시도할 구분자들
        encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']  # 시도할 인코딩들
        
        for encoding in encodings:
            for sep in separators:
                try:
                    if columns:
                        df = pd.read_csv(file_path, sep=sep, header=None, 
                                       names=columns, low_memory=False, encoding=encoding)
                    else:
                        df = pd.read_csv(file_path, sep=sep, low_memory=False, encoding=encoding)
                    
                    # 데이터가 제대로 로드되었는지 확인
                    if not df.empty and len(df.columns) >= 3:
                        logger.debug(f"성공적으로 로드됨: {file_path.name} (구분자: '{sep}', 인코딩: {encoding})")
                        return df
                        
                except Exception:
                    continue
        
        logger.warning(f"모든 구분자/인코딩 조합으로 로드 실패: {file_path.name}")
        return None
    
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
    
    def _run_multi_path_analysis(self, data_paths: List[str], output_path: str = "analysis_output") -> Dict[str, Any]:
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
        os.makedirs(output_path, exist_ok=True)
        
        # 처리된 데이터 저장
        processed_csv = os.path.join(output_path, "processed_data.csv")
        essential_columns = self._get_essential_columns()
        available_columns = [col for col in essential_columns if col in combined_data.columns]
        
        if available_columns:
            combined_data[available_columns].to_csv(processed_csv, index=False)
            print(f"처리된 데이터 저장: {processed_csv} ({len(combined_data)} 행)")
        
        # 분석 리포트 생성
        report_path = os.path.join(output_path, "analysis_report.txt")
        self._generate_multi_path_report(report_path, patterns, continuity_info)
        print(f"분석 리포트 저장: {report_path}")
        
        # 시각화 생성
        viz_dir = os.path.join(output_path, "visualizations")
        self._create_visualizations(combined_data, patterns, viz_dir, multi_path=True)
        print(f"시각화 파일 생성 완료: {viz_dir}")
        
        print(f"\n분석 완료! 결과는 '{output_path}' 폴더에 저장되었습니다.")
        
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
    
    def run_analysis(self, data_path, output_path: str = "analysis_output") -> Dict[str, Any]:
        """
        다중 채널 배터리 패턴 분석 실행
        
        Args:
            data_path: 데이터 경로
            output_path: 출력 디렉토리
            
        Returns:
            분석 결과 (채널별 구분 포함)
        """
        logger.info("=== 다중 채널 배터리 패턴 분석 시작 ===")
        
        try:
            # 채널 감지
            channels = self.detect_channels(data_path)
            
            if not channels:
                logger.error("채널을 찾을 수 없습니다.")
                return {'error': 'No channels detected'}
            
            # 전체 데이터 로드 (채널별 구분 포함)
            data = self.load_and_concatenate_data(data_path)
            
            if data.empty:
                logger.error("로드된 데이터가 없습니다.")
                return {'error': 'No data loaded'}
            
            # 결과 저장용 딕셔너리
            results = {
                'channels': {},
                'cross_channel_analysis': {},
                'summary': {
                    'total_channels': len(channels),
                    'channel_list': list(channels.keys()),
                    'is_multi_channel': self.is_multi_channel,
                    'data_shape': data.shape,
                    'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            # 각 채널별로 개별 분석
            for channel_id in channels.keys():
                logger.info(f"채널 {channel_id} 분석 시작...")
                
                channel_data = data[data['channel_id'] == channel_id].copy()
                
                if not channel_data.empty:
                    channel_result = self._analyze_single_channel(channel_id, channel_data, output_path)
                    results['channels'][channel_id] = channel_result
                    
                    logger.info(f"채널 {channel_id} 분석 완료")
                else:
                    logger.warning(f"채널 {channel_id}: 데이터 없음")
                    results['channels'][channel_id] = {'error': 'No data'}
            
            # 다중 채널인 경우 교차 분석
            if self.is_multi_channel:
                logger.info("교차 채널 분석 시작...")
                results['cross_channel_analysis'] = self._perform_cross_channel_analysis(data, channels)
                logger.info("교차 채널 분석 완료")
            
            # 전체 데이터 추가
            results['data'] = data
            
            logger.info("=== 다중 채널 배터리 패턴 분석 완료 ===")
            return results
            
        except Exception as e:
            logger.error(f"분석 중 오류 발생: {e}")
            return {'error': str(e)}
        
        else:
            logger.error(f"지원되지 않는 입력 타입: {type(data_path)}")
            return {'error': f'Unsupported input type: {type(data_path)}'}
    
    def _analyze_single_channel(self, channel_id: str, channel_data: pd.DataFrame, output_path: str) -> Dict[str, Any]:
        """
        단일 채널 분석
        
        Args:
            channel_id: 채널 ID
            channel_data: 채널 데이터
            output_path: 출력 경로
            
        Returns:
            채널 분석 결과
        """
        try:
            # 기존 분석 로직 재사용
            self.equipment_type = channel_data['equipment_type'].iloc[0] if not channel_data.empty else 'Unknown'
            
            # 용량 분석
            capacity_analysis = self._analyze_capacity(channel_data)
            
            # 패턴 분석
            if self.equipment_type == 'PNE':
                pattern_analysis = self._analyze_pne_patterns(channel_data)
            elif self.equipment_type == 'Toyo':
                pattern_analysis = self._analyze_toyo_patterns(channel_data)
            else:
                pattern_analysis = {}
            
            # 시각화 생성
            visualization_results = self._create_channel_visualizations(channel_id, channel_data, output_path)
            
            return {
                'channel_id': channel_id,
                'equipment_type': self.equipment_type,
                'data_shape': channel_data.shape,
                'capacity_analysis': capacity_analysis,
                'pattern_analysis': pattern_analysis,
                'visualizations': visualization_results,
                'data_quality': {
                    'total_rows': len(channel_data),
                    'null_count': channel_data.isnull().sum().sum(),
                    'source_files': channel_data['source_file'].unique().tolist()
                }
            }
            
        except Exception as e:
            logger.error(f"채널 {channel_id} 분석 중 오류: {e}")
            return {'error': str(e)}
    
    def _perform_cross_channel_analysis(self, data: pd.DataFrame, channels: Dict) -> Dict[str, Any]:
        """
        교차 채널 분석
        
        Args:
            data: 전체 데이터
            channels: 채널 정보
            
        Returns:
            교차 분석 결과
        """
        try:
            cross_analysis = {}
            
            # 채널별 통계 비교
            channel_stats = {}
            for channel_id in channels.keys():
                channel_data = data[data['channel_id'] == channel_id]
                
                if not channel_data.empty:
                    # 기본 통계
                    stats = {
                        'row_count': len(channel_data),
                        'equipment_type': channel_data['equipment_type'].iloc[0],
                        'source_files': channel_data['source_file'].unique().tolist()
                    }
                    
                    # 용량 정보 (Toyo의 경우)
                    if 'Cap[mAh]' in channel_data.columns:
                        cap_data = channel_data['Cap[mAh]'].dropna()
                        if not cap_data.empty:
                            stats.update({
                                'capacity_mean': cap_data.mean(),
                                'capacity_max': cap_data.max(),
                                'capacity_min': cap_data.min(),
                                'capacity_std': cap_data.std(),
                                'cycle_count': len(cap_data)
                            })
                    
                    # 전압 정보
                    voltage_cols = [col for col in channel_data.columns if 'Voltage' in col or 'Volt' in col]
                    if voltage_cols:
                        voltage_col = voltage_cols[0]
                        volt_data = pd.to_numeric(channel_data[voltage_col], errors='coerce').dropna()
                        if not volt_data.empty:
                            stats.update({
                                'voltage_mean': volt_data.mean(),
                                'voltage_max': volt_data.max(),
                                'voltage_min': volt_data.min(),
                                'voltage_std': volt_data.std()
                            })
                    
                    channel_stats[channel_id] = stats
            
            cross_analysis['channel_statistics'] = channel_stats
            
            # 채널 간 성능 비교 (용량 기준)
            if len(channel_stats) > 1:
                capacity_comparison = {}
                for ch_id, stats in channel_stats.items():
                    if 'capacity_mean' in stats:
                        capacity_comparison[ch_id] = {
                            'mean_capacity': stats['capacity_mean'],
                            'cycle_count': stats.get('cycle_count', 0),
                            'capacity_retention': stats['capacity_min'] / stats['capacity_max'] * 100 if stats.get('capacity_max', 0) > 0 else 0
                        }
                
                if capacity_comparison:
                    # 최고/최저 성능 채널 식별
                    best_channel = max(capacity_comparison.keys(), key=lambda x: capacity_comparison[x]['mean_capacity'])
                    worst_channel = min(capacity_comparison.keys(), key=lambda x: capacity_comparison[x]['mean_capacity'])
                    
                    cross_analysis['performance_comparison'] = {
                        'capacity_comparison': capacity_comparison,
                        'best_performing_channel': best_channel,
                        'worst_performing_channel': worst_channel,
                        'performance_spread': capacity_comparison[best_channel]['mean_capacity'] - capacity_comparison[worst_channel]['mean_capacity']
                    }
            
            # 데이터 품질 비교
            quality_comparison = {}
            for ch_id, stats in channel_stats.items():
                quality_comparison[ch_id] = {
                    'data_completeness': (stats['row_count'] - data[data['channel_id'] == ch_id].isnull().sum().sum()) / stats['row_count'] * 100 if stats['row_count'] > 0 else 0,
                    'file_count': len(stats['source_files'])
                }
            
            cross_analysis['data_quality_comparison'] = quality_comparison
            
            return cross_analysis
            
        except Exception as e:
            logger.error(f"교차 채널 분석 중 오류: {e}")
            return {'error': str(e)}
    
    def _create_channel_visualizations(self, channel_id: str, channel_data: pd.DataFrame, output_path: str) -> Dict[str, Any]:
        """
        채널별 시각화 생성
        
        Args:
            channel_id: 채널 ID
            channel_data: 채널 데이터
            output_path: 출력 경로
            
        Returns:
            시각화 결과
        """
        try:
            vis_results = {}
            
            # 채널별 출력 디렉토리 생성
            channel_output = Path(output_path) / f"channel_{channel_id}"
            channel_output.mkdir(parents=True, exist_ok=True)
            
            # 기본 시각화들 (기존 로직 재사용)
            if 'Cap[mAh]' in channel_data.columns:
                # 용량 트렌드 플롯
                fig, ax = plt.subplots(figsize=(12, 6))
                cap_data = channel_data['Cap[mAh]'].dropna()
                if not cap_data.empty:
                    ax.plot(range(len(cap_data)), cap_data, 'b-o', markersize=3)
                    ax.set_title(f'Channel {channel_id} - Capacity Trend')
                    ax.set_xlabel('Cycle')
                    ax.set_ylabel('Capacity (mAh)')
                    ax.grid(True, alpha=0.3)
                    
                    plot_path = channel_output / f"{channel_id}_capacity_trend.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    vis_results['capacity_trend'] = str(plot_path)
            
            # 전압 플롯
            voltage_cols = [col for col in channel_data.columns if 'Voltage' in col or 'Volt' in col]
            if voltage_cols:
                voltage_col = voltage_cols[0]
                volt_data = pd.to_numeric(channel_data[voltage_col], errors='coerce').dropna()
                
                if not volt_data.empty and len(volt_data) > 1:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(range(len(volt_data)), volt_data, 'r-', alpha=0.7)
                    ax.set_title(f'Channel {channel_id} - Voltage Profile')
                    ax.set_xlabel('Data Point')
                    ax.set_ylabel('Voltage')
                    ax.grid(True, alpha=0.3)
                    
                    plot_path = channel_output / f"{channel_id}_voltage_profile.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    vis_results['voltage_profile'] = str(plot_path)
            
            return vis_results
            
        except Exception as e:
            logger.error(f"채널 {channel_id} 시각화 생성 중 오류: {e}")
            return {'error': str(e)}

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

    def _run_single_path_analysis(self, data_path: str, output_path: str = "analysis_output") -> Dict[str, Any]:
        """
        단일 경로 분석 실행 (향상된 로깅 및 디버깅 포함)
        
        Args:
            data_path: 데이터 경로
            output_path: 출력 디렉토리
            
        Returns:
            분석 결과
        """
        logger.info("="*80)
        logger.info(f"배터리 패턴 분석 시작: {data_path}")
        logger.info("="*80)
        
        analysis_start_time = datetime.now()
        self.data_path = data_path
        
        try:
            # 1. 경로에서 배터리 정보 추출
            logger.info("1단계: 경로에서 배터리 정보 추출 중...")
            self.capacity_info = self.extract_capacity_from_path(data_path)
            logger.info(f"추출된 배터리 정보:")
            for key, value in self.capacity_info.items():
                logger.info(f"  • {key}: {value}")
            
            # 2. 장비 타입 감지 (새로운 enhanced 버전 사용)
            logger.info("2단계: 장비 타입 감지 중...")
            self.equipment_type = self.capacity_info.get('equipment_type', 'Unknown')
            logger.info(f"최종 감지된 장비 타입: {self.equipment_type}")
            
            # 디버깅 정보: 경로 구조 간략 표시
            self._log_path_structure_summary(data_path)
            
            # 3. 데이터 로드 및 연결
            logger.info("3단계: 데이터 로드 및 연결 중...")
            combined_data = self.load_and_concatenate_data(data_path)
            
            if len(combined_data) == 0:
                logger.error("❌ 로드된 데이터가 없습니다!")
                logger.error("가능한 원인:")
                logger.error("  1. 잘못된 데이터 경로")
                logger.error("  2. 지원되지 않는 데이터 형식") 
                logger.error("  3. 손상된 데이터 파일")
                logger.error("  4. 장비 타입 감지 오류")
                
                return {
                    'error': 'No data loaded',
                    'battery_info': self.capacity_info,
                    'debug_info': {
                        'equipment_type': self.equipment_type,
                        'data_path': data_path,
                        'path_exists': Path(data_path).exists()
                    }
                }
            
            logger.info(f"✅ 총 {len(combined_data):,}행의 데이터가 성공적으로 로드되었습니다.")
            logger.info(f"데이터 컬럼 수: {len(combined_data.columns)}")
            logger.info(f"주요 컬럼: {list(combined_data.columns[:10])}")
            
            # 4. 패턴 분석
            logger.info("4단계: 사이클 패턴 분석 중...")
            patterns = self.analyze_cycle_patterns(combined_data)
            logger.info(f"패턴 분석 완료: {len(patterns)} 개의 패턴 유형 감지")
            
            # 5. 결과 출력
            logger.info("5단계: 결과 파일 생성 중...")
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"출력 디렉토리: {output_path}")
            
            # CSV 출력
            csv_file = self.generate_processed_csv(
                combined_data, 
                str(output_path / "processed_data.csv")
            )
            logger.info(f"CSV 파일 생성됨: {csv_file}")
            
            # 보고서 생성
            report_file = str(output_path / "analysis_report.txt")
            report_content = self.generate_analysis_report(patterns, report_file)
            logger.info(f"분석 보고서 생성됨: {report_file}")
            
            # 시각화 생성
            viz_dir = str(output_path / "visualizations")
            self.create_visualizations(combined_data, patterns, viz_dir)
            logger.info(f"시각화 파일들 생성됨: {viz_dir}")
            
            # 분석 시간 계산
            analysis_duration = datetime.now() - analysis_start_time
            logger.info(f"총 분석 시간: {analysis_duration}")
            
            # 결과 반환
            result = {
                'battery_info': self.capacity_info,
                'patterns': patterns,
                'data': combined_data,  # 데이터도 포함
                'data_summary': {
                    'total_rows': len(combined_data),
                    'total_columns': len(combined_data.columns),
                    'equipment_type': self.equipment_type,
                    'analysis_duration': str(analysis_duration),
                    'output_files': {
                        'csv': csv_file,
                        'report': report_file,
                        'visualizations': viz_dir
                    }
                }
            }
            
            logger.info("="*80)
            logger.info("✅ 분석 완료!")
            logger.info(f"결과 저장 위치: {output_path}")
            logger.info("="*80)
            
            return result
            
        except Exception as e:
            logger.error("="*80)
            logger.error(f"❌ 분석 중 오류 발생: {str(e)}")
            logger.error("="*80)
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'error': str(e),
                'battery_info': getattr(self, 'capacity_info', {}),
                'equipment_type': getattr(self, 'equipment_type', 'Unknown'),
                'debug_info': {
                    'data_path': data_path,
                    'analysis_duration': str(datetime.now() - analysis_start_time)
                }
            }
    
    def _log_path_structure_summary(self, data_path: str):
        """
        경로 구조 요약 로그 출력
        
        Args:
            data_path: 데이터 경로
        """
        try:
            path_obj = Path(data_path)
            if not path_obj.exists():
                logger.warning(f"경로가 존재하지 않음: {data_path}")
                return
            
            # 하위 디렉토리 개수
            dirs = [p for p in path_obj.rglob('*') if p.is_dir()]
            files = [p for p in path_obj.rglob('*') if p.is_file()]
            csv_files = [p for p in path_obj.rglob('*.csv')]
            
            logger.info(f"경로 구조 요약:")
            logger.info(f"  • 총 디렉토리: {len(dirs)}개")
            logger.info(f"  • 총 파일: {len(files)}개")
            logger.info(f"  • CSV 파일: {len(csv_files)}개")
            
            # 특징적인 폴더/파일 확인
            restore_folders = [p for p in dirs if 'restore' in p.name.lower()]
            m01ch_folders = [p for p in dirs if p.name.startswith('M01Ch')]
            capacity_logs = [p for p in files if p.name == 'CAPACITY.LOG']
            numeric_files = [p for p in files if p.name.isdigit()]
            
            if restore_folders:
                logger.info(f"  • Restore 폴더: {len(restore_folders)}개")
            if m01ch_folders:
                logger.info(f"  • M01Ch 폴더: {len(m01ch_folders)}개")
            if capacity_logs:
                logger.info(f"  • CAPACITY.LOG 파일: {len(capacity_logs)}개")
            if numeric_files:
                logger.info(f"  • 숫자 파일: {len(numeric_files)}개")
                
        except Exception as e:
            logger.warning(f"경로 구조 요약 실패: {e}")

def main():
    """메인 실행 함수"""
    print("배터리 패턴 동적 분석기")
    print("=" * 50)
    
    # 사용자 입력 받기
    data_path = input("데이터 경로를 입력하세요: ").strip()
    
    if not data_path:
        print("경로가 입력되지 않았습니다. 예시 경로로 실행합니다.")
        data_path = "data/PNE_generated"
    else:
        # 백슬래시 경로 처리 (Windows 경로 정규화)
        # 사용자가 D:\MP1 같은 형태로 입력한 경우 처리
        try:
            # pathlib.Path로 경로를 정규화하고 존재 여부 확인
            normalized_path = Path(data_path)
            if not normalized_path.exists():
                # 경로가 존재하지 않으면 대안 경로들 시도
                alternative_paths = [
                    data_path.replace('\\', '/'),  # 슬래시로 변환
                    data_path.replace('/', '\\'),  # 백슬래시로 변환
                    str(Path(data_path).resolve())  # 절대경로로 변환
                ]
                
                path_found = False
                for alt_path in alternative_paths:
                    if Path(alt_path).exists():
                        data_path = alt_path
                        path_found = True
                        print(f"경로를 '{alt_path}'로 정규화했습니다.")
                        break
                
                if not path_found:
                    print(f"⚠️  경로 '{data_path}'가 존재하지 않지만 분석을 시도합니다.")
            else:
                data_path = str(normalized_path)
        except Exception as e:
            print(f"⚠️  경로 정규화 중 오류: {e}")
            print(f"원본 경로 '{data_path}'로 계속 진행합니다.")
    
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
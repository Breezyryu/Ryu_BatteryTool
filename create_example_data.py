#!/usr/bin/env python3
"""
예시 데이터 생성기
실제 배터리 테스트 환경을 시뮬레이션한 예시 데이터 생성
SiC+Graphite/LCO 시스템 수명 시험 패턴 (1200+ 사이클)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import random

# 배터리 수명 패턴 생성기 임포트
try:
    from battery_life_pattern_generator import (
        BatteryLifePatternGenerator,
        BatterySystemConfig,
        OCVModel_SiC_LCO,
        VoltageRelaxationModel,
        CapacityFadeModel,
        ChargeController,
        StepType
    )
except ImportError:
    print("경고: battery_life_pattern_generator.py를 찾을 수 없습니다.")
    print("기본 예시 데이터 생성 모드로 실행합니다.")

class ExampleDataCreator:
    """예시 데이터 생성 클래스"""
    
    def __init__(self):
        """초기화"""
        self.battery_config = BatterySystemConfig(
            anode="SiC+Graphite",
            cathode="LCO",
            capacity_mah=4352
        )
    
    def create_lges_example(self, capacity_mah: int = 4352, base_path: str = "example_data"):
        """
        LGES G3 MP1 스타일 예시 데이터 생성
        
        Args:
            capacity_mah: 배터리 용량 (mAh)
            base_path: 기본 경로
        """
        # 경로 생성
        data_path = Path(base_path) / f"LGES_G3_MP1_{capacity_mah}mAh_상온수명"
        pne_path = data_path / "M01Ch003[003]" / "Restore"
        pne_path.mkdir(parents=True, exist_ok=True)
        
        print(f"LGES 예시 데이터 생성: {data_path}")
        
        # PNE SaveData 파일들 생성
        self._create_pne_savedata_files(pne_path, capacity_mah, num_files=20)
        
        # Index 파일들 생성
        self._create_pne_index_files(pne_path, 20)
        
        print(f"PNE 예시 데이터 생성 완료: {pne_path}")
        
        return str(data_path)
    
    def create_toyo_example(self, capacity_mah: int = 4800, base_path: str = "example_data"):
        """
        Toyo 스타일 예시 데이터 생성
        
        Args:
            capacity_mah: 배터리 용량 (mAh)  
            base_path: 기본 경로
        """
        # 경로 생성
        data_path = Path(base_path) / f"Toyo_{capacity_mah}mAh_RT23"
        data_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Toyo 예시 데이터 생성: {data_path}")
        
        # CAPACITY.LOG 생성
        self._create_capacity_log(data_path, capacity_mah)
        
        # 측정 파일들 생성
        self._create_toyo_measurement_files(data_path, capacity_mah, num_files=50)
        
        print(f"Toyo 예시 데이터 생성 완료: {data_path}")
        
        return str(data_path)
    
    def _create_pne_savedata_files(self, pne_path: Path, capacity_mah: int, num_files: int = 20):
        """PNE SaveData 파일들 생성"""
        capacity_ah = capacity_mah / 1000
        base_time = datetime(2024, 7, 26, 15, 30, 0)
        
        # 46개 컬럼 정의
        columns = [
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
        
        total_cycle = 1
        accumulated_step = 1
        current_time = base_time
        
        for file_idx in range(1, num_files + 1):
            data_points = []
            
            # 파일당 약 300-400 데이터 포인트 생성
            points_per_file = random.randint(250, 450)
            
            for point_idx in range(points_per_file):
                # 사이클 패턴 시뮬레이션
                cycle_position = point_idx % 100  # 100포인트당 1사이클
                
                if cycle_position < 30:
                    # 충전 단계
                    step_type = 1  # 충전
                    voltage_base = 3.7 + (cycle_position / 30) * (4.53 - 3.7)
                    current_base = random.uniform(0.5, 2.0) * capacity_ah * 1000000  # uA
                    chg_capacity = current_base * (cycle_position * 10) / 3600000000  # uAh
                    dchg_capacity = 0
                    end_state = 65  # CC
                    
                elif cycle_position < 60:
                    # 방전 단계  
                    step_type = 2  # 방전
                    voltage_base = 4.53 - ((cycle_position - 30) / 30) * (4.53 - 3.0)
                    current_base = -random.uniform(0.3, 1.5) * capacity_ah * 1000000  # uA (음수)
                    chg_capacity = 0
                    dchg_capacity = abs(current_base) * ((cycle_position - 30) * 10) / 3600000000
                    end_state = 66  # CV 또는 방전
                    
                else:
                    # 휴지 단계
                    step_type = 3  # 휴지
                    voltage_base = 3.8 + random.uniform(-0.1, 0.1)
                    current_base = random.uniform(-1000, 1000)  # 작은 잔류 전류
                    chg_capacity = 0
                    dchg_capacity = 0
                    end_state = 64  # 휴지
                
                # 전압에 노이즈 추가
                voltage_uv = int((voltage_base + random.uniform(-0.02, 0.02)) * 1000000)
                current_ua = int(current_base + random.uniform(-1000, 1000))
                
                # 데이터 포인트 생성
                data_point = [
                    accumulated_step,  # Index
                    2,  # default(2)
                    step_type,  # Step_type
                    2 if step_type != 3 else 255,  # ChgDchg
                    2,  # Current_application
                    1 if voltage_base > 4.5 else 0,  # CCCV
                    end_state,  # EndState
                    accumulated_step,  # Step_count
                    voltage_uv,  # Voltage[uV]
                    current_ua,  # Current[uA]
                    int(chg_capacity),  # Chg_Capacity[uAh]
                    int(dchg_capacity),  # Dchg_Capacity[uAh]
                    int(abs(voltage_base * current_base / 1000000)),  # Chg_Power[mW]
                    int(abs(voltage_base * current_base / 1000000)) if current_base < 0 else 0,  # Dchg_Power[mW]
                    0,  # Chg_WattHour[Wh]
                    0,  # Dchg_WattHour[Wh]
                    0,  # repeat_pattern_count
                    point_idx * 1000 + random.randint(-100, 100),  # StepTime[1/100s]
                    0,  # TotTime[day]
                    point_idx * 1000,  # TotTime[1/100s]
                    random.randint(10000, 20000),  # Impedance
                    random.randint(22000, 25000),  # Temperature1-4
                    random.randint(22000, 25000),
                    random.randint(22000, 25000),
                    random.randint(22000, 25000),
                    0,  # col25
                    0,  # Repeat_count
                    total_cycle + (point_idx // 100),  # TotalCycle
                    total_cycle + (point_idx // 100),  # Current_Cycle
                    voltage_uv,  # Average_Voltage[uV]
                    current_ua,  # Average_Current[uA]
                    0, 0,  # col31, col32
                    0,  # CV_section
                    20240726,  # Date
                    int(1.53e8 + point_idx * 100),  # Time
                    0, 0, 0, 0,  # col35-38
                    0, 0, 0, 0,  # CC_charge, CV_section2, Discharge, col42
                    0,  # Average_voltage_section
                    accumulated_step,  # Accumulated_step
                    voltage_uv + random.randint(-1000, 1000),  # Voltage_max[uV]
                    voltage_uv - random.randint(-1000, 1000)   # Voltage_min[uV]
                ]
                
                data_points.append(data_point)
                accumulated_step += 1
            
            # DataFrame 생성 및 저장
            df = pd.DataFrame(data_points)
            filename = f"ch03_SaveData{file_idx:04d}.csv"
            file_path = pne_path / filename
            
            # 탭 구분자로 저장 (헤더 없음)
            df.to_csv(file_path, sep='\t', header=False, index=False)
            
            print(f"  생성: {filename} - {len(df)} 행")
            
            total_cycle += points_per_file // 100 + 1
    
    def _create_pne_index_files(self, pne_path: Path, num_files: int):
        """PNE 인덱스 파일들 생성"""
        # savingFileIndex_start.csv
        start_data = [[i, f"ch03_SaveData{i:04d}.csv", 0] for i in range(1, num_files + 1)]
        start_df = pd.DataFrame(start_data)
        start_file = pne_path / "savingFileIndex_start.csv"
        start_df.to_csv(start_file, header=False, index=False)
        
        # savingFileIndex_last.csv
        last_data = [[i, f"ch03_SaveData{i:04d}.csv", 1] for i in range(1, num_files + 1)]
        last_df = pd.DataFrame(last_data)
        last_file = pne_path / "savingFileIndex_last.csv"
        last_df.to_csv(last_file, header=False, index=False)
        
        # ch03_SaveEndData.csv
        end_data = pd.DataFrame([["Test Completed", datetime.now().strftime('%Y%m%d')]])
        end_file = pne_path / "ch03_SaveEndData.csv"
        end_data.to_csv(end_file, header=False, index=False)
    
    def _create_capacity_log(self, data_path: Path, capacity_mah: int):
        """CAPACITY.LOG 파일 생성"""
        capacity_data = []
        capacity_ah = capacity_mah / 1000
        base_date = datetime(2024, 1, 31, 16, 18, 0)
        
        # 200사이클 데이터 생성
        for cycle in range(1, 201):
            current_time = base_date + timedelta(hours=cycle * 4)
            
            # 용량 감소 시뮬레이션
            fade_rate = 0.0002  # 사이클당 0.02% 감소
            current_capacity = capacity_mah * (1 - fade_rate * cycle) * random.uniform(0.98, 1.02)
            
            # 충전 데이터
            charge_entry = {
                'Date': current_time.strftime('%Y/%m/%d'),
                'Time': current_time.strftime('%H:%M:%S'),
                'Condition': 1,
                'Mode': 1 if cycle % 100 == 1 else 2,  # 1=보증용량, 2=수명패턴
                'Cycle': cycle,
                'TotlCycle': cycle,
                'Cap[mAh]': current_capacity * random.uniform(1.05, 1.15),  # 오버차지
                'PassTime': f"{random.randint(3, 5):02d}:{random.randint(10, 50):02d}:{random.randint(10, 59):02d}",
                'TotlPassTime': f"{cycle * 4:02d}:{random.randint(10, 59):02d}:{random.randint(10, 59):02d}",
                'Pow[mWh]': current_capacity * 4.1 * random.uniform(0.95, 1.05),
                'AveVolt[V]': round(4.1 + random.uniform(-0.1, 0.1), 4),
                'PeakVolt[V]': round(4.53 + random.uniform(-0.02, 0.02), 4),
                'PeakTemp[Deg]': round(23.0 + random.uniform(-0.5, 1.5), 2),
                'Ocv': round(3.8 + random.uniform(-0.1, 0.1), 4),
                'Finish': 'Cur',
                'DchCycle': 0,
                'PassedDate': 0
            }
            capacity_data.append(charge_entry)
            
            # 방전 데이터
            discharge_time = current_time + timedelta(hours=2)
            discharge_entry = {
                'Date': discharge_time.strftime('%Y/%m/%d'),
                'Time': discharge_time.strftime('%H:%M:%S'),
                'Condition': 2,
                'Mode': 1 if cycle % 100 == 1 else 2,
                'Cycle': cycle,
                'TotlCycle': cycle,
                'Cap[mAh]': current_capacity,
                'PassTime': f"{random.randint(2, 4):02d}:{random.randint(10, 50):02d}:{random.randint(10, 59):02d}",
                'TotlPassTime': f"{cycle * 4 + 2:02d}:{random.randint(10, 59):02d}:{random.randint(10, 59):02d}",
                'Pow[mWh]': current_capacity * 3.9 * random.uniform(0.95, 1.05),
                'AveVolt[V]': round(3.9 + random.uniform(-0.1, 0.1), 4),
                'PeakVolt[V]': round(4.5 + random.uniform(-0.05, 0.05), 4),
                'PeakTemp[Deg]': round(23.2 + random.uniform(-0.5, 1.5), 2),
                'Ocv': round(4.2 + random.uniform(-0.1, 0.1), 4),
                'Finish': 'Vol',
                'DchCycle': 1,
                'PassedDate': 0
            }
            capacity_data.append(discharge_entry)
        
        # DataFrame 생성 및 저장
        capacity_df = pd.DataFrame(capacity_data)
        capacity_file = data_path / "CAPACITY.LOG"
        capacity_df.to_csv(capacity_file, index=False)
        
        print(f"  생성: CAPACITY.LOG - {len(capacity_df)} 행")
    
    def _create_toyo_measurement_files(self, data_path: Path, capacity_mah: int, num_files: int = 50):
        """Toyo 측정 파일들 생성"""
        capacity_ah = capacity_mah / 1000
        base_time = datetime(2024, 1, 31, 16, 18, 0)
        
        for file_idx in range(1, num_files + 1):
            current_time = base_time + timedelta(hours=file_idx * 8)
            measurements = []
            
            # 파일당 약 400-500개 측정점
            num_points = random.randint(350, 550)
            
            for i in range(num_points):
                time_offset = i * random.randint(20, 40)  # 20-40초 간격
                measurement_time = current_time + timedelta(seconds=time_offset)
                
                # 사이클 위치에 따른 전압/전류 시뮬레이션
                cycle_pos = i / num_points
                
                if cycle_pos < 0.4:  # 충전 구간
                    voltage = 3.7 + (4.53 - 3.7) * (cycle_pos / 0.4)
                    if cycle_pos < 0.3:  # CC 충전
                        current = random.uniform(1.0, 2.5) * capacity_ah * 1000  # mA
                    else:  # CV 충전
                        current = random.uniform(0.1, 1.0) * capacity_ah * 1000 * (1 - (cycle_pos - 0.3) / 0.1)
                        
                elif cycle_pos < 0.8:  # 방전 구간
                    voltage = 4.4 - (4.4 - 3.0) * ((cycle_pos - 0.4) / 0.4)
                    if cycle_pos < 0.6:  # 1단계 방전
                        current = -random.uniform(0.8, 1.2) * capacity_ah * 1000
                    else:  # 2단계 방전
                        current = -random.uniform(0.4, 0.8) * capacity_ah * 1000
                        
                else:  # 휴지 구간
                    voltage = 3.7 + random.uniform(-0.1, 0.1)
                    current = random.uniform(-10, 10)  # 잔류 전류
                
                # 노이즈 추가
                voltage += random.uniform(-0.005, 0.005)
                current += random.uniform(-50, 50)
                
                measurement = {
                    'Date': measurement_time.strftime('%Y/%m/%d'),
                    'Time': measurement_time.strftime('%H:%M:%S'),
                    'PassTime[Sec]': f"{time_offset:08d}",
                    'Voltage[V]': f"+{voltage:.4f}",
                    'Current[mA]': f"{current:.6f}",
                    'Temp1[Deg]': f"+{23.0 + random.uniform(-1.0, 2.0):.2f}",
                    'Condition': 1,
                    'Mode': 1 if file_idx <= 5 else 2,  # 처음 5개는 보증용량
                    'Cycle': file_idx,
                    'TotlCycle': file_idx,
                    'PassedDate': 0,
                    'Temp1[Deg].1': f"+{23.0 + random.uniform(-1.0, 2.0):.2f}"
                }
                
                measurements.append(measurement)
            
            # 헤더 생성 (설정 정보 + 컬럼 헤더)
            header_lines = [
                "0,0,1,0,0,0,0",  # 설정 라인
                "",
                "",
                "Date,Time,PassTime[Sec],Voltage[V],Current[mA],,,Temp1[Deg],,,,Condition,Mode,Cycle,TotlCycle,PassedDate,Temp1[Deg]"
            ]
            
            # 파일 저장
            filename = f"{file_idx:06d}"
            file_path = data_path / filename
            
            # 수동으로 헤더와 데이터 작성
            with open(file_path, 'w') as f:
                # 헤더 작성
                for line in header_lines:
                    f.write(line + '\n')
                
                # 데이터 작성
                for measurement in measurements:
                    line = ','.join([
                        measurement['Date'],
                        measurement['Time'],
                        measurement['PassTime[Sec]'],
                        measurement['Voltage[V]'],
                        measurement['Current[mA]'],
                        ',,',  # 빈 컬럼들
                        measurement['Temp1[Deg]'],
                        ',,,',  # 빈 컬럼들
                        str(measurement['Condition']),
                        str(measurement['Mode']),
                        str(measurement['Cycle']),
                        str(measurement['TotlCycle']),
                        str(measurement['PassedDate']),
                        measurement['Temp1[Deg].1']
                    ])
                    f.write(line + '\n')
            
            print(f"  생성: {filename} - {len(measurements)} 행")
    
    def create_life_pattern_pne(self, pattern_type: str = "pre_development", 
                               capacity_mah: int = 4352, base_path: str = "example_data"):
        """
        수명 시험 패턴 PNE 형식 데이터 생성
        
        Args:
            pattern_type: 'pre_development' (1200cy) or 'production' (1600cy)
            capacity_mah: 배터리 용량 (mAh)
            base_path: 기본 경로
        """
        # 경로 생성
        if pattern_type == "pre_development":
            data_path = Path(base_path) / f"LGES_SiC_LCO_{capacity_mah}mAh_선행PF_1200cy"
            total_cycles = 1200
        else:
            data_path = Path(base_path) / f"LGES_SiC_LCO_{capacity_mah}mAh_상품화_1600cy"
            total_cycles = 1600
            
        pne_path = data_path / "M01Ch003[003]" / "Restore"
        pne_path.mkdir(parents=True, exist_ok=True)
        
        print(f"수명 패턴 PNE 데이터 생성: {data_path}")
        print(f"패턴: {pattern_type}, 총 사이클: {total_cycles}")
        
        # 배터리 생성기 초기화
        config = BatterySystemConfig(capacity_mah=capacity_mah)
        generator = BatteryLifePatternGenerator(config)
        
        # 데이터 생성
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        
        if pattern_type == "pre_development":
            all_data = generator.generate_pre_development_pattern(start_time)
        else:
            all_data = generator.generate_production_pattern(start_time)
        
        # PNE 형식으로 변환 및 파일 저장
        self._save_pne_format_files(all_data, pne_path, total_cycles)
        
        print(f"PNE 수명 패턴 데이터 생성 완료: {pne_path}")
        return str(data_path)
    
    def create_life_pattern_toyo(self, pattern_type: str = "pre_development",
                                 capacity_mah: int = 4352, base_path: str = "example_data"):
        """
        수명 시험 패턴 Toyo 형식 데이터 생성
        
        Args:
            pattern_type: 'pre_development' (1200cy) or 'production' (1600cy)
            capacity_mah: 배터리 용량 (mAh)
            base_path: 기본 경로
        """
        # 경로 생성
        if pattern_type == "pre_development":
            data_path = Path(base_path) / f"Toyo_SiC_LCO_{capacity_mah}mAh_선행PF_1200cy"
            total_cycles = 1200
        else:
            data_path = Path(base_path) / f"Toyo_SiC_LCO_{capacity_mah}mAh_상품화_1600cy"
            total_cycles = 1600
            
        data_path.mkdir(parents=True, exist_ok=True)
        
        print(f"수명 패턴 Toyo 데이터 생성: {data_path}")
        print(f"패턴: {pattern_type}, 총 사이클: {total_cycles}")
        
        # 배터리 생성기 초기화
        config = BatterySystemConfig(capacity_mah=capacity_mah)
        generator = BatteryLifePatternGenerator(config)
        
        # 데이터 생성
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        
        if pattern_type == "pre_development":
            all_data = generator.generate_pre_development_pattern(start_time)
        else:
            all_data = generator.generate_production_pattern(start_time)
        
        # Toyo 형식으로 변환 및 파일 저장
        self._save_toyo_format_files(all_data, data_path, total_cycles)
        
        print(f"Toyo 수명 패턴 데이터 생성 완료: {data_path}")
        return str(data_path)
    
    def _save_pne_format_files(self, data: list, pne_path: Path, total_cycles: int):
        """PNE 형식으로 데이터 저장"""
        # 100 사이클마다 새 파일
        cycles_per_file = 100
        num_files = (total_cycles + cycles_per_file - 1) // cycles_per_file
        
        file_index = 1
        current_cycle = 0
        file_data = []
        
        for point in data:
            if point['cycle'] > current_cycle:
                current_cycle = point['cycle']
                
                # 100 사이클마다 파일 저장
                if current_cycle % cycles_per_file == 0 and file_data:
                    self._write_pne_file(pne_path, file_index, file_data)
                    file_index += 1
                    file_data = []
            
            # PNE 형식 데이터 변환
            pne_point = self._convert_to_pne_format(point)
            file_data.append(pne_point)
        
        # 마지막 파일 저장
        if file_data:
            self._write_pne_file(pne_path, file_index, file_data)
        
        # 인덱스 파일 생성
        self._create_pne_index_files(pne_path, file_index)
    
    def _convert_to_pne_format(self, point: dict) -> list:
        """데이터 포인트를 PNE 46컬럼 형식으로 변환"""
        # Step type에 따른 ChgDchg 값
        if point['step_type'] == StepType.CHARGE.value:
            chg_dchg = 2 if point.get('mode') == 'CC' else 1  # CC=2, CV=1
            end_state = 65 if point.get('mode') == 'CC' else 66
        elif point['step_type'] == StepType.DISCHARGE.value:
            chg_dchg = 2  # CC
            end_state = 65
        else:  # REST
            chg_dchg = 255
            end_state = 64
        
        # CCCV 상태
        cccv = 1 if point.get('mode') == 'CV' else 0
        
        # 전압, 전류 단위 변환
        voltage_uv = int(point['voltage'] * 1000000)
        current_ua = int(point['current'] * 1000)
        
        # 46개 컬럼 데이터
        return [
            point['step'],  # 0: Index
            2,  # 1: default(2)
            point['step_type'],  # 2: Step_type
            chg_dchg,  # 3: ChgDchg
            2,  # 4: Current_application
            cccv,  # 5: CCCV
            end_state,  # 6: EndState
            point['step'],  # 7: Step_count
            voltage_uv,  # 8: Voltage[uV]
            current_ua,  # 9: Current[uA]
            int(point.get('chg_capacity', 0)),  # 10: Chg_Capacity[uAh]
            int(point.get('dchg_capacity', 0)),  # 11: Dchg_Capacity[uAh]
            0,  # 12: Chg_Power[mW]
            0,  # 13: Dchg_Power[mW]
            0,  # 14: Chg_WattHour[Wh]
            0,  # 15: Dchg_WattHour[Wh]
            0,  # 16: repeat_pattern_count
            0,  # 17: StepTime[1/100s]
            0,  # 18: TotTime[day]
            0,  # 19: TotTime[1/100s]
            int(point.get('impedance', 0) * 1000) if 'impedance' in point else 0,  # 20: Impedance
            int(point.get('temperature', 23.0) * 1000),  # 21: Temperature1
            int(point.get('temperature', 23.0) * 1000),  # 22: Temperature2
            int(point.get('temperature', 23.0) * 1000),  # 23: Temperature3
            int(point.get('temperature', 23.0) * 1000),  # 24: Temperature4
            0,  # 25: col25
            0,  # 26: Repeat_count
            point['cycle'],  # 27: TotalCycle
            point['cycle'],  # 28: Current_Cycle
            voltage_uv,  # 29: Average_Voltage[uV]
            current_ua,  # 30: Average_Current[uA]
            0,  # 31: col31
            0,  # 32: col32
            0,  # 33: CV_section
            int(point['timestamp'].strftime('%Y%m%d')),  # 34: Date
            int(point['timestamp'].strftime('%H%M%S') + '00'),  # 35: Time
            0,  # 36: col36
            0,  # 37: col37
            0,  # 38: col38
            0,  # 39: col39
            0,  # 40: col40
            0,  # 41: col41
            0,  # 42: col42
            0,  # 43: col43
            0,  # 44: col44
            voltage_uv + 1000,  # 45: Voltage_max[uV]
        ]
    
    def _write_pne_file(self, pne_path: Path, file_index: int, data: list):
        """PNE 파일 작성"""
        filename = f"ch03_SaveData{file_index:04d}.csv"
        file_path = pne_path / filename
        
        # DataFrame 생성 및 저장 (탭 구분자, 헤더 없음)
        df = pd.DataFrame(data)
        df.to_csv(file_path, sep='\t', header=False, index=False)
        
        print(f"  생성: {filename} - {len(df)} 행")
    
    def _save_toyo_format_files(self, data: list, data_path: Path, total_cycles: int):
        """Toyo 형식으로 데이터 저장"""
        # CAPACITY.LOG 생성
        self._create_capacity_log_from_data(data, data_path)
        
        # 사이클별 측정 파일 생성
        cycles_per_file = 10  # 10 사이클마다 하나의 파일
        file_index = 1
        current_cycle = 0
        file_data = []
        
        for point in data:
            if point['cycle'] > current_cycle:
                current_cycle = point['cycle']
                
                # 10 사이클마다 파일 저장
                if current_cycle % cycles_per_file == 0 and file_data:
                    self._write_toyo_file(data_path, file_index, file_data)
                    file_index += 1
                    file_data = []
            
            file_data.append(point)
        
        # 마지막 파일 저장
        if file_data:
            self._write_toyo_file(data_path, file_index, file_data)
    
    def _create_capacity_log_from_data(self, data: list, data_path: Path):
        """실제 데이터로부터 CAPACITY.LOG 생성"""
        capacity_entries = []
        current_cycle = 0
        cycle_data = {'charge': None, 'discharge': None}
        
        for point in data:
            if point['cycle'] > current_cycle:
                # 이전 사이클 데이터 저장
                if current_cycle > 0 and cycle_data['charge'] and cycle_data['discharge']:
                    # 충전 엔트리
                    capacity_entries.append({
                        'Date': cycle_data['charge']['timestamp'].strftime('%Y/%m/%d'),
                        'Time': cycle_data['charge']['timestamp'].strftime('%H:%M:%S'),
                        'Condition': 1,
                        'Mode': 1 if current_cycle % 100 == 1 else 2,
                        'Cycle': current_cycle,
                        'TotlCycle': current_cycle,
                        'Cap[mAh]': cycle_data['charge']['capacity'],
                        'PassTime': '04:00:00',
                        'TotlPassTime': f"{current_cycle * 8:04d}:00:00",
                        'Pow[mWh]': cycle_data['charge']['capacity'] * 4.1,
                        'AveVolt[V]': 4.1,
                        'PeakVolt[V]': 4.53,
                        'PeakTemp[Deg]': 23.5,
                        'Ocv': OCVModel_SiC_LCO.get_ocv(100),
                        'Finish': 'Cur',
                        'DchCycle': 0,
                        'PassedDate': 0
                    })
                    
                    # 방전 엔트리
                    capacity_entries.append({
                        'Date': cycle_data['discharge']['timestamp'].strftime('%Y/%m/%d'),
                        'Time': cycle_data['discharge']['timestamp'].strftime('%H:%M:%S'),
                        'Condition': 2,
                        'Mode': 1 if current_cycle % 100 == 1 else 2,
                        'Cycle': current_cycle,
                        'TotlCycle': current_cycle,
                        'Cap[mAh]': cycle_data['discharge']['capacity'],
                        'PassTime': '03:00:00',
                        'TotlPassTime': f"{current_cycle * 8 + 4:04d}:00:00",
                        'Pow[mWh]': cycle_data['discharge']['capacity'] * 3.7,
                        'AveVolt[V]': 3.7,
                        'PeakVolt[V]': 4.2,
                        'PeakTemp[Deg]': 24.0,
                        'Ocv': OCVModel_SiC_LCO.get_ocv(0),
                        'Finish': 'Vol',
                        'DchCycle': 1,
                        'PassedDate': 0
                    })
                
                current_cycle = point['cycle']
                cycle_data = {'charge': None, 'discharge': None}
            
            # 충전/방전 용량 수집
            if point['step_type'] == StepType.CHARGE.value and not cycle_data['charge']:
                cycle_data['charge'] = {
                    'timestamp': point['timestamp'],
                    'capacity': self.battery_config.capacity_mah * CapacityFadeModel.get_capacity_retention(current_cycle)
                }
            elif point['step_type'] == StepType.DISCHARGE.value and not cycle_data['discharge']:
                cycle_data['discharge'] = {
                    'timestamp': point['timestamp'],
                    'capacity': self.battery_config.capacity_mah * CapacityFadeModel.get_capacity_retention(current_cycle)
                }
        
        # DataFrame 생성 및 저장
        if capacity_entries:
            capacity_df = pd.DataFrame(capacity_entries)
            capacity_file = data_path / "CAPACITY.LOG"
            capacity_df.to_csv(capacity_file, index=False)
            print(f"  생성: CAPACITY.LOG - {len(capacity_df)} 행")
    
    def _write_toyo_file(self, data_path: Path, file_index: int, data: list):
        """Toyo 측정 파일 작성"""
        filename = f"{file_index:06d}"
        file_path = data_path / filename
        
        # 헤더 생성
        header_lines = [
            "0,0,1,0,0,0,0",
            "",
            "",
            "Date,Time,PassTime[Sec],Voltage[V],Current[mA],,,Temp1[Deg],,,,Condition,Mode,Cycle,TotlCycle,PassedDate,Temp1[Deg]"
        ]
        
        # 데이터 변환
        measurements = []
        for point in data:
            measurement = {
                'Date': point['timestamp'].strftime('%Y/%m/%d'),
                'Time': point['timestamp'].strftime('%H:%M:%S'),
                'PassTime[Sec]': '00000000',
                'Voltage[V]': f"+{point['voltage']:.4f}",
                'Current[mA]': f"{point['current']:.6f}",
                'Temp1[Deg]': f"+{point.get('temperature', 23.0):.2f}",
                'Condition': 1 if point['step_type'] == StepType.CHARGE.value else 2,
                'Mode': 1 if point['cycle'] % 100 == 1 else 2,
                'Cycle': point['cycle'],
                'TotlCycle': point['cycle'],
                'PassedDate': 0,
                'Temp1[Deg].1': f"+{point.get('temperature', 23.0):.2f}"
            }
            measurements.append(measurement)
        
        # 파일 작성
        with open(file_path, 'w') as f:
            # 헤더 작성
            for line in header_lines:
                f.write(line + '\n')
            
            # 데이터 작성
            for measurement in measurements:
                line = ','.join([
                    measurement['Date'],
                    measurement['Time'],
                    measurement['PassTime[Sec]'],
                    measurement['Voltage[V]'],
                    measurement['Current[mA]'],
                    ',,',
                    measurement['Temp1[Deg]'],
                    ',,,',
                    str(measurement['Condition']),
                    str(measurement['Mode']),
                    str(measurement['Cycle']),
                    str(measurement['TotlCycle']),
                    str(measurement['PassedDate']),
                    measurement['Temp1[Deg].1']
                ])
                f.write(line + '\n')
        
        print(f"  생성: {filename} - {len(measurements)} 행")
    
    def create_all_examples(self, base_path: str = "example_data"):
        """모든 예시 데이터 생성"""
        print("=== 배터리 테스트 예시 데이터 생성 ===")
        
        # 1. LGES PNE 예시 (4352mAh)
        lges_path = self.create_lges_example(4352, base_path)
        
        # 2. Toyo 예시 (4800mAh)  
        toyo_path = self.create_toyo_example(4800, base_path)
        
        # 3. 다른 용량 예시들
        print("\n추가 예시 데이터 생성...")
        
        # 3500mAh PNE 예시
        small_pne_path = self.create_lges_example(3500, base_path)
        
        # 5200mAh Toyo 예시
        large_toyo_path = self.create_toyo_example(5200, base_path)
        
        print(f"\n=== 예시 데이터 생성 완료 ===")
        print(f"생성된 경로들:")
        print(f"  - {lges_path}")
        print(f"  - {toyo_path}")
        print(f"  - {small_pne_path}")
        print(f"  - {large_toyo_path}")
        
        return [lges_path, toyo_path, small_pne_path, large_toyo_path]

def main():
    """메인 실행 함수"""
    creator = ExampleDataCreator()
    
    print("=" * 60)
    print("배터리 테스트 데이터 생성기")
    print("SiC+Graphite/LCO 시스템")
    print("=" * 60)
    
    print("\n데이터 타입 선택:")
    print("1. 간단한 예시 데이터 (LGES PNE)")
    print("2. 간단한 예시 데이터 (Toyo)")
    print("3. 모든 간단한 예시")
    print("4. 수명 패턴 데이터 - PNE 형식 (1200/1600 사이클)")
    print("5. 수명 패턴 데이터 - Toyo 형식 (1200/1600 사이클)")
    print("6. 수명 패턴 데이터 - 모든 형식")
    
    choice = input("\n선택 (1-6): ").strip()
    
    if choice == '1':
        capacity = input("배터리 용량 (mAh, 기본값: 4352): ").strip()
        capacity = int(capacity) if capacity.isdigit() else 4352
        path = creator.create_lges_example(capacity)
        print(f"\n생성 완료: {path}")
        
    elif choice == '2':
        capacity = input("배터리 용량 (mAh, 기본값: 4800): ").strip()
        capacity = int(capacity) if capacity.isdigit() else 4800
        path = creator.create_toyo_example(capacity)
        print(f"\n생성 완료: {path}")
        
    elif choice == '3':
        paths = creator.create_all_examples()
        print(f"\n모든 예시 생성 완료!")
        
    elif choice == '4':
        print("\n수명 패턴 선택:")
        print("1. 선행PF (1200 사이클)")
        print("2. 상품화 (1600 사이클)")
        pattern = input("패턴 선택 (1/2): ").strip()
        pattern_type = "pre_development" if pattern == '1' else "production"
        
        capacity = input("배터리 용량 (mAh, 기본값: 4352): ").strip()
        capacity = int(capacity) if capacity.isdigit() else 4352
        
        path = creator.create_life_pattern_pne(pattern_type, capacity)
        print(f"\n생성 완료: {path}")
        
    elif choice == '5':
        print("\n수명 패턴 선택:")
        print("1. 선행PF (1200 사이클)")
        print("2. 상품화 (1600 사이클)")
        pattern = input("패턴 선택 (1/2): ").strip()
        pattern_type = "pre_development" if pattern == '1' else "production"
        
        capacity = input("배터리 용량 (mAh, 기본값: 4352): ").strip()
        capacity = int(capacity) if capacity.isdigit() else 4352
        
        path = creator.create_life_pattern_toyo(pattern_type, capacity)
        print(f"\n생성 완료: {path}")
        
    elif choice == '6':
        print("\n수명 패턴 선택:")
        print("1. 선행PF (1200 사이클)")
        print("2. 상품화 (1600 사이클)")
        pattern = input("패턴 선택 (1/2): ").strip()
        pattern_type = "pre_development" if pattern == '1' else "production"
        
        capacity = input("배터리 용량 (mAh, 기본값: 4352): ").strip()
        capacity = int(capacity) if capacity.isdigit() else 4352
        
        print("\n모든 형식으로 생성 중...")
        pne_path = creator.create_life_pattern_pne(pattern_type, capacity)
        toyo_path = creator.create_life_pattern_toyo(pattern_type, capacity, "example_data_toyo")
        
        print(f"\n생성 완료:")
        print(f"  - PNE: {pne_path}")
        print(f"  - Toyo: {toyo_path}")
        
    else:
        print("잘못된 선택입니다.")

if __name__ == "__main__":
    main()
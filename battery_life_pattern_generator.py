#!/usr/bin/env python3
"""
배터리 수명 시험 패턴 생성기
SiC+Graphite 블렌드 음극 / LCO 양극 시스템
실제 수명 시험 패턴 (1200+ 사이클) 데이터 생성
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

# ===== 배터리 시스템 정의 =====

class ChargeMode(Enum):
    """충전 모드"""
    CC = "CC"      # Constant Current
    CV = "CV"      # Constant Voltage
    CCCV = "CCCV"  # CC-CV
    REST = "REST"  # 휴지

class StepType(Enum):
    """스텝 타입 (PNE 형식)"""
    CHARGE = 1      # 충전
    DISCHARGE = 2   # 방전
    REST = 3        # 휴지
    OCV = 4         # 개방회로전압
    IMPEDANCE = 5   # 임피던스
    LOOP = 8        # 루프

@dataclass
class BatterySystemConfig:
    """배터리 시스템 구성"""
    anode: str = "SiC+Graphite"
    cathode: str = "LCO"
    nominal_voltage: float = 3.7  # V
    max_voltage: float = 4.53     # V
    min_voltage: float = 3.0      # V
    capacity_mah: int = 4352      # mAh

# ===== OCV-SOC 모델 (SiC+Graphite/LCO) =====

class OCVModel_SiC_LCO:
    """SiC+Graphite 블렌드 음극 / LCO 양극 OCV 모델"""
    
    # SiC+Graphite/LCO 시스템 OCV-SOC 테이블
    OCV_SOC_TABLE = [
        (0,   2.80),  # 완전 방전 (SiC 영향)
        (5,   3.20),  # SiC 플래토 시작
        (10,  3.35),  # SiC+Graphite 혼합
        (15,  3.45),  
        (20,  3.52),  # Graphite 주도 시작
        (25,  3.58),
        (30,  3.62),  # LCO 플래토
        (35,  3.66),
        (40,  3.70),
        (45,  3.74),
        (50,  3.78),  # 중간 SOC
        (55,  3.82),
        (60,  3.86),
        (65,  3.91),
        (70,  3.96),  # LCO 상부 플래토
        (75,  4.01),
        (80,  4.06),
        (85,  4.12),
        (90,  4.18),  # 급격한 상승
        (95,  4.25),
        (100, 4.35)   # 완충 (LCO)
    ]
    
    @classmethod
    def get_ocv(cls, soc_percent: float) -> float:
        """SOC로부터 OCV 계산 (선형 보간)"""
        if soc_percent <= 0:
            return cls.OCV_SOC_TABLE[0][1]
        if soc_percent >= 100:
            return cls.OCV_SOC_TABLE[-1][1]
        
        # 선형 보간
        for i in range(len(cls.OCV_SOC_TABLE) - 1):
            soc1, ocv1 = cls.OCV_SOC_TABLE[i]
            soc2, ocv2 = cls.OCV_SOC_TABLE[i + 1]
            if soc1 <= soc_percent <= soc2:
                ratio = (soc_percent - soc1) / (soc2 - soc1)
                return ocv1 + ratio * (ocv2 - ocv1)
        
        return cls.OCV_SOC_TABLE[-1][1]
    
    @classmethod
    def get_soc(cls, ocv: float) -> float:
        """OCV로부터 SOC 계산 (역 보간)"""
        if ocv <= cls.OCV_SOC_TABLE[0][1]:
            return cls.OCV_SOC_TABLE[0][0]
        if ocv >= cls.OCV_SOC_TABLE[-1][1]:
            return cls.OCV_SOC_TABLE[-1][0]
        
        # 역 선형 보간
        for i in range(len(cls.OCV_SOC_TABLE) - 1):
            soc1, ocv1 = cls.OCV_SOC_TABLE[i]
            soc2, ocv2 = cls.OCV_SOC_TABLE[i + 1]
            if ocv1 <= ocv <= ocv2:
                ratio = (ocv - ocv1) / (ocv2 - ocv1)
                return soc1 + ratio * (soc2 - soc1)
        
        return 50.0  # 기본값

# ===== 전압 이완/회복 모델 =====

class VoltageRelaxationModel:
    """SiC 특성을 반영한 전압 이완/회복 모델"""
    
    @staticmethod
    def voltage_after_charge(t: float, v_end: float, soc: float) -> float:
        """
        충전 후 전압 이완 (SiC 특성 반영)
        t: 휴지 시작 후 경과 시간 (초)
        v_end: 충전 종료 시 전압
        soc: 현재 SOC (%)
        """
        v_ocv = OCVModel_SiC_LCO.get_ocv(soc)
        delta_v = v_end - v_ocv
        
        # SiC 영향으로 2단계 이완
        if t < 60:  # 초기 빠른 이완
            return v_ocv + delta_v * np.exp(-t / 30)
        else:  # 느린 이완 (SiC 영향)
            fast_component = delta_v * 0.3 * np.exp(-60 / 30)
            slow_component = delta_v * 0.7 * np.exp(-(t - 60) / 150)
            return v_ocv + fast_component + slow_component
    
    @staticmethod
    def voltage_after_discharge(t: float, v_end: float, soc: float) -> float:
        """
        방전 후 전압 회복 (SiC 특성 반영)
        t: 휴지 시작 후 경과 시간 (초)
        v_end: 방전 종료 시 전압
        soc: 현재 SOC (%)
        """
        v_ocv = OCVModel_SiC_LCO.get_ocv(soc)
        delta_v = v_ocv - v_end
        
        # SiC 영향으로 2단계 회복
        if t < 60:  # 초기 빠른 회복
            return v_end + delta_v * 0.7 * (1 - np.exp(-t / 30))
        else:  # 느린 회복 (SiC 영향)
            base = v_end + delta_v * 0.7
            slow_recovery = delta_v * 0.3 * (1 - np.exp(-(t - 60) / 150))
            return base + slow_recovery

# ===== 용량 감소 모델 =====

class CapacityFadeModel:
    """SiC 특성을 반영한 용량 감소 모델"""
    
    @staticmethod
    def get_capacity_retention(cycle_number: int) -> float:
        """
        사이클별 용량 유지율 계산
        SiC SEI 형성으로 초기 빠른 감소
        """
        if cycle_number <= 100:
            # 초기 빠른 감소 (SiC SEI 형성)
            return 1.0 - 0.0005 * cycle_number
        elif cycle_number <= 500:
            # 중간 단계 안정화
            base = 0.95  # 100 사이클 후 용량
            return base - 0.0001 * (cycle_number - 100)
        else:
            # 장기 안정 단계
            base = 0.91  # 500 사이클 후 용량
            return base - 0.00005 * (cycle_number - 500)

# ===== 전압 보호 알고리즘 =====

class VoltageProtectionAlgorithm:
    """SiC 음극 보호를 위한 전압 조정"""
    
    # 사이클별 전압 조정값
    PROTECTION_SCHEDULE = [
        (201,  -0.01, +0.05),  # (사이클, 상한 조정, 하한 조정)
        (251,  -0.02, +0.10),
        (301,  -0.03, +0.15),
        (501,  -0.04, +0.20),
        (701,  -0.05, +0.25),
        (1001, -0.06, +0.30),
    ]
    
    @classmethod
    def get_voltage_limits(cls, cycle: int, base_max: float, base_min: float) -> Tuple[float, float]:
        """현재 사이클에서의 전압 제한값 계산"""
        max_adjust = 0.0
        min_adjust = 0.0
        
        for cycle_threshold, max_adj, min_adj in cls.PROTECTION_SCHEDULE:
            if cycle >= cycle_threshold:
                max_adjust = max_adj
                min_adjust = min_adj
        
        return base_max + max_adjust, base_min + min_adjust

# ===== 충전 컨트롤러 =====

class ChargeController:
    """CC/CV 충전 제어 및 Cut-off 조건 관리"""
    
    # 보증용량 측정 충전
    WARRANTY_CHARGE = {
        'c_rate': 0.2,
        'voltage': 4.53,
        'cutoff_c_rate': 0.02,
        'mode': ChargeMode.CCCV
    }
    
    # 멀티스텝 충전 (수명패턴)
    MULTI_STEP_CHARGE = [
        {'step': 1, 'c_rate': 2.0, 'voltage': 4.14, 'mode': ChargeMode.CC},
        {'step': 2, 'c_rate': 1.65, 'voltage': 4.16, 'mode': ChargeMode.CC},
        {'step': 3, 'c_rate': 1.4, 'voltage': 4.30, 'mode': ChargeMode.CCCV, 'cutoff': 0.14},
        {'step': 4, 'c_rate': 1.0, 'voltage': 4.53, 'mode': ChargeMode.CCCV, 'cutoff': 0.1}
    ]
    
    # 2단계 방전
    TWO_STEP_DISCHARGE = [
        {'c_rate': 1.0, 'voltage': 3.6},
        {'c_rate': 0.5, 'voltage': 3.2}
    ]
    
    # RSS 측정
    RSS_CHARGE = {
        'c_rate': 0.1,
        'voltage': 4.53,
        'cutoff_c_rate': 0.01,
        'mode': ChargeMode.CCCV
    }
    
    @staticmethod
    def simulate_cv_current(initial_current: float, time_in_cv: float, tau: float = 300) -> float:
        """CV 충전 중 전류 감소 시뮬레이션"""
        return initial_current * np.exp(-time_in_cv / tau)

# ===== 배터리 수명 패턴 생성기 =====

class BatteryLifePatternGenerator:
    """실제 수명 시험 패턴 생성기"""
    
    def __init__(self, config: BatterySystemConfig):
        self.config = config
        self.capacity_ah = config.capacity_mah / 1000.0
        self.current_cycle = 0
        self.total_cycle = 0
        self.accumulated_step = 0
        self.current_soc = 50.0  # 초기 SOC
        
    def generate_warranty_cycle(self, start_time: datetime) -> List[Dict]:
        """보증용량 측정 사이클 생성"""
        data_points = []
        current_time = start_time
        
        # 충전 (0.2C CCCV)
        charge_points = self._generate_cccv_charge(
            current_time,
            c_rate=0.2,
            target_voltage=4.53,
            cutoff_c_rate=0.02,
            initial_soc=self.current_soc
        )
        data_points.extend(charge_points)
        current_time = start_time + timedelta(seconds=len(charge_points) * 10)
        self.current_soc = 100.0
        
        # 휴지 10분
        rest_points = self._generate_rest(
            current_time,
            duration=600,
            after_charge=True,
            soc=self.current_soc
        )
        data_points.extend(rest_points)
        current_time = current_time + timedelta(seconds=600)
        
        # 방전 (0.2C CC)
        discharge_points = self._generate_cc_discharge(
            current_time,
            c_rate=0.2,
            cutoff_voltage=3.0,
            initial_soc=self.current_soc
        )
        data_points.extend(discharge_points)
        current_time = current_time + timedelta(seconds=len(discharge_points) * 10)
        self.current_soc = 0.0
        
        # 휴지 10분
        rest_points = self._generate_rest(
            current_time,
            duration=600,
            after_charge=False,
            soc=self.current_soc
        )
        data_points.extend(rest_points)
        
        return data_points
    
    def generate_life_cycle(self, start_time: datetime) -> List[Dict]:
        """수명패턴 사이클 생성"""
        data_points = []
        current_time = start_time
        
        # 멀티스텝 충전
        for step_config in ChargeController.MULTI_STEP_CHARGE:
            if step_config['mode'] == ChargeMode.CC:
                # CC 충전만
                step_points = self._generate_cc_charge(
                    current_time,
                    c_rate=step_config['c_rate'],
                    target_voltage=step_config['voltage'],
                    initial_soc=self.current_soc
                )
            else:  # CCCV
                step_points = self._generate_cccv_charge(
                    current_time,
                    c_rate=step_config['c_rate'],
                    target_voltage=step_config['voltage'],
                    cutoff_c_rate=step_config['cutoff'],
                    initial_soc=self.current_soc
                )
            
            data_points.extend(step_points)
            current_time = current_time + timedelta(seconds=len(step_points) * 10)
            
            # SOC 업데이트
            charge_capacity = sum([p['chg_capacity'] for p in step_points])
            self.current_soc = min(100.0, self.current_soc + (charge_capacity / (self.config.capacity_mah * 1000)) * 100)
        
        # 휴지 10분
        rest_points = self._generate_rest(
            current_time,
            duration=600,
            after_charge=True,
            soc=self.current_soc
        )
        data_points.extend(rest_points)
        current_time = current_time + timedelta(seconds=600)
        
        # 2단계 방전
        for discharge_config in ChargeController.TWO_STEP_DISCHARGE:
            step_points = self._generate_cc_discharge(
                current_time,
                c_rate=discharge_config['c_rate'],
                cutoff_voltage=discharge_config['voltage'],
                initial_soc=self.current_soc
            )
            data_points.extend(step_points)
            current_time = current_time + timedelta(seconds=len(step_points) * 10)
            
            # SOC 업데이트
            discharge_capacity = sum([p['dchg_capacity'] for p in step_points])
            self.current_soc = max(0.0, self.current_soc - (discharge_capacity / (self.config.capacity_mah * 1000)) * 100)
        
        # 휴지 10분
        rest_points = self._generate_rest(
            current_time,
            duration=600,
            after_charge=False,
            soc=self.current_soc
        )
        data_points.extend(rest_points)
        
        return data_points
    
    def generate_rss_cycle(self, start_time: datetime) -> List[Dict]:
        """RSS 패턴 사이클 생성"""
        data_points = []
        current_time = start_time
        
        # 0.1C 충전
        charge_points = self._generate_cccv_charge(
            current_time,
            c_rate=0.1,
            target_voltage=4.53,
            cutoff_c_rate=0.01,
            initial_soc=0.0
        )
        data_points.extend(charge_points)
        current_time = current_time + timedelta(seconds=len(charge_points) * 10)
        
        # 0.1C 방전
        discharge_points = self._generate_cc_discharge(
            current_time,
            c_rate=0.1,
            cutoff_voltage=3.0,
            initial_soc=100.0
        )
        data_points.extend(discharge_points)
        current_time = current_time + timedelta(seconds=len(discharge_points) * 10)
        
        # 휴지 10분
        rest_points = self._generate_rest(current_time, 600, False, 0.0)
        data_points.extend(rest_points)
        current_time = current_time + timedelta(seconds=600)
        
        # SOC 30%, 50%, 70% 구간 RSS 측정
        for target_soc in [30, 50, 70]:
            # 충전으로 목표 SOC 도달
            charge_to_soc = self._charge_to_soc(current_time, target_soc)
            data_points.extend(charge_to_soc)
            current_time = current_time + timedelta(seconds=len(charge_to_soc) * 10)
            
            # RSS 측정 (임피던스)
            rss_point = self._measure_rss(current_time, target_soc)
            data_points.append(rss_point)
            current_time = current_time + timedelta(seconds=10)
        
        return data_points
    
    def _generate_cc_charge(self, start_time: datetime, c_rate: float, 
                           target_voltage: float, initial_soc: float) -> List[Dict]:
        """CC 충전 데이터 생성"""
        data_points = []
        current_soc = initial_soc
        current_voltage = OCVModel_SiC_LCO.get_ocv(current_soc)
        
        # 전압 보호 적용
        max_voltage, _ = VoltageProtectionAlgorithm.get_voltage_limits(
            self.total_cycle, target_voltage, self.config.min_voltage
        )
        target_voltage = min(target_voltage, max_voltage)
        
        time_step = 10  # 10초 간격
        current_ma = c_rate * self.config.capacity_mah
        
        while current_voltage < target_voltage and current_soc < 100:
            self.accumulated_step += 1
            
            # 전압 상승 시뮬레이션
            charge_amount = (current_ma * time_step) / 3600  # mAh
            soc_increase = (charge_amount / self.config.capacity_mah) * 100
            current_soc = min(100, current_soc + soc_increase)
            
            # 전압 계산 (IR 드롭 포함)
            ocv = OCVModel_SiC_LCO.get_ocv(current_soc)
            ir_drop = 0.05 * c_rate  # IR 드롭
            current_voltage = ocv + ir_drop + random.uniform(-0.005, 0.005)
            
            data_point = {
                'step': self.accumulated_step,
                'step_type': StepType.CHARGE.value,
                'voltage': current_voltage,
                'current': current_ma,
                'chg_capacity': charge_amount * 1000,  # uAh
                'dchg_capacity': 0,
                'soc': current_soc,
                'cycle': self.total_cycle,
                'timestamp': start_time + timedelta(seconds=len(data_points) * time_step),
                'mode': 'CC',
                'temperature': 23.0 + random.uniform(-0.5, 0.5)
            }
            data_points.append(data_point)
            
            if current_voltage >= target_voltage:
                break
        
        return data_points
    
    def _generate_cccv_charge(self, start_time: datetime, c_rate: float,
                             target_voltage: float, cutoff_c_rate: float,
                             initial_soc: float) -> List[Dict]:
        """CCCV 충전 데이터 생성"""
        data_points = []
        
        # CC 단계
        cc_points = self._generate_cc_charge(start_time, c_rate, target_voltage, initial_soc)
        data_points.extend(cc_points)
        
        if not cc_points:
            return data_points
        
        # CV 단계
        current_soc = cc_points[-1]['soc']
        cv_start_time = cc_points[-1]['timestamp']
        initial_current = c_rate * self.config.capacity_mah
        cutoff_current = cutoff_c_rate * self.config.capacity_mah
        
        time_in_cv = 0
        time_step = 10
        
        while True:
            self.accumulated_step += 1
            time_in_cv += time_step
            
            # CV 중 전류 감소
            current_ma = ChargeController.simulate_cv_current(initial_current, time_in_cv)
            
            if current_ma <= cutoff_current:
                break
            
            # 충전량 계산
            charge_amount = (current_ma * time_step) / 3600  # mAh
            soc_increase = (charge_amount / self.config.capacity_mah) * 100
            current_soc = min(100, current_soc + soc_increase)
            
            data_point = {
                'step': self.accumulated_step,
                'step_type': StepType.CHARGE.value,
                'voltage': target_voltage + random.uniform(-0.002, 0.002),
                'current': current_ma,
                'chg_capacity': charge_amount * 1000,  # uAh
                'dchg_capacity': 0,
                'soc': current_soc,
                'cycle': self.total_cycle,
                'timestamp': cv_start_time + timedelta(seconds=time_in_cv),
                'mode': 'CV',
                'temperature': 23.0 + random.uniform(-0.5, 0.5)
            }
            data_points.append(data_point)
            
            if time_in_cv > 3600:  # 최대 1시간
                break
        
        return data_points
    
    def _generate_cc_discharge(self, start_time: datetime, c_rate: float,
                              cutoff_voltage: float, initial_soc: float) -> List[Dict]:
        """CC 방전 데이터 생성"""
        data_points = []
        current_soc = initial_soc
        current_voltage = OCVModel_SiC_LCO.get_ocv(current_soc)
        
        # 전압 보호 적용
        _, min_voltage = VoltageProtectionAlgorithm.get_voltage_limits(
            self.total_cycle, self.config.max_voltage, cutoff_voltage
        )
        cutoff_voltage = max(cutoff_voltage, min_voltage)
        
        time_step = 10
        current_ma = c_rate * self.config.capacity_mah
        
        while current_voltage > cutoff_voltage and current_soc > 0:
            self.accumulated_step += 1
            
            # 방전량 계산
            discharge_amount = (current_ma * time_step) / 3600  # mAh
            soc_decrease = (discharge_amount / self.config.capacity_mah) * 100
            current_soc = max(0, current_soc - soc_decrease)
            
            # 전압 계산 (IR 드롭 포함)
            ocv = OCVModel_SiC_LCO.get_ocv(current_soc)
            ir_drop = 0.05 * c_rate
            current_voltage = ocv - ir_drop + random.uniform(-0.005, 0.005)
            
            data_point = {
                'step': self.accumulated_step,
                'step_type': StepType.DISCHARGE.value,
                'voltage': current_voltage,
                'current': -current_ma,  # 방전은 음수
                'chg_capacity': 0,
                'dchg_capacity': discharge_amount * 1000,  # uAh
                'soc': current_soc,
                'cycle': self.total_cycle,
                'timestamp': start_time + timedelta(seconds=len(data_points) * time_step),
                'mode': 'CC',
                'temperature': 23.5 + random.uniform(-0.5, 0.5)
            }
            data_points.append(data_point)
            
            if current_voltage <= cutoff_voltage:
                break
        
        return data_points
    
    def _generate_rest(self, start_time: datetime, duration: int,
                      after_charge: bool, soc: float) -> List[Dict]:
        """휴지 구간 데이터 생성 (전압 이완/회복)"""
        data_points = []
        time_step = 10
        num_points = duration // time_step
        
        # 초기 전압 (충/방전 직후)
        if after_charge:
            initial_voltage = OCVModel_SiC_LCO.get_ocv(soc) + 0.18  # 과전압
        else:
            initial_voltage = OCVModel_SiC_LCO.get_ocv(soc) - 0.20  # 부족전압
        
        for i in range(num_points):
            self.accumulated_step += 1
            time_elapsed = i * time_step
            
            # 전압 이완/회복 계산
            if after_charge:
                voltage = VoltageRelaxationModel.voltage_after_charge(
                    time_elapsed, initial_voltage, soc
                )
            else:
                voltage = VoltageRelaxationModel.voltage_after_discharge(
                    time_elapsed, initial_voltage, soc
                )
            
            data_point = {
                'step': self.accumulated_step,
                'step_type': StepType.REST.value,
                'voltage': voltage + random.uniform(-0.002, 0.002),
                'current': random.uniform(-10, 10),  # 잔류 전류
                'chg_capacity': 0,
                'dchg_capacity': 0,
                'soc': soc,
                'cycle': self.total_cycle,
                'timestamp': start_time + timedelta(seconds=time_elapsed),
                'mode': 'REST',
                'temperature': 23.0 + random.uniform(-0.3, 0.3)
            }
            data_points.append(data_point)
        
        return data_points
    
    def _charge_to_soc(self, start_time: datetime, target_soc: float) -> List[Dict]:
        """특정 SOC까지 충전"""
        data_points = []
        current_soc = self.current_soc
        
        if current_soc >= target_soc:
            return data_points
        
        # 0.1C로 목표 SOC까지 충전
        time_step = 10
        current_ma = 0.1 * self.config.capacity_mah
        
        while current_soc < target_soc:
            self.accumulated_step += 1
            
            charge_amount = (current_ma * time_step) / 3600
            soc_increase = (charge_amount / self.config.capacity_mah) * 100
            current_soc = min(target_soc, current_soc + soc_increase)
            
            voltage = OCVModel_SiC_LCO.get_ocv(current_soc) + 0.05
            
            data_point = {
                'step': self.accumulated_step,
                'step_type': StepType.CHARGE.value,
                'voltage': voltage,
                'current': current_ma,
                'chg_capacity': charge_amount * 1000,
                'dchg_capacity': 0,
                'soc': current_soc,
                'cycle': self.total_cycle,
                'timestamp': start_time + timedelta(seconds=len(data_points) * time_step),
                'mode': 'CC',
                'temperature': 23.0
            }
            data_points.append(data_point)
        
        self.current_soc = current_soc
        return data_points
    
    def _measure_rss(self, timestamp: datetime, soc: float) -> Dict:
        """RSS (임피던스) 측정"""
        self.accumulated_step += 1
        
        # SOC별 임피던스 시뮬레이션 (SiC 영향)
        base_impedance = 15.0  # mOhm
        soc_factor = 1.0 + (100 - soc) * 0.005  # SOC가 낮을수록 임피던스 증가
        cycle_factor = 1.0 + self.total_cycle * 0.0001  # 사이클 증가에 따른 임피던스 증가
        
        impedance = base_impedance * soc_factor * cycle_factor
        
        return {
            'step': self.accumulated_step,
            'step_type': StepType.IMPEDANCE.value,
            'voltage': OCVModel_SiC_LCO.get_ocv(soc),
            'current': 0,
            'chg_capacity': 0,
            'dchg_capacity': 0,
            'soc': soc,
            'cycle': self.total_cycle,
            'timestamp': timestamp,
            'mode': 'RSS',
            'temperature': 23.0,
            'impedance': impedance
        }
    
    def generate_pre_development_pattern(self, start_time: datetime) -> List[Dict]:
        """
        선행PF 패턴 생성
        [보증용량 1cycle + RSS 2cycle + 수명패턴 97cycle] × 12번 = 1200 사이클
        """
        all_data = []
        current_time = start_time
        
        for repeat in range(12):  # 12번 반복
            print(f"  생성 중: 반복 {repeat + 1}/12 (사이클 {self.total_cycle + 1}-{self.total_cycle + 100})")
            
            # 보증용량 1 사이클
            self.total_cycle += 1
            warranty_data = self.generate_warranty_cycle(current_time)
            all_data.extend(warranty_data)
            current_time = warranty_data[-1]['timestamp'] + timedelta(seconds=10)
            
            # RSS 2 사이클
            for _ in range(2):
                self.total_cycle += 1
                rss_data = self.generate_rss_cycle(current_time)
                all_data.extend(rss_data)
                current_time = rss_data[-1]['timestamp'] + timedelta(seconds=10)
            
            # 수명패턴 97 사이클
            for _ in range(97):
                self.total_cycle += 1
                
                # 용량 감소 적용
                retention = CapacityFadeModel.get_capacity_retention(self.total_cycle)
                self.config.capacity_mah = int(4352 * retention)
                self.capacity_ah = self.config.capacity_mah / 1000.0
                
                life_data = self.generate_life_cycle(current_time)
                all_data.extend(life_data)
                current_time = life_data[-1]['timestamp'] + timedelta(seconds=10)
        
        return all_data
    
    def generate_production_pattern(self, start_time: datetime) -> List[Dict]:
        """
        상품화 패턴 생성
        [보증용량 1cycle + 수명패턴 99cycle] × 16번 = 1600 사이클
        """
        all_data = []
        current_time = start_time
        
        for repeat in range(16):  # 16번 반복
            print(f"  생성 중: 반복 {repeat + 1}/16 (사이클 {self.total_cycle + 1}-{self.total_cycle + 100})")
            
            # 보증용량 1 사이클
            self.total_cycle += 1
            warranty_data = self.generate_warranty_cycle(current_time)
            all_data.extend(warranty_data)
            current_time = warranty_data[-1]['timestamp'] + timedelta(seconds=10)
            
            # 수명패턴 99 사이클
            for _ in range(99):
                self.total_cycle += 1
                
                # 용량 감소 적용
                retention = CapacityFadeModel.get_capacity_retention(self.total_cycle)
                self.config.capacity_mah = int(4352 * retention)
                self.capacity_ah = self.config.capacity_mah / 1000.0
                
                life_data = self.generate_life_cycle(current_time)
                all_data.extend(life_data)
                current_time = life_data[-1]['timestamp'] + timedelta(seconds=10)
        
        return all_data

# ===== 메인 실행 함수 =====

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("배터리 수명 시험 패턴 생성기")
    print("SiC+Graphite/LCO 시스템")
    print("=" * 60)
    
    # 배터리 시스템 구성
    config = BatterySystemConfig(
        anode="SiC+Graphite",
        cathode="LCO",
        capacity_mah=4352
    )
    
    # 생성기 초기화
    generator = BatteryLifePatternGenerator(config)
    
    # 시작 시간
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    
    print("\n선택할 패턴:")
    print("1. 선행PF (1200 사이클)")
    print("2. 상품화 (1600 사이클)")
    print("3. 테스트 (100 사이클)")
    
    choice = input("\n패턴 선택 (1-3): ").strip()
    
    if choice == '1':
        print("\n선행PF 패턴 생성 중...")
        data = generator.generate_pre_development_pattern(start_time)
        pattern_name = "pre_development"
        total_cycles = 1200
    elif choice == '2':
        print("\n상품화 패턴 생성 중...")
        data = generator.generate_production_pattern(start_time)
        pattern_name = "production"
        total_cycles = 1600
    elif choice == '3':
        print("\n테스트 패턴 생성 중 (100 사이클)...")
        # 테스트용 100 사이클만
        data = []
        current_time = start_time
        for i in range(100):
            generator.total_cycle += 1
            if i % 10 == 0:
                print(f"  사이클 {generator.total_cycle}")
            cycle_data = generator.generate_life_cycle(current_time)
            data.extend(cycle_data)
            current_time = cycle_data[-1]['timestamp'] + timedelta(seconds=10)
        pattern_name = "test"
        total_cycles = 100
    else:
        print("잘못된 선택입니다.")
        return
    
    print(f"\n생성 완료:")
    print(f"  - 총 사이클: {total_cycles}")
    print(f"  - 총 데이터 포인트: {len(data):,}")
    print(f"  - 패턴 타입: {pattern_name}")
    
    # 데이터 저장 여부
    save = input("\n데이터를 저장하시겠습니까? (y/n): ").strip().lower()
    if save == 'y':
        output_dir = Path(f"battery_life_data_{pattern_name}")
        output_dir.mkdir(exist_ok=True)
        
        # 데이터프레임 생성
        df = pd.DataFrame(data)
        
        # CSV 저장
        output_file = output_dir / f"battery_life_{pattern_name}_{total_cycles}cy.csv"
        df.to_csv(output_file, index=False)
        print(f"\n저장 완료: {output_file}")
        print(f"  - 파일 크기: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
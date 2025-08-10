#!/usr/bin/env python3
"""
배터리 수명 패턴 생성기 테스트

SiC+Graphite/LCO 시스템 배터리 패턴 생성기의
핵심 기능들을 테스트하는 스크립트입니다.

주요 테스트:
- OCV-SOC 모델 검증
- 전압 이완/회복 모델 검증  
- 용량 감소 모델 검증
- 패턴 생성 기능 검증
- PNE 형식 변환 검증
"""

import sys
import traceback
from datetime import datetime
from battery_life_pattern_generator import (
    BatteryLifePatternGenerator,
    BatterySystemConfig,
    OCVModel_SiC_LCO,
    VoltageRelaxationModel,
    CapacityFadeModel
)

def test_ocv_model():
    """OCV 모델 테스트"""
    print("=== OCV 모델 테스트 ===")
    
    test_soc_values = [0, 10, 20, 30, 50, 70, 80, 90, 100]
    
    for soc in test_soc_values:
        ocv = OCVModel_SiC_LCO.get_ocv(soc)
        print(f"SOC {soc}%: {ocv:.3f}V")
    
    print("OCV 모델 테스트 완료\n")

def test_voltage_relaxation():
    """전압 이완 모델 테스트"""
    print("=== 전압 이완 모델 테스트 ===")
    
    # 충전 후 이완
    print("충전 후 전압 이완:")
    for t in [0, 30, 60, 120, 300, 600]:
        v = VoltageRelaxationModel.voltage_after_charge(t, 4.53, 100)
        print(f"  {t}초: {v:.3f}V")
    
    print("\n방전 후 전압 회복:")
    for t in [0, 30, 60, 120, 300, 600]:
        v = VoltageRelaxationModel.voltage_after_discharge(t, 3.0, 0)
        print(f"  {t}초: {v:.3f}V")
    
    print("전압 이완 모델 테스트 완료\n")

def test_capacity_fade():
    """용량 감소 모델 테스트"""
    print("=== 용량 감소 모델 테스트 ===")
    
    test_cycles = [0, 50, 100, 200, 500, 1000, 1200, 1600]
    
    for cycle in test_cycles:
        retention = CapacityFadeModel.get_capacity_retention(cycle)
        capacity = 4352 * retention
        print(f"사이클 {cycle}: {retention:.3f} ({capacity:.0f}mAh)")
    
    print("용량 감소 모델 테스트 완료\n")

def test_small_pattern():
    """작은 패턴 생성 테스트 (10 사이클)"""
    print("=== 소규모 패턴 테스트 ===")
    
    try:
        # 작은 배터리 구성
        config = BatterySystemConfig(capacity_mah=1000)  # 1Ah로 테스트
        generator = BatteryLifePatternGenerator(config)
        
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        current_time = start_time
        
        all_data = []
        
        # 10 사이클만 생성
        for i in range(10):
            generator.total_cycle += 1
            print(f"  사이클 {generator.total_cycle} 생성 중...")
            
            if i % 3 == 0:
                # 보증용량 측정
                cycle_data = generator.generate_warranty_cycle(current_time)
            elif i % 3 == 1:
                # RSS 측정
                cycle_data = generator.generate_rss_cycle(current_time)
            else:
                # 수명패턴
                cycle_data = generator.generate_life_cycle(current_time)
            
            all_data.extend(cycle_data)
            current_time = cycle_data[-1]['timestamp'] if cycle_data else current_time
            
            print(f"    - 데이터 포인트: {len(cycle_data)}")
        
        print(f"\n소규모 패턴 생성 완료:")
        print(f"  - 총 사이클: 10")
        print(f"  - 총 데이터 포인트: {len(all_data)}")
        
        # 첫 번째와 마지막 데이터 포인트 출력
        if all_data:
            first = all_data[0]
            last = all_data[-1]
            
            print(f"\n첫 번째 데이터 포인트:")
            print(f"  - 사이클: {first['cycle']}")
            print(f"  - 전압: {first['voltage']:.3f}V")
            print(f"  - 전류: {first['current']:.1f}mA")
            print(f"  - SOC: {first['soc']:.1f}%")
            
            print(f"\n마지막 데이터 포인트:")
            print(f"  - 사이클: {last['cycle']}")
            print(f"  - 전압: {last['voltage']:.3f}V")
            print(f"  - 전류: {last['current']:.1f}mA")
            print(f"  - SOC: {last['soc']:.1f}%")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        traceback.print_exc()
    
    print("소규모 패턴 테스트 완료\n")

def test_data_format():
    """데이터 형식 테스트"""
    print("=== 데이터 형식 테스트 ===")
    
    try:
        from create_example_data import ExampleDataCreator
        
        creator = ExampleDataCreator()
        
        # 테스트 데이터 포인트
        test_point = {
            'step': 1,
            'step_type': 1,  # CHARGE
            'voltage': 4.2,
            'current': 1000.0,
            'chg_capacity': 500000,
            'dchg_capacity': 0,
            'soc': 80.0,
            'cycle': 1,
            'timestamp': datetime(2024, 1, 1, 12, 0, 0),
            'mode': 'CC',
            'temperature': 23.5
        }
        
        # PNE 형식 변환 테스트
        pne_data = creator._convert_to_pne_format(test_point)
        
        print(f"PNE 형식 변환:")
        print(f"  - 원본 전압: {test_point['voltage']}V")
        print(f"  - PNE 전압: {pne_data[8]}uV ({pne_data[8]/1000000:.3f}V)")
        print(f"  - 원본 전류: {test_point['current']}mA")
        print(f"  - PNE 전류: {pne_data[9]}uA ({pne_data[9]/1000:.3f}mA)")
        print(f"  - PNE 데이터 길이: {len(pne_data)} (예상: 46)")
        
        if len(pne_data) == 46:
            print("  [O] PNE 형식 변환 성공")
        else:
            print("  [X] PNE 형식 변환 실패")
        
    except Exception as e:
        print(f"데이터 형식 테스트 오류: {e}")
        traceback.print_exc()
    
    print("데이터 형식 테스트 완료\n")

def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("배터리 수명 패턴 생성기 테스트")
    print("=" * 60)
    
    # 기본 모델 테스트
    test_ocv_model()
    test_voltage_relaxation()
    test_capacity_fade()
    
    # 패턴 생성 테스트
    test_small_pattern()
    
    # 데이터 형식 테스트
    test_data_format()
    
    print("=" * 60)
    print("모든 테스트 완료")
    print("=" * 60)

if __name__ == "__main__":
    main()
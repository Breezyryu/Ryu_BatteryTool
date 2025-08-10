#!/usr/bin/env python3
"""
배터리 수명 패턴 간단 데모

SiC+Graphite/LCO 배터리 시스템의 실제 수명 시험 패턴을
시연하는 스크립트입니다.

기능:
- 빠른 테스트 (10 사이클)
- 작은 데모 파일 생성 (50 사이클)
- PNE 형식 데이터 생성

사용법:
    python simple_demo.py
"""

def quick_test():
    """빠른 테스트 (10 사이클)"""
    print("=" * 60)
    print("빠른 테스트 - 10 사이클 배터리 패턴")
    print("=" * 60)
    
    try:
        from battery_life_pattern_generator import BatteryLifePatternGenerator, BatterySystemConfig
        from datetime import datetime
        
        # 소형 배터리 구성
        config = BatterySystemConfig(capacity_mah=1000)  # 1Ah 테스트
        generator = BatteryLifePatternGenerator(config)
        
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        current_time = start_time
        all_data = []
        
        print("\n생성 중...")
        for i in range(10):
            generator.total_cycle += 1
            
            if i % 3 == 0:
                cycle_data = generator.generate_warranty_cycle(current_time)
                pattern = "보증용량"
            elif i % 3 == 1:
                cycle_data = generator.generate_rss_cycle(current_time)
                pattern = "RSS"
            else:
                cycle_data = generator.generate_life_cycle(current_time)
                pattern = "수명패턴"
            
            all_data.extend(cycle_data)
            current_time = cycle_data[-1]['timestamp'] if cycle_data else current_time
            
            print(f"  사이클 {generator.total_cycle} ({pattern}): {len(cycle_data)} 포인트")
        
        print(f"\n[완료] 테스트 완료!")
        print(f"   - 총 사이클: 10")
        print(f"   - 총 데이터 포인트: {len(all_data):,}")
        
        # 샘플 데이터 출력
        if all_data:
            sample = all_data[len(all_data)//2]  # 중간 데이터
            print(f"\n[샘플] 데이터:")
            print(f"   - 사이클: {sample['cycle']}")
            print(f"   - 전압: {sample['voltage']:.3f}V")
            print(f"   - 전류: {sample['current']:.1f}mA") 
            print(f"   - SOC: {sample['soc']:.1f}%")
            print(f"   - 온도: {sample.get('temperature', 23):.1f}C")
        
        return True
        
    except Exception as e:
        print(f"[오류] 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_small_demo():
    """작은 데모 파일 생성"""
    print("\n" + "=" * 60)
    print("작은 데모 파일 생성 (50 사이클)")
    print("=" * 60)
    
    try:
        from create_example_data import ExampleDataCreator
        from pathlib import Path
        
        creator = ExampleDataCreator()
        
        # 작은 배터리로 50사이클만 생성
        print("\n[PNE] 50 사이클 데모 데이터 생성 중...")
        
        from battery_life_pattern_generator import BatteryLifePatternGenerator, BatterySystemConfig
        from datetime import datetime
        
        config = BatterySystemConfig(capacity_mah=2000)  # 2Ah
        generator = BatteryLifePatternGenerator(config)
        
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        current_time = start_time
        all_data = []
        
        # 50 사이클만 생성 (빠른 데모용)
        for i in range(50):
            generator.total_cycle += 1
            
            if i % 10 == 0:
                cycle_data = generator.generate_warranty_cycle(current_time)
                pattern = "보증용량"
            elif i % 10 == 1:
                cycle_data = generator.generate_rss_cycle(current_time)
                pattern = "RSS"
            else:
                cycle_data = generator.generate_life_cycle(current_time)
                pattern = "수명패턴"
            
            all_data.extend(cycle_data)
            current_time = cycle_data[-1]['timestamp'] if cycle_data else current_time
            
            if i % 10 == 0 or i == 49:
                print(f"  진행: {i+1}/50 사이클 ({pattern})")
        
        # PNE 형식으로 저장
        demo_path = Path("demo_data_small") / "LGES_SiC_LCO_2000mAh_데모_50cy"
        pne_path = demo_path / "M01Ch003[003]" / "Restore"
        pne_path.mkdir(parents=True, exist_ok=True)
        
        creator._save_pne_format_files(all_data, pne_path, 50)
        
        print(f"\n[완료] 데모 파일 생성 완료!")
        print(f"[경로] {demo_path}")
        
        # 파일 확인
        files = list(pne_path.glob("*.csv"))
        total_size = sum(f.stat().st_size for f in files)
        
        print(f"[결과] 생성된 파일:")
        print(f"   - 파일 수: {len(files)}")
        print(f"   - 총 크기: {total_size/1024/1024:.2f} MB")
        print(f"   - 총 데이터 포인트: {len(all_data):,}")
        
        # 첫 번째 파일의 몇 줄 확인
        if files:
            first_file = sorted(files)[0]
            with open(first_file, 'r') as f:
                lines = f.readlines()
            print(f"   - 첫 번째 파일 줄 수: {len(lines)}")
        
        return True
        
    except Exception as e:
        print(f"[오류] 데모 파일 생성 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    print("배터리 수명 패턴 생성기 간단 데모")
    print("=" * 60)
    
    # 1. 빠른 테스트
    success1 = quick_test()
    
    if success1:
        print("\n빠른 테스트 성공! 이제 데모 파일을 생성합니다.")
        
        # 2. 작은 데모 파일 생성
        success2 = create_small_demo()
        
        if success2:
            print("\n" + "=" * 60)
            print("[완료] 모든 데모가 성공적으로 완료되었습니다!")
            print("\n[특징] SiC+Graphite/LCO 배터리 시스템:")
            print("   - 4단계 충전 (2.0C->1.65C->1.4C->1.0C)")
            print("   - 2단계 방전 (1.0C->0.5C)")
            print("   - CV Cut-off (Step 3: 0.14C, Step 4: 0.1C)")
            print("   - 전압 이완/회복 (SiC 특성)")
            print("   - 용량 감소 모델 (SEI 형성)")
            print("   - RSS 임피던스 측정")
            print("\n[사용법] 전체 1200/1600 사이클 생성:")
            print("   python create_example_data.py")
            print("=" * 60)
        else:
            print("[오류] 데모 파일 생성에 실패했습니다.")
    else:
        print("[오류] 빠른 테스트에 실패했습니다.")

if __name__ == "__main__":
    main()
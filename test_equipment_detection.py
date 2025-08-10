#!/usr/bin/env python3
"""
Test script for equipment type detection fixes
Tests the enhanced equipment detection and data loading functionality
"""

import logging
from pathlib import Path
from battery_pattern_analyzer import BatteryPatternAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_equipment_detection():
    """Test the equipment type detection with various path examples"""
    print("=" * 80)
    print("Equipment Type Detection Test")
    print("=" * 80)
    
    analyzer = BatteryPatternAnalyzer()
    
    # Test cases with different path formats
    test_paths = [
        "D:\\pne\\LGES_G3_MP1_4352mAh_상온수명",
        "C:\\data\\Samsung_SDI_NCM811_4000mAh",
        "\\\\server\\share\\ATL_high_capacity_4200mAh_고온",
        "data/toyo_test_files",
        "unknown_format_test_path",
        "BYD_Blade_Battery_4500mAh_Test",
        "COSMX_test_data_3800mAh"
    ]
    
    print(f"Testing {len(test_paths)} different path formats...\n")
    
    for i, test_path in enumerate(test_paths, 1):
        print(f"[Test {i}] Path: {test_path}")
        
        # Extract battery info (includes equipment type detection)
        battery_info = analyzer.extract_capacity_from_path(test_path)
        
        print(f"  - Manufacturer: {battery_info.get('manufacturer', 'Unknown')}")
        print(f"  - Equipment Type: {battery_info.get('equipment_type', 'Unknown')}")
        print(f"  - Capacity: {battery_info.get('capacity_mah', 'Unknown')} mAh")
        print(f"  - Additional Info: {[k for k in battery_info.keys() if k not in ['manufacturer', 'equipment_type', 'capacity_mah', 'capacity_ah', 'path']]}")
        print()

def test_path_structure_detection():
    """Test PNE/Toyo structure detection with real directory structures"""
    print("=" * 80)
    print("Path Structure Detection Test")
    print("=" * 80)
    
    analyzer = BatteryPatternAnalyzer()
    
    # Test with current directory structure
    current_dir = Path.cwd()
    print(f"Testing current directory: {current_dir}")
    
    # Test enhanced detection method
    equipment_type = analyzer._determine_equipment_type_enhanced(str(current_dir))
    print(f"Detected equipment type: {equipment_type}")
    
    # Test structure detection methods
    has_pne = analyzer._has_pne_structure(current_dir)
    has_toyo = analyzer._has_toyo_structure(current_dir)
    
    print(f"Has PNE structure: {has_pne}")
    print(f"Has Toyo structure: {has_toyo}")
    
    # Show some directory contents
    analyzer._log_path_structure_summary(str(current_dir))

def test_data_loading_fallback():
    """Test data loading with fallback mechanisms"""
    print("=" * 80)
    print("Data Loading Fallback Test")
    print("=" * 80)
    
    analyzer = BatteryPatternAnalyzer()
    
    # Test with current directory (should fail gracefully)
    current_dir = Path.cwd()
    print(f"Testing data loading from: {current_dir}")
    
    # Set equipment type to Unknown to trigger fallback
    analyzer.equipment_type = 'Unknown'
    
    try:
        data = analyzer.load_and_concatenate_data(str(current_dir))
        print(f"Data loading result: {len(data)} rows loaded")
        if not data.empty:
            print(f"Columns: {list(data.columns[:5])}...")
    except Exception as e:
        print(f"Expected error (no real data): {e}")

def main():
    """Run all tests"""
    print("Starting Battery Analyzer Equipment Detection Tests")
    print("=" * 80)
    
    try:
        # Test 1: Equipment Detection
        test_equipment_detection()
        
        # Test 2: Path Structure Detection
        test_path_structure_detection()
        
        # Test 3: Data Loading Fallback
        test_data_loading_fallback()
        
        print("=" * 80)
        print("All tests completed!")
        print("=" * 80)
        
        print("\nTest Summary:")
        print("1. Equipment type detection from path names")
        print("2. PNE/Toyo structure detection")
        print("3. Enhanced error handling and logging")
        print("4. Fallback mechanisms for unknown equipment types")
        print("5. Improved debugging information")
        
        print("\nReady for testing with real battery data paths!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
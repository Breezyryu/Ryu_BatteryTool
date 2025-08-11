#!/usr/bin/env python3
"""
Multi-Channel Analysis Test Script
"""

from battery_pattern_analyzer import BatteryPatternAnalyzer
import sys
import os

def test_multi_channel_detection():
    """Test multi-channel detection system"""
    
    analyzer = BatteryPatternAnalyzer()
    
    # Test with Toyo example data
    toyo_path = "example_data_toyo/Toyo_SiC_LCO_4352mAh_선행PF_1200cy"
    
    if os.path.exists(toyo_path):
        print("=== Testing Toyo Multi-Channel Detection ===")
        
        # Test channel detection
        channels = analyzer.detect_channels(toyo_path)
        print(f"Detected {len(channels)} channels:")
        
        for ch_id, ch_info in channels.items():
            print(f"  {ch_id}: {ch_info['path']}")
            print(f"    - Equipment: {ch_info['equipment_type']}")
            print(f"    - Files: {ch_info['file_count']}")
        
        # Test data loading
        print("\n=== Testing Data Loading ===")
        data = analyzer.load_and_concatenate_data(toyo_path)
        
        if not data.empty:
            print(f"Loaded data shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            print(f"Channel distribution:")
            if 'channel_id' in data.columns:
                channel_counts = data['channel_id'].value_counts()
                for ch_id, count in channel_counts.items():
                    print(f"  {ch_id}: {count:,} rows")
        else:
            print("No data loaded")
    else:
        print(f"Test path not found: {toyo_path}")
    
    # Test with PNE example data
    pne_path = "example_data/LGES_SiC_LCO_4352mAh_선행PF_1200cy"
    
    if os.path.exists(pne_path):
        print("\n=== Testing PNE Multi-Channel Detection ===")
        
        # Test channel detection
        channels = analyzer.detect_channels(pne_path)
        print(f"Detected {len(channels)} channels:")
        
        for ch_id, ch_info in channels.items():
            print(f"  {ch_id}: {ch_info['path']}")
            print(f"    - Equipment: {ch_info['equipment_type']}")
            print(f"    - Files: {ch_info['file_count']}")
    else:
        print(f"PNE test path not found: {pne_path}")

if __name__ == "__main__":
    print("Multi-Channel Battery Analysis Test")
    print("=" * 50)
    
    test_multi_channel_detection()
    
    print("\n" + "=" * 50)
    print("Test completed successfully!")
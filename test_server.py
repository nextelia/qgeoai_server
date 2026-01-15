"""
Test script for QGeoAI Server
Validates Phase 1 implementation
"""

import sys
import time
import json
from pathlib import Path

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent))

from qgeoai_client import QGeoAIClient


def test_server_lifecycle():
    """Test server start/stop and health check"""
    print("\n=== Testing Server Lifecycle ===")
    
    client = QGeoAIClient()
    
    # Check if server is already running
    if client.is_server_running():
        print("✓ Server is already running")
    else:
        print("✗ Server is not running, attempting to start...")
        if client.start_server(wait=True, max_wait=15):
            print("✓ Server started successfully")
        else:
            print("✗ Failed to start server")
            return False
    
    # Verify server is responsive
    if client.is_server_running():
        print("✓ Server health check passed")
    else:
        print("✗ Server health check failed")
        return False
    
    return True


def test_authentication():
    """Test token-based authentication"""
    print("\n=== Testing Authentication ===")
    
    client = QGeoAIClient()
    
    # Check token file exists
    token_file = client.config_dir / 'server.token'
    if token_file.exists():
        print(f"✓ Token file exists: {token_file}")
    else:
        print(f"✗ Token file not found: {token_file}")
        return False
    
    # Check token is loaded
    if client.token:
        print(f"✓ Token loaded (length: {len(client.token)})")
    else:
        print("✗ Failed to load token")
        return False
    
    return True


def test_status_endpoint():
    """Test /status endpoint"""
    print("\n=== Testing Status Endpoint ===")
    
    client = QGeoAIClient()
    
    try:
        status = client.get_status()
        print("✓ Status endpoint accessible")
        print(f"  Status: {status.get('status')}")
        
        if 'versions' in status:
            print("  Dependencies:")
            for key, value in status['versions'].items():
                print(f"    - {key}: {value}")
        
        return True
    except Exception as e:
        print(f"✗ Status endpoint failed: {e}")
        return False


def test_toolbox_info():
    """Test toolbox info endpoint"""
    print("\n=== Testing Toolbox Info ===")
    
    client = QGeoAIClient()
    
    try:
        import requests
        response = requests.get(
            f"{client.base_url}/toolbox/info",
            headers=client.headers,
            timeout=5
        )
        response.raise_for_status()
        info = response.json()
        
        print("✓ Toolbox info endpoint accessible")
        print(f"  Available operations: {info.get('available_operations')}")
        print(f"  Regulariser available: {info.get('regulariser_available')}")
        
        return True
    except Exception as e:
        print(f"✗ Toolbox info failed: {e}")
        return False


def test_regularize_mock():
    """Test regularize endpoint with mock data (without actual buildingregulariser)"""
    print("\n=== Testing Regularize Endpoint (Mock) ===")
    
    client = QGeoAIClient()
    
    # Create mock input file
    mock_input = client.config_dir / 'test_input.geojson'
    mock_output = client.config_dir / 'test_output.geojson'
    
    # Simple GeoJSON for testing
    mock_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [0, 0], [1, 0], [1, 1], [0, 1], [0, 0]
                    ]]
                },
                "properties": {"id": 1}
            }
        ]
    }
    
    try:
        # Write mock input
        mock_input.write_text(json.dumps(mock_geojson))
        print(f"✓ Created mock input: {mock_input}")
        
        # Try to call regularize (will fail if buildingregulariser not installed)
        try:
            result = client.regularize_buildings(
                str(mock_input),
                str(mock_output),
                parallel_threshold=1.0,
                simplify_tolerance=0.5
            )
            print("✓ Regularize endpoint called successfully")
            print(f"  Status: {result.get('status')}")
            print(f"  Output: {result.get('output_path')}")
            return True
        except Exception as e:
            if "503" in str(e) or "buildingregulariser not available" in str(e):
                print("⚠ Regularize endpoint works, but buildingregulariser not installed (expected)")
                return True
            else:
                print(f"✗ Unexpected error: {e}")
                return False
            
    except Exception as e:
        print(f"✗ Mock test failed: {e}")
        return False
    finally:
        # Cleanup
        if mock_input.exists():
            mock_input.unlink()
        if mock_output.exists():
            mock_output.unlink()


def test_placeholder_endpoints():
    """Test placeholder endpoints for future phases"""
    print("\n=== Testing Placeholder Endpoints ===")
    
    client = QGeoAIClient()
    
    endpoints = [
        ("/train/info", "Training"),
        ("/predict/info", "Prediction"),
        ("/annotate/info", "Annotation"),
    ]
    
    all_ok = True
    for endpoint, name in endpoints:
        try:
            import requests
            response = requests.get(
                f"{client.base_url}{endpoint}",
                timeout=5
            )
            response.raise_for_status()
            info = response.json()
            print(f"✓ {name} info endpoint: {info.get('status')}")
        except Exception as e:
            print(f"✗ {name} info endpoint failed: {e}")
            all_ok = False
    
    return all_ok


def main():
    """Run all tests"""
    print("=" * 60)
    print("QGeoAI Server Test Suite - Phase 1")
    print("=" * 60)
    
    tests = [
        ("Server Lifecycle", test_server_lifecycle),
        ("Authentication", test_authentication),
        ("Status Endpoint", test_status_endpoint),
        ("Toolbox Info", test_toolbox_info),
        ("Regularize Endpoint", test_regularize_mock),
        ("Placeholder Endpoints", test_placeholder_endpoints),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
        
        time.sleep(0.5)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

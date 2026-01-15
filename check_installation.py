"""
QGeoAI Server Diagnostic Tool
Checks installation status and configuration
"""

import sys
from pathlib import Path


def check_installation():
    """Run diagnostic checks on QGeoAI server installation"""
    
    print("=" * 70)
    print("QGeoAI Server Diagnostic")
    print("=" * 70)
    
    results = []
    
    # Check 1: Config directory
    print("\n[1] Checking config directory...")
    config_dir = Path.home() / '.qgeoai'
    if config_dir.exists():
        print(f"  ✓ Config directory exists: {config_dir}")
        results.append(True)
    else:
        print(f"  ✗ Config directory not found: {config_dir}")
        results.append(False)
    
    # Check 2: Server directory
    print("\n[2] Checking server installation...")
    server_dir = config_dir / 'server'
    if server_dir.exists():
        print(f"  ✓ Server directory exists: {server_dir}")
        
        required_files = [
            'server.py',
            'endpoints/__init__.py',
            'endpoints/toolbox.py',
            'utils/__init__.py',
            'utils/port_finder.py',
            'utils/token_manager.py',
            'requirements.txt',
            'sam2/build_sam.py',
            'sam2/sam2_image_predictor.py',
        ]
        
        all_files_present = True
        for file in required_files:
            file_path = server_dir / file
            if file_path.exists():
                print(f"    ✓ {file}")
            else:
                print(f"    ✗ {file} MISSING")
                all_files_present = False
        
        results.append(all_files_present)
    else:
        print(f"  ✗ Server directory not found: {server_dir}")
        results.append(False)
    
    # Check 3: Python environment
    print("\n[3] Checking Python environment...")
    env_dir = config_dir / 'env'
    
    if sys.platform == 'win32':
        python_exe = env_dir / 'Scripts' / 'python.exe'
        pip_exe = env_dir / 'Scripts' / 'pip.exe'
    else:
        python_exe = env_dir / 'bin' / 'python'
        pip_exe = env_dir / 'bin' / 'pip'
    
    if python_exe.exists():
        print(f"  ✓ Python executable: {python_exe}")
        
        # Check Python version
        import subprocess
        try:
            result = subprocess.run(
                [str(python_exe), '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            version = result.stdout.strip() or result.stderr.strip()
            print(f"    Version: {version}")
        except Exception as e:
            print(f"    ⚠ Could not get version: {e}")
        
        results.append(True)
    else:
        print(f"  ✗ Python executable not found: {python_exe}")
        print(f"    Please create environment: python -m venv {env_dir}")
        results.append(False)
    
    # Check 4: Dependencies
    print("\n[4] Checking dependencies...")
    if python_exe.exists():
        required_packages = [
            'fastapi',
            'uvicorn',
            'pydantic',
            'requests',
        ]
        
        import subprocess
        try:
            result = subprocess.run(
                [str(pip_exe), 'list', '--format=freeze'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            installed = result.stdout.lower()
            all_deps_ok = True
            
            for package in required_packages:
                if package in installed:
                    print(f"  ✓ {package}")
                else:
                    print(f"  ✗ {package} NOT INSTALLED")
                    all_deps_ok = False
            
            # Check optional heavy dependencies
            optional_packages = ['torch', 'ultralytics', 'numpy', 'pillow', 'opencv-python', 'hydra-core']
            print("\n  Optional dependencies:")
            for package in optional_packages:
                if package in installed:
                    # Try to get version
                    for line in result.stdout.split('\n'):
                        if line.lower().startswith(package):
                            print(f"    ✓ {line}")
                            break
                else:
                    print(f"    - {package} (not installed)")
            
            results.append(all_deps_ok)
        except Exception as e:
            print(f"  ⚠ Could not check dependencies: {e}")
            results.append(False)
    else:
        print("  ⚠ Skipping (Python not found)")
        results.append(False)
    
    # Check 5: Client module
    print("\n[5] Checking client module...")
    client_file = config_dir / 'qgeoai_client.py'
    if client_file.exists():
        print(f"  ✓ Client module: {client_file}")
        results.append(True)
    else:
        print(f"  ✗ Client module not found: {client_file}")
        results.append(False)
    
    # Check 6: Launcher scripts
    print("\n[6] Checking launcher scripts...")
    if sys.platform == 'win32':
        launcher = config_dir / 'start_server.bat'
    else:
        launcher = config_dir / 'start_server.sh'
    
    if launcher.exists():
        print(f"  ✓ Launcher script: {launcher}")
        results.append(True)
    else:
        print(f"  ✗ Launcher script not found: {launcher}")
        results.append(False)
    
    # Check 7: Server status
    print("\n[7] Checking server status...")
    token_file = config_dir / 'server.token'
    port_file = config_dir / 'server.port'
    
    server_running = False
    if port_file.exists():
        try:
            port = int(port_file.read_text().strip())
            print(f"  ✓ Server port file found: {port}")
            
            # Try to connect
            import requests
            try:
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                if response.status_code == 200:
                    print(f"  ✓ Server is RUNNING on port {port}")
                    server_running = True
                else:
                    print(f"  ⚠ Server responded with status {response.status_code}")
            except requests.RequestException:
                print(f"  - Server is NOT running (port {port})")
        except Exception as e:
            print(f"  ⚠ Could not check server: {e}")
    else:
        print("  - Server port file not found (server not running)")
    
    if token_file.exists():
        print(f"  ✓ Token file exists: {token_file}")
    else:
        print("  - Token file not found (server not started yet)")
    
    results.append(True)  # This check is informational

    print("\n[8] Checking SAM2 checkpoints...")
    checkpoint_dir = config_dir / 'sam2_checkpoints'
    if checkpoint_dir.exists():
        print(f"  ✓ Checkpoint directory: {checkpoint_dir}")
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            for cp in checkpoints:
                size_mb = cp.stat().st_size / (1024 * 1024)
                print(f"    ✓ {cp.name} ({size_mb:.1f} MB)")
        else:
            print("    - No checkpoints downloaded yet (will download on first use)")
    else:
        print(f"  - Checkpoint directory not found: {checkpoint_dir}")
        print("    (Will be created on first SAM2 load)")
    
    results.append(True)  # Informational, checkpoints download on demand
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"\n✓ All checks passed ({passed}/{total})")
        if server_running:
            print("\n✓ Server is currently running")
        else:
            print("\n⚠ Server is installed but not running")
            print(f"  Start with: {launcher}")
        print("\nInstallation is complete and ready to use!")
        return 0
    else:
        failed = total - passed
        print(f"\n✗ {failed} check(s) failed ({passed}/{total} passed)")
        print("\nRecommended actions:")
        
        if not config_dir.exists():
            print("  1. Run install_server.py to install the server")
        
        if not python_exe.exists():
            print(f"  2. Create Python environment: python -m venv {env_dir}")
        
        if config_dir.exists() and not server_dir.exists():
            print("  3. Run install_server.py to copy server files")
        
        if python_exe.exists() and not all_deps_ok:
            print(f"  4. Install dependencies: {pip_exe} install -r {server_dir / 'requirements.txt'}")
        
        return 1


def main():
    """Main entry point"""
    try:
        sys.exit(check_installation())
    except KeyboardInterrupt:
        print("\n\nDiagnostic cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

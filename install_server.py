"""
Installation script for QGeoAI Server
Deploys server files to ~/.qgeoai/server and installs dependencies
"""

import sys
import shutil
import subprocess
from pathlib import Path


def install_server():
    """Install QGeoAI server to user's home directory"""
    
    print("=" * 60)
    print("QGeoAI Server Installation")
    print("=" * 60)
    
    # Paths
    home = Path.home()
    qgeoai_dir = home / '.qgeoai'
    server_dir = qgeoai_dir / 'server'
    env_dir = qgeoai_dir / 'env'
    
    # Source directory (where this script is)
    source_dir = Path(__file__).parent
    
    print(f"\nSource directory: {source_dir}")
    print(f"Target directory: {server_dir}")
    print(f"Python environment: {env_dir}")
    
    # Create directories
    print("\n1. Creating directories...")
    qgeoai_dir.mkdir(parents=True, exist_ok=True)
    server_dir.mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")
    
    # Copy server files
    print("\n2. Copying server files...")
    
    files_to_copy = [
        'server.py',
        'endpoints/',
        'core/',
        'utils/',
        'sam2/',
        'models',
        'requirements.txt',
    ]
    
    for item in files_to_copy:
        source = source_dir / item
        target = server_dir / item
        
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
            print(f"  ✓ Copied {item}/")
        elif source.is_file():
            shutil.copy2(source, target)
            print(f"  ✓ Copied {item}")
        else:
            print(f"  ⚠ Warning: {item} not found")
    
    # Copy client file to qgeoai root for easy import
    client_source = source_dir / 'qgeoai_client.py'
    client_target = qgeoai_dir / 'qgeoai_client.py'
    if client_source.exists():
        shutil.copy2(client_source, client_target)
        print(f"  ✓ Copied qgeoai_client.py to {qgeoai_dir}")
    
    # Check if virtual environment exists
    print("\n3. Checking Python environment...")
    
    if sys.platform == 'win32':
        python_exe = env_dir / 'Scripts' / 'python.exe'
        pip_exe = env_dir / 'Scripts' / 'pip.exe'
    else:
        python_exe = env_dir / 'bin' / 'python'
        pip_exe = env_dir / 'bin' / 'pip'
    
    if not python_exe.exists():
        print(f"✗ Python environment not found at {env_dir}")
        print("  Please create the environment first with:")
        print(f"  python -m venv {env_dir}")
        return False
    
    print(f"✓ Python environment found: {python_exe}")
    
    # Install dependencies
    print("\n4. Installing server dependencies...")

    requirements = server_dir / 'requirements.txt'
    opencv_marker = server_dir / '.opencv_resolved'

    if requirements.exists():
        try:
            if not opencv_marker.exists():
                print("  First install: resolving OpenCV conflict...")
                print("  (This may take 15-30 minutes)")
                
                # Step 1: Install everything EXCEPT albumentations first
                print("  Step 1/3: Installing base dependencies...")
                subprocess.check_call([
                    str(pip_exe), 'install', '-r', str(requirements),
                    '--ignore-installed', 'albumentations'
                ])
                
                # Step 2: Install albumentations WITHOUT deps (opencv already installed)
                print("  Step 2/3: Installing albumentations...")
                subprocess.check_call([
                    str(pip_exe), 'install', 
                    'albumentations>=1.4.0,<2.0.0',
                    '--no-deps'
                ])
                
                # Step 3: Install albumentations' other deps (not opencv)
                print("  Step 3/3: Installing albumentations dependencies...")
                subprocess.check_call([
                    str(pip_exe), 'install', 
                    'albucore>=0.0.20',
                    'numpy', 'scipy', 'scikit-image', 'pyyaml'
                ])
                
                opencv_marker.touch()
                print("  ✓ OpenCV conflict resolved")
            else:
                print("  Updating dependencies...")
                subprocess.check_call([
                    str(pip_exe), 'install', '-r', str(requirements), '--upgrade'
                ])
            
            print("✓ Dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install dependencies: {e}")
            return False
    else:
        print("⚠ requirements.txt not found, skipping dependency installation")
    
    # Check for CUDA and reinstall PyTorch with GPU support if available
    print("\n5. Checking for CUDA GPU support...")

    try:
        # Try to detect NVIDIA GPU using nvidia-smi
        nvidia_result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if nvidia_result.returncode == 0:
            driver_version = nvidia_result.stdout.strip().split('\n')[0]
            print(f"  ✓ NVIDIA GPU detected (driver: {driver_version})")
            
            # Determine CUDA version based on driver
            try:
                major_version = int(driver_version.split('.')[0])
                if major_version >= 525:
                    cuda_index = "https://download.pytorch.org/whl/cu121"
                    cuda_version = "12.1"
                else:
                    cuda_index = "https://download.pytorch.org/whl/cu118"
                    cuda_version = "11.8"
            except:
                cuda_index = "https://download.pytorch.org/whl/cu121"
                cuda_version = "12.1"
            
            print(f"  Installing PyTorch with CUDA {cuda_version} support...")
            print("  (This may take a few minutes)")
            
            subprocess.check_call([
                str(pip_exe),
                'install',
                'torch', 'torchvision', 'torchaudio',
                '--index-url', cuda_index,
                '--force-reinstall',  # ← AJOUTE --force-reinstall
                '--no-deps'  # ← AJOUTE --no-deps pour éviter de casser les autres deps
            ])
            print(f"✓ PyTorch with CUDA {cuda_version} installed")
        else:
            print("  - No NVIDIA GPU detected, using CPU-only PyTorch")
            
    except FileNotFoundError:
        print("  - nvidia-smi not found, using CPU-only PyTorch")
    except Exception as e:
        print(f"  ⚠ Could not detect GPU: {e}")
        print("  - Keeping CPU-only PyTorch")
    
    # Test server can be imported
    print("\n6. Testing server import...")
    try:
        result = subprocess.run(
            [str(python_exe), '-c', 'import fastapi; import uvicorn; print("OK")'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and "OK" in result.stdout:
            print("✓ Server dependencies verified")
        else:
            print("⚠ Warning: Could not verify server dependencies")
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")

    except Exception as e:
        print(f"⚠ Warning: Could not test import: {e}")
    
    # Download SAM2 checkpoints
    print("\n7. Downloading SAM2 checkpoints (this may take several minutes)...")
    
    checkpoint_dir = qgeoai_dir / 'sam2_checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints = [
        ('sam2.1_hiera_tiny.pt', 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt'),
        ('sam2.1_hiera_small.pt', 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt'),
        ('sam2.1_hiera_base_plus.pt', 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt'),
        ('sam2.1_hiera_large.pt', 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'),
    ]
    
    import urllib.request
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, int(downloaded * 100 / total_size))
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r    Progress: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='', flush=True)
    
    for i, (checkpoint_name, checkpoint_url) in enumerate(checkpoints, 1):
        checkpoint_path = checkpoint_dir / checkpoint_name
        print(f"\n  [{i}/{len(checkpoints)}] {checkpoint_name}")
        
        if checkpoint_path.exists():
            size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            print(f"    ✓ Already exists ({size_mb:.1f} MB)")
        else:
            try:
                urllib.request.urlretrieve(checkpoint_url, checkpoint_path, report_progress)
                print()  # New line after progress
                size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                print(f"    ✓ Downloaded ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"\n    ⚠ Warning: Could not download: {e}")
                print("      Will attempt to download on first use.")
    
    # Summary of checkpoints
    total_size = sum(
        (checkpoint_dir / name).stat().st_size 
        for name, _ in checkpoints 
        if (checkpoint_dir / name).exists()
    ) / (1024 * 1024)
    downloaded_count = sum(1 for name, _ in checkpoints if (checkpoint_dir / name).exists())
    print(f"\n✓ SAM2 checkpoints: {downloaded_count}/{len(checkpoints)} downloaded ({total_size:.0f} MB total)")
    
    # Remind about YOLO11 models
    print("\n8. YOLO11 Models (optional)...")
    models_dir = server_dir / 'models'
    models_readme = models_dir / 'README.md'

    if models_readme.exists():
        print(f"✓ Models directory created: {models_dir}")
        print("  ℹ️  For YOLO11 training with pretrained weights:")
        print("     1. Download models from: https://docs.ultralytics.com/models/yolo11/")
        print("     2. Place .pt files in the models/ directory")
        print("     3. Compatible : yolo11.pt, yolo11-seg.pt, yolo11-obb.pt (n,s,m,l,x)")
    else:
        print("⚠ Models directory not found in source")

    # Create launcher script
    print("\n9. Creating launcher script...")
    
    if sys.platform == 'win32':
        launcher = qgeoai_dir / 'start_server.bat'
        launcher_content = f'@echo off\n"{python_exe}" "{server_dir / "server.py"}" %*\n'
    else:
        launcher = qgeoai_dir / 'start_server.sh'
        launcher_content = f'#!/bin/bash\n"{python_exe}" "{server_dir / "server.py"}" "$@"\n'
        
    launcher.write_text(launcher_content)
    
    if not sys.platform == 'win32':
        launcher.chmod(0o755)
    
    print(f"✓ Launcher created: {launcher}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Installation Complete!")
    print("=" * 60)
    print(f"\nServer installed to: {server_dir}")
    print(f"Client module: {qgeoai_dir / 'qgeoai_client.py'}")
    print(f"\nTo start the server manually:")
    print(f"  {launcher}")
    print(f"\nOr from Python:")
    print(f"  from qgeoai_client import QGeoAIClient")
    print(f"  client = QGeoAIClient()")
    print(f"  client.start_server()")
    
    return True


def main():
    """Main entry point"""
    try:
        success = install_server()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Installation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

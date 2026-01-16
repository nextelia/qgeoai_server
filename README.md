<div align="center">

# QGeoAI Server

### Local AI server for QGIS geospatial workflows

<p>
  <img src="https://img.shields.io/badge/QGIS-3.40%20LTE-589632?style=for-the-badge&logo=qgis&logoColor=white" alt="QGIS"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Release-0.9.0-blue?style=for-the-badge" alt="Release"/>
  <img src="https://img.shields.io/badge/License-GPLv2+-green?style=for-the-badge" alt="License"/>
</p>

<p>
  <a href="https://qgeoai.nextelia.fr">Documentation</a> •
  <a href="#installation">Installation</a> •
  <a href="#troubleshooting">Troubleshooting</a>
</p>

<img src="assets/logo.png" alt="QGeoAI Server" width="900"/>

---

**Local HTTP server for deep learning in QGIS - Annotation, training, inference**

</div>

## Overview

QGeoAI Server is a local server that enables QGIS plugins (QAnnotate, QModel Trainer, QPredict, QToolbox) to access AI tools (PyTorch, SAM2, YOLO) without burdening QGIS.

<div align="center">
<table>
<tr>
<td width="50%">

**Architecture Benefits**
- Dependency isolation (PyTorch, Ultralytics)
- No model reloading between operations
- QGIS stays lightweight and responsive
- Automatic CUDA support for NVIDIA GPUs

</td>
<td width="50%">

**Privacy & Security**
- 100% local (`127.0.0.1` only)
- No external connections
- No telemetry or tracking
- Token-based authentication

</td>
</tr>
</table>
</div>

## Installation

### Prerequisites

- **Python 3.10+**
- **5 GB disk space** (SAM2 models)
- **NVIDIA GPU** (optional, for CUDA acceleration)

### Quick Install
```bash
# 1. Create Python environment
# Windows
python -m venv %USERPROFILE%\.qgeoai\env

# Linux / Mac
python3 -m venv ~/.qgeoai/env

# 2. Install server
cd qgeoai_server
python install_server.py
```

**Installation process:**
1. Copies files to `~/.qgeoai/server/`
2. Installs dependencies (FastAPI, PyTorch, etc.)
3. Downloads SAM2 models (4 models, ~1.5 GB)
4. Detects and configures GPU if available
5. Creates startup scripts

**Installation time**: 10-20 minutes

**optional:**
To use YOLO11 with pretrained weights:

1. **Download**: https://docs.ultralytics.com/models/yolo11/
2. **Place** `.pt` files in: `%USERPROFILE%\.qgeoai\server\models\` (Windows) or `~/.qgeoai/server/models/` (Linux/Mac)
3. **Compatible models**:
   - Detection: `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`
   - Segmentation: `yolo11n-seg.pt`, `yolo11s-seg.pt`, etc.
   - OBB: `yolo11n-obb.pt`, `yolo11s-obb.pt`, etc.

### Verify Installation
```bash
python check_installation.py
```

Expected output:
```
✓ QGeoAI directory exists
✓ Python environment found
✓ Server files installed
✓ Client module available
✓ Dependencies installed
✓ PyTorch with CUDA support
```

## Usage

### Automatic Startup (Recommended)

The server starts automatically when you use any QGeoAI plugin in QGIS. No manual intervention needed.

### Manual Startup
```bash
# Windows
%USERPROFILE%\.qgeoai\start_server.bat

# Linux / Mac
~/.qgeoai/start_server.sh
```

### Python API
```python
from qgeoai_client import QGeoAIClient

client = QGeoAIClient()

# Auto-start if needed
if not client.is_server_running():
    client.start_server()
```

## Architecture
```
~/.qgeoai/
├── env/                    # Isolated Python environment
│   ├── Lib/               # (Windows)
│   └── bin/python         # (Linux/Mac)
├── server/                # Server code
│   ├── server.py         # FastAPI server
│   ├── endpoints/        # API by functionality
│   ├── core/             # Business logic
│   └── sam2/             # SAM2 model
├── sam2_checkpoints/     # SAM2 model weights
├── logs/                 # Server logs
├── qgeoai_client.py      # Client for plugins
├── server.token          # Security token (generated at startup)
├── server.port           # Port used (8765 by default)
└── start_server.bat/sh   # Launch script
```

## Plugins


- **QAnnotate** Dataset creation
- **QModel Trainer** Model training
- **QPredic** Inference
- **QToolbox** Processing


## Troubleshooting

<details>
<summary><b>Server won't start</b></summary>

<pre><code class="language-bash">
# Check Python environment
~/.qgeoai/env/bin/python --version  # Linux/Mac
%USERPROFILE%\.qgeoai\env\Scripts\python --version  # Windows

# Check dependencies
~/.qgeoai/env/bin/pip list | grep fastapi

# Reinstall if needed
cd qgeoai_server
python install_server.py
</code></pre>

</details>


<details>
<summary><b>Port already in use</b></summary>

The server automatically finds a free port between 8765 and 8775. Check the port used:

<pre><code class="language-bash">
cat ~/.qgeoai/server.port  # Linux/Mac
type %USERPROFILE%\.qgeoai\server.port  # Windows
</code></pre>

</details>


<details>
<summary><b>GPU not detected</b></summary>

<pre><code class="language-bash">
# Verify drivers
nvidia-smi

# Reinstall PyTorch with CUDA
cd qgeoai_server
python install_server.py  # Auto-detects GPU
</code></pre>

Note: CPU mode works but is slower for SAM2 and YOLO.

</details>


<details>
<summary><b>QGIS can't find server</b></summary>

In QGIS Python Console:

<pre><code class="language-python">
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / '.qgeoai'))

from qgeoai_client import QGeoAIClient
client = QGeoAIClient()
print(client.is_server_running())  # Should be True
</code></pre>

</details>


<details>
<summary><b>Check logs</b></summary>

<pre><code class="language-bash">
# Server logs
cat ~/.qgeoai/logs/server.log  # Linux/Mac
type %USERPROFILE%\.qgeoai\logs\server.log  # Windows

# Startup logs
cat ~/.qgeoai/logs/server_debug.log  # Linux/Mac
type %USERPROFILE%\.qgeoai\logs\server_debug.log  # Windows
</code></pre>

</details>


## Security & Privacy

**Your data never leaves your machine**

<table>
<tr>
<td width="50%">

**Security Features**
- Token authentication per session
- Binds to `127.0.0.1` (localhost only)
- No external API calls
- Open source (fully auditable)

</td>
<td width="50%">

**Safe for Sensitive Data**
- All processing is local
- No internet transmission
- No data collection
- Same security as QGIS itself

</td>
</tr>
</table>

## Support & Contribution

### Community Support

- **Documentation**: [qgeoai.nextelia.fr](https://qgeoai.nextelia.fr)
- **Issues**: [GitHub Issues](../../issues)
- **Diagnostics**: Run `python check_installation.py`

### Professional Support

For operational support, custom integrations, or training:

- **Email**: [hello@nextelia.fr](mailto:hello@nextelia.fr)
- **Website**: [nextelia.fr](https://nextelia.fr)

The suite is free and open-source. Professional services ensure sustainable development.

### Contributing

This project is developed as an open, production-grade GeoAI platform for QGIS.

We welcome:
- Bug reports with reproducible steps
- Documentation improvements
- Real-world feedback

For larger changes, new features or architectural modifications, please open an issue first to discuss alignment with the project roadmap.

The roadmap is maintained by the core team to ensure long-term stability, consistency and production readiness.

## License

<div align="center">

**QGeoAI Server** - Part of **QGeoAI Tools** suite

Developed by [**Nextelia®**](https://nextelia.fr)

This project is released under GPLv2+ to ensure that GeoAI workflows remain open, auditable and sustainable for the geospatial community.

</div>

## Acknowledgments

<table>
<tr>
<td><b>Project</b></td>
<td><b>License</b></td>
<td><b>Purpose</b></td>
</tr>
<tr>
<td><a href="https://github.com/facebookresearch/segment-anything-2">SAM2</a> (Meta AI)</td>
<td>Apache-2.0 / BSD-3</td>
<td>Segmentation</td>
</tr>
<tr>
<td><a href="https://github.com/ultralytics/ultralytics">Ultralytics</a></td>
<td>AGPL-3.0</td>
<td>YOLO framework</td>
</tr>
<tr>
<td><a href="https://pytorch.org">PyTorch</a></td>
<td>BSD-3</td>
<td>Deep learning</td>
</tr>
<tr>
<td><a href="https://fastapi.tiangolo.com">FastAPI</a></td>
<td>MIT</td>
<td>Web framework</td>
</tr>
<tr>
<td><a href="https://github.com/qubvel/segmentation_models.pytorch">segmentation-models-pytorch</a></td>
<td>MIT</td>
<td>Architectures</td>
</tr>
<tr>
<td><a href="https://github.com/DPIRD-DMA/Building-Regulariser">Building-Regulariser</a></td>
<td>MIT</td>
<td>Geometry tools</td>
</tr>
<tr>
<td><a href="https://github.com/DPIRD-DMA/Smoothify">Smoothify</a></td>
<td>MIT</td>
<td>Smoothing</td>
</tr>
</table>

---

<div align="center">

**Made for the geospatial community**

**Copyright** © 2026 Nextelia® • **Version** 0.9.0 • **Updated** January 15, 2026

</div>

  <!-- QGIS 3.40 LTE -->
  ![QGIS](https://img.shields.io/badge/QGIS-3.40%20LTE-2F6F8E?style=flat-square)

  <!-- Python >=3.10 -->
  ![Python](https://img.shields.io/badge/Python-%3E%3D3.10-F0DB4F?style=flat-square)

  <!-- Documentation -->
  [![Documentation](https://img.shields.io/badge/Documentation-Read-4CAF50?style=flat-square)](https://qgeoai.nextelia.fr/)

  <!-- Release 0.9 -->
  ![Release](https://img.shields.io/badge/Release-0.9-607D8B?style=flat-square)

  <!-- Licence GPLv2+ -->
  ![Licence](https://img.shields.io/badge/License-GPLv2+-009688?style=flat-square)


# QGeoAI Server

**Local server for QGIS QGeoAI Tools plugins** - Annotation, model training, inference and geospatial processing

--- 

<img src="assets/logo.png" alt="QGeoAI Server" width="1024"/>

## 🎯 What is it?
---

QGeoAI Server is a **local server that runs only on your computer**. It enables QGIS plugins (QAnnotate, QModel Trainer, QPredict, QToolbox) to access AI tools (PyTorch, SAM2, YOLO) without burdening QGIS.

### Why a local server?

- **Dependency isolation**: Heavy libraries (PyTorch, Ultralytics) run in a separate Python environment
- **Performance**: No need to reload models for each operation
- **Stability**: QGIS stays lightweight and responsive
- **GPU**: Automatic CUDA support if you have an NVIDIA card

### 🔒 Your data stays with you

- ✅ **100% local**: Server runs only on `127.0.0.1` (localhost)
- ✅ **No external connections**: Your data never leaves your machine
- ✅ **No tracking**: No telemetry, no data collection
- ✅ **Secure**: Unique authentication token generated at each startup

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- 5 GB disk space (for SAM2 models)
- Optional: NVIDIA GPU with drivers installed (for CUDA acceleration)

### Installation in 2 steps

#### 1. Create Python environment

```bash
# Windows
python -m venv %USERPROFILE%\.qgeoai\env

# Linux / Mac
python3 -m venv ~/.qgeoai/env
```

#### 2. Install the server

```bash
cd qgeoai_server
python install_server.py
```

**What does the installation do?**

1. Copies files to `~/.qgeoai/server/`
2. Installs required dependencies (FastAPI, PyTorch, etc.)
3. Downloads SAM2 models (4 models, ~1.5 GB total)
4. Automatically detects and configures GPU support if available
5. Creates a startup script

⏱️ **Installation time**: 10-20 minutes (depending on your connection)

### Verify installation

```bash
python check_installation.py
```

You should see:
```
✓ QGeoAI directory exists
✓ Python environment found
✓ Server files installed
✓ Client module available
✓ Dependencies installed
✓ PyTorch with CUDA support
```

---

## 🚀 Usage

### Automatic startup (recommended)

The server starts **automatically** when you use a QGeoAI plugin in QGIS. Nothing to do!

### Manual startup (if needed)

```bash
# Windows
%USERPROFILE%\.qgeoai\start_server.bat

# Linux / Mac
~/.qgeoai/start_server.sh
```

### From Python / QGIS

```python
from qgeoai_client import QGeoAIClient

client = QGeoAIClient()

# Check if server is running
if not client.is_server_running():
    client.start_server()
```

---

## 🔧 Architecture

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

---

## 🛠️ Plugins using the server

- **QAnnotate**: Create annotated datasets for model training
- **QModel Trainer**: Train deep learning models directly in QGIS
- **QPredict**: Run inference with models trained using QModel Trainer
- **QToolbox**: Collection of geospatial processing tools

---

## 🐛 Troubleshooting

### Server won't start

```bash
# Check Python environment
~/.qgeoai/env/bin/python --version  # Linux/Mac
%USERPROFILE%\.qgeoai\env\Scripts\python --version  # Windows

# Check dependencies
~/.qgeoai/env/bin/pip list | grep fastapi

# Reinstall if needed
cd qgeoai_server
python install_server.py
```

### "Port already in use"

The server automatically finds a free port between 8765 and 8775. Check the port used:

```bash
cat ~/.qgeoai/server.port  # Linux/Mac
type %USERPROFILE%\.qgeoai\server.port  # Windows
```

### GPU / CUDA error

If your GPU is not detected:

1. Verify NVIDIA drivers are installed: `nvidia-smi`
2. Reinstall PyTorch with CUDA: 
   ```bash
   cd qgeoai_server
   python install_server.py  # Automatically detects and configures
   ```
3. The server also works in CPU-only mode. However, using a GPU is highly recommended.

### QGIS plugins can't find the server

```python
# In QGIS Python Console
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / '.qgeoai'))

from qgeoai_client import QGeoAIClient
client = QGeoAIClient()
print(client.is_server_running())  # Should display True
```

### Check logs

```bash
# Server logs
cat ~/.qgeoai/logs/server.log  # Linux/Mac
type %USERPROFILE%\.qgeoai\logs\server.log  # Windows

# Startup logs (if startup problem)
cat ~/.qgeoai/logs/server_debug.log  # Linux/Mac
type %USERPROFILE%\.qgeoai\logs\server_debug.log  # Windows
```

---

## 🔐 Security and Privacy

### How does it work?

1. **Token authentication**: A unique random token is generated at each startup and stored in `~/.qgeoai/server.token`
2. **Localhost only**: Server listens on `127.0.0.1`, not `0.0.0.0` → impossible to access from the network
3. **No telemetry**: No data sent to external servers
4. **Open source**: You can inspect the code and verify it does nothing suspicious

### Can I use QGeoAI Server on sensitive data?

**Yes, absolutely.** All operations are performed locally on your machine. No data is transmitted over the Internet. It's exactly like using classic QGIS, but with better software architecture.

---

## 🤝 Support and Contribution

### Having issues?

1. Check the documentation: https://qgeoai.nextelia.fr/
2. Run `python check_installation.py` for diagnostics
3. Check logs in `~/.qgeoai/logs/`
4. Open an issue on GitHub with:
   - Your OS (Windows / Linux / Mac)
   - Relevant logs
   - Steps to reproduce the issue
5. For operational support, personalized assistance or integration into your projects, contact us directly via hello@nextelia.fr or the dedicated contact form.

⚠️ The suite is free and open-source. Professional support and custom integrations are available as paid services to ensure sustainable development.

### Contributing

This project is developed as an open, production-grade GeoAI platform for QGIS.

We welcome:
• Bug reports
• Reproducible issues
• Documentation improvements
• Feedback from real-world use cases

For larger changes, new features or architectural modifications, please open an issue first to discuss alignment with the project roadmap.

The roadmap is maintained by the core team to ensure long-term stability, consistency and production readiness.

---

## 📄 License

**QGeoAI Server** is part of the **QGeoAI Tools** suite developed by Nextelia®  
This project is released under GPLv2+ to ensure that GeoAI workflows remain open, auditable and sustainable for the geospatial community.

---

## 🙏 Acknowledgments

This project uses:

- **SAM2** (Copyright Meta AI) - Segment Anything Model 2 (Apache-2.0, BSD-3-Clause) See licenses/sam2 for details
- **Ultralytics** - YOLO framework (AGPL-3.0)
- **PyTorch** - Deep learning
- **FastAPI** - Modern web framework (MIT)
- **segmentation-models-pytorch** - Segmentation architectures (MIT)
- [Building-Regulariser](https://github.com/DPIRD-DMA/Building-Regulariser) (MIT)
- [Smoothify](https://github.com/DPIRD-DMA/Smoothify) (MIT)

---

**Copyright**: © 2026 Nextelia®  
**Version**: 0.9.0  
**Last updated**: January 15, 2026
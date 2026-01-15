# ⚡ Quick Start - QGeoAI Server

**5-minute installation!**

---

## 1️⃣ Create the environment

```bash
# Windows
python -m venv %USERPROFILE%\.qgeoai\env

# Linux / Mac
python3 -m venv ~/.qgeoai/env
```

---

## 2️⃣ Install the server

```bash
cd qgeoai_server
python install_server.py
```

☕ **Wait 10-20 minutes** (downloading SAM2 models)

---

## 3️⃣ Verify installation

```bash
python check_installation.py
```

You should see ✅ everywhere.

---

## 4️⃣ YOLO11 Models (optional)

To use YOLO11 with pretrained weights:

1. **Download**: https://docs.ultralytics.com/models/yolo11/
2. **Place** `.pt` files in: `%USERPROFILE%\.qgeoai\server\models\` (Windows) or `~/.qgeoai/server/models/` (Linux/Mac)
3. **Compatible models**:
   - Detection: `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`
   - Segmentation: `yolo11n-seg.pt`, `yolo11s-seg.pt`, etc.
   - OBB: `yolo11n-obb.pt`, `yolo11s-obb.pt`, etc.


---

## 5️⃣ That's it!

The server **starts automatically** when you use a QGeoAI plugin in QGIS.

---

## 🆘 Troubleshooting

### Startup error

```bash
# Reinstall
cd qgeoai_server
python install_server.py
```

### GPU not detected

This is normal if you don't have an NVIDIA card. The server also works in CPU mode!

### QGIS plugins can't find the server

In QGIS Python Console:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / '.qgeoai'))

from qgeoai_client import QGeoAIClient
client = QGeoAIClient()

# Start manually if needed
if not client.is_server_running():
    client.start_server()
```

---

## 📖 Full documentation

See **README.md** for all details.

---

## ❓ Quick FAQ

**Q: Is my data sent to the Internet?**  
A: No! The server runs 100% locally on your machine (localhost).

**Q: Does the server always run in the background?**  
A: Yes, until explicitly stopped. It remains active even after closing QGIS.  
⚠️ **Important**: If you close QGIS during an operation (training, prediction), it will be abruptly interrupted.

**Q: Does it consume a lot of resources?**  
A: At rest: ~500 MB RAM. In use: 2-4 GB with GPU.  
⚠️ The server may slightly slow down QGIS startup (few seconds).

**Q: Do I need to restart the server often?**  
A: No. Once installed, the server starts automatically and manages all plugins on its own.

**Q: Is GPU mandatory?**  
A: No. Everything also works on CPU, just a bit slower for SAM2 and YOLO.

---

**Need help?** → See README.md, "Troubleshooting" section
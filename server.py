"""
QGeoAI Server - Local HTTP server for QGIS plugins
Provides isolated Python environment for heavy dependencies (PyTorch, Ultralytics, etc.)
"""

import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware

from utils.port_finder import find_free_port
from utils.token_manager import TokenManager
from endpoints import toolbox, annotation, mask_export
from endpoints.qmodeltrainer import training as qmodeltrainer_training
from endpoints.qpredict import models_router, predict_router

# Configure logging to both file and console
from pathlib import Path as LogPath

log_dir = LogPath.home() / '.qgeoai' / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'server.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info("QGEOAI SERVER - LOGGING INITIALIZED")
logger.info(f"Log file: {log_dir / 'server.log'}")
logger.info("=" * 80)

# Configuration
CONFIG_DIR = Path.home() / '.qgeoai'
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Initialize token manager
token_manager = TokenManager(CONFIG_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info("QGeoAI Server starting...")
    token_manager.generate_token()
    logger.info(f"Security token generated and saved to {token_manager.token_file}")
    
    # Save port information
    port_file = CONFIG_DIR / 'server.port'
    port_file.write_text(str(app.state.port))
    logger.info(f"Server port saved to {port_file}")
    
    yield
    
    # Shutdown
    logger.info("QGeoAI Server shutting down...")
    token_manager.cleanup()


# Create FastAPI app
app = FastAPI(
    title="QGeoAI Server",
    description="Local server for QGeoAI QGIS plugins",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (localhost only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1", "http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency for token verification
def verify_token(authorization: str = Header(None)):
    """Verify the bearer token from request header"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    token = authorization.replace("Bearer ", "")
    if not token_manager.verify_token(token):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return True


# Core endpoints
@app.get("/health")
def health_check():
    """
    Health check endpoint (no authentication required)
    Used by clients to verify server is running
    """
    return {
        "status": "ok",
        "service": "qgeoai-server"
    }


@app.get("/status")
def server_status(_: bool = Depends(verify_token)):
    """
    Server status with dependency versions (requires authentication)
    """
    try:
        import torch
        import ultralytics
        import numpy as np
        
        return {
            "status": "running",
            "versions": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "ultralytics": ultralytics.__version__,
                "numpy": np.__version__,
            },
            "config_dir": str(CONFIG_DIR),
        }
    except ImportError as e:
        logger.error(f"Failed to import dependencies: {e}")
        return {
            "status": "degraded",
            "error": f"Missing dependency: {str(e)}",
            "config_dir": str(CONFIG_DIR),
        }


@app.post("/shutdown")
def shutdown(_: bool = Depends(verify_token)):
    """
    Gracefully shutdown the server (requires authentication)
    """
    logger.info("Shutdown requested via API")
    import os
    import signal
    
    # Schedule shutdown after response is sent
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "shutting_down"}


# Include routers
app.include_router(toolbox.router, prefix="/toolbox", tags=["toolbox"])
app.include_router(annotation.router, prefix="/annotate", tags=["annotation"])
app.include_router(mask_export.router, prefix="/api", tags=["mask_export"])

# QModel Trainer routers
app.include_router(qmodeltrainer_training.router)

# QPredict routers 
app.include_router(models_router, dependencies=[Depends(verify_token)])
app.include_router(predict_router, dependencies=[Depends(verify_token)])


def main():
    """Main entry point for the server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QGeoAI Local Server")
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Preferred port number (will find next available if occupied)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address (should always be 127.0.0.1 for security)"
    )
    
    args = parser.parse_args()
    
    # Security check
    if args.host != "127.0.0.1":
        logger.error("Server must run on 127.0.0.1 for security reasons")
        sys.exit(1)
    
    # Find available port
    port = find_free_port(args.port, max_attempts=10)
    if port is None:
        logger.error(f"Could not find available port starting from {args.port}")
        sys.exit(1)
    
    logger.info(f"Starting server on {args.host}:{port}")
    
    # Store port in app state for lifespan
    app.state.port = port
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        log_level="info",
        access_log=False  # Reduce noise
    )


if __name__ == "__main__":
    main()
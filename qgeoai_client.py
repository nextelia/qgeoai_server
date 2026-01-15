"""
QGeoAI Client
HTTP client for QGIS plugins to communicate with the QGeoAI server
"""

import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests
from requests.exceptions import RequestException, Timeout

logger = logging.getLogger(__name__)


class QGeoAIClient:
    """
    Client for communicating with the QGeoAI local server
    Handles server lifecycle, authentication, and API calls
    """
    
    def __init__(self, config_dir: Optional[Path] = None, timeout: int = 30):
        """
        Initialize the client
        
        Args:
            config_dir: Path to QGeoAI config directory (default: ~/.qgeoai)
            timeout: Default timeout for requests in seconds
        """
        self.config_dir = Path(config_dir) if config_dir else Path.home() / '.qgeoai'
        self.timeout = timeout
        self._base_url = None
        self._token = None
        self._server_process = None
    
    @property
    def base_url(self) -> str:
        """Get the server base URL"""
        if self._base_url is None:
            port = self._get_server_port()
            self._base_url = f"http://127.0.0.1:{port}"
        return self._base_url
    
    @property
    def token(self) -> Optional[str]:
        """Get the authentication token"""
        if self._token is None:
            self._token = self._get_token()
        return self._token
    
    @property
    def headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    def _get_server_port(self) -> int:
        """Read server port from config file, with fallback to search"""
        port_file = self.config_dir / 'server.port'
        
        # Try to read from file first
        if port_file.exists():
            try:
                port = int(port_file.read_text().strip())
                # Verify this port is actually responding
                try:
                    response = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
                    if response.status_code == 200:
                        return port
                    else:
                        logger.warning(f"Port {port} in file but server not responding")
                except:
                    logger.warning(f"Port {port} in file but not accessible")
            except (ValueError, IOError) as e:
                logger.warning(f"Failed to read port from {port_file}: {e}")
        
        # If file doesn't exist or port not responding, search for server
        logger.info("Searching for server on ports 8765-8775...")
        for port in range(8765, 8776):
            try:
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
                if response.status_code == 200:
                    logger.info(f"Found server on port {port}")
                    return port
            except:
                continue
        
        # Default to 8765 if nothing found
        logger.warning("Server not found, defaulting to port 8765")
        return 8765
    
    def _get_token(self) -> Optional[str]:
        """Read authentication token from config file"""
        token_file = self.config_dir / 'server.token'
        if token_file.exists():
            try:
                return token_file.read_text().strip()
            except IOError as e:
                logger.warning(f"Failed to read token from {token_file}: {e}")
        return None
    
    def is_server_running(self) -> bool:
        """
        Check if the server is running and responsive
        
        Returns:
            True if server is running, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=2
            )
            return response.status_code == 200
        except RequestException:
            return False
    
    def start_server(self, wait: bool = True, max_wait: int = 15) -> bool:
        """
        Start the QGeoAI server as a background process
        
        Args:
            wait: Whether to wait for server to be ready
            max_wait: Maximum seconds to wait for server startup
        
        Returns:
            True if server started successfully, False otherwise
        """
        logger.info("=" * 60)
        logger.info("START_SERVER called")
        
        if self.is_server_running():
            logger.info("✅ Server is already running - skipping start")
            logger.info("=" * 60)
            return True
        
        logger.info("Server not running, need to start it")
        
        # Path to server script
        server_script = self.config_dir / 'server' / 'server.py'
        logger.info(f"Server script path: {server_script}")
        logger.info(f"Server script exists: {server_script.exists()}")
        
        if not server_script.exists():
            logger.error(f"❌ Server script not found: {server_script}")
            logger.info("=" * 60)
            return False
        
        # Path to Python environment
        if sys.platform == 'win32':
            python_exe = self.config_dir / 'env' / 'Scripts' / 'pythonw.exe'
            if not python_exe.exists():
                # Fallback to python.exe if pythonw not found
                python_exe = self.config_dir / 'env' / 'Scripts' / 'python.exe'
        else:
            python_exe = self.config_dir / 'env' / 'bin' / 'python'
        
        logger.info(f"Python executable: {python_exe}")
        logger.info(f"Python executable exists: {python_exe.exists()}")
        
        if not python_exe.exists():
            logger.error(f"❌ Python executable not found: {python_exe}")
            logger.info("=" * 60)
            return False
        
        try:
            logger.info(f"Starting server process: {python_exe} {server_script}")
            
            # DEBUG MODE: Logs visibles dans fichier
            log_dir = self.config_dir / 'logs'
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / 'server_debug.log'
            
            logger.info(f"Server logs: {log_file}")
            
            # Use subprocess to start detached process
            if sys.platform == 'win32':
                with open(log_file, 'w') as f:
                    self._server_process = subprocess.Popen(
                        [str(python_exe), str(server_script)],
                        stdout=f,
                        stderr=subprocess.STDOUT,
                    )
            else:
                with open(log_file, 'w') as f:
                    self._server_process = subprocess.Popen(
                        [str(python_exe), str(server_script)],
                        start_new_session=True,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                    )
            
            logger.info(f"✅ Process started with PID: {self._server_process.pid}")
            
            if wait:
                logger.info(f"Waiting up to {max_wait} seconds for server to be ready...")
                start_time = time.time()
                attempt = 0
                
                while time.time() - start_time < max_wait:
                    attempt += 1
                    
                    if self.is_server_running():
                        elapsed = time.time() - start_time
                        logger.info(f"✅ Server is now responding (after {elapsed:.1f}s, {attempt} attempts)")
                        logger.info("=" * 60)
                        return True
                    
                    time.sleep(0.5)
                
                elapsed = time.time() - start_time
                logger.error(f"❌ Server did not start within {max_wait} seconds")
                logger.error(f"Made {attempt} connection attempts over {elapsed:.1f}s")
                logger.error("Check server logs at: C:\\Users\\ludov\\.qgeoai\\logs\\server.log")
                logger.info("=" * 60)
                return False
            
            logger.info("✅ Process started (not waiting for ready)")
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start server: {e}", exc_info=True)
            logger.info("=" * 60)
            return False
    
    def stop_server(self) -> bool:
        """
        Stop the QGeoAI server gracefully
        
        Returns:
            True if server stopped successfully, False otherwise
        """
        try:
            response = requests.post(
                f"{self.base_url}/shutdown",
                headers=self.headers,
                timeout=5
            )
            return response.status_code == 200
        except RequestException as e:
            logger.error(f"Failed to stop server: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get server status and dependency versions
        
        Returns:
            Status dictionary
        """
        try:
            response = requests.get(
                f"{self.base_url}/status",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"Failed to get status: {e}")
            raise
    
    # Toolbox operations
    def regularize_buildings(
        self,
        input_path: str,
        output_path: str,
        **params
    ) -> Dict[str, Any]:
        """
        Regularize building geometries
        
        Args:
            input_path: Path to input GeoJSON file
            output_path: Path to output GeoJSON file
            **params: Additional regularization parameters
        
        Returns:
            Response dictionary with status and output path
        """
        try:
            response = requests.post(
                f"{self.base_url}/toolbox/regularize",
                headers=self.headers,
                json={
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "params": params
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"Regularization failed: {e}")
            raise
    
    def resample_raster(
        self,
        input_path: str,
        output_path: str,
        target_resolution: float,
        resampling_method: str = 'bilinear'
    ) -> Dict[str, Any]:
        """
        Resample a raster to a different resolution
        
        Args:
            input_path: Path to input raster file
            output_path: Path to output raster file
            target_resolution: Target resolution in raster units
            resampling_method: Resampling method (nearest, bilinear, cubic)
        
        Returns:
            Response dictionary with status and output information
        """
        try:
            response = requests.post(
                f"{self.base_url}/toolbox/resample",
                headers=self.headers,
                json={
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "target_resolution": target_resolution,
                    "resampling_method": resampling_method
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"Resampling failed: {e}")
            raise

    def smoothify_geometries(
        self,
        input_path: str,
        output_path: str,
        segment_length: Optional[float] = None,
        smooth_iterations: int = 3,
        merge_collection: bool = True,
        merge_field: Optional[str] = None,
        merge_multipolygons: bool = True,
        preserve_area: bool = True,
        area_tolerance: float = 0.01,
        num_cores: int = 0
    ) -> Dict[str, Any]:
        """
        Smooth polygon or line geometries
        
        Args:
            input_path: Path to input vector file
            output_path: Path to output vector file
            segment_length: Segment length in map units (None = auto-detect)
            smooth_iterations: Number of smoothing iterations (3-5 recommended)
            merge_collection: Merge adjacent geometries before smoothing
            merge_field: Field name to group geometries for merging
            merge_multipolygons: Merge adjacent polygons within MultiPolygons
            preserve_area: Preserve original area (polygons only)
            area_tolerance: Area preservation tolerance percentage
            num_cores: Number of CPU cores (0 = all available)
        
        Returns:
            Response dictionary with status and output information
        """
        try:
            response = requests.post(
                f"{self.base_url}/toolbox/smoothify",
                headers=self.headers,
                json={
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "segment_length": segment_length,
                    "smooth_iterations": smooth_iterations,
                    "merge_collection": merge_collection,
                    "merge_field": merge_field,
                    "merge_multipolygons": merge_multipolygons,
                    "preserve_area": preserve_area,
                    "area_tolerance": area_tolerance,
                    "num_cores": num_cores
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"Smoothing failed: {e}")
            raise

    # Placeholder methods for future phases
    def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start a training job (Phase 2)"""
        raise NotImplementedError("Training endpoints will be implemented in Phase 2")
    
    def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """Get training job status (Phase 2)"""
        raise NotImplementedError("Training endpoints will be implemented in Phase 2")
    
    def cancel_training(self, job_id: str) -> Dict[str, Any]:
        """Cancel a training job (Phase 2)"""
        raise NotImplementedError("Training endpoints will be implemented in Phase 2")
    
    def run_inference(self, model_path: str, input_path: str, **params) -> Dict[str, Any]:
        """Run model inference (Phase 3)"""
        raise NotImplementedError("Prediction endpoints will be implemented in Phase 3")
    
    
    # =========================================================================
    # SAM2 Annotation operations
    # =========================================================================
    
    def sam2_status(self) -> Dict[str, Any]:
        """
        Get SAM2 model status
        
        Returns:
            Status dictionary with loaded state, model type, device info
        """
        try:
            response = requests.get(
                f"{self.base_url}/annotate/sam2/status",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"SAM2 status failed: {e}")
            raise
    
    def sam2_load(
        self,
        model_type: str = "sam2.1_hiera_tiny",
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load SAM2 model into server memory
        
        Args:
            model_type: Model variant (sam2.1_hiera_tiny, sam2.1_hiera_small, 
                       sam2.1_hiera_base_plus, sam2.1_hiera_large)
            checkpoint_path: Custom checkpoint path (None = auto-download)
        
        Returns:
            Response with status and device info
        """
        try:
            response = requests.post(
                f"{self.base_url}/annotate/sam2/load",
                headers=self.headers,
                json={
                    "model_type": model_type,
                    "checkpoint_path": checkpoint_path
                },
                timeout=120  # Loading can take time, especially with download
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"SAM2 load failed: {e}")
            raise
    
    def sam2_predict(
        self,
        image_path: str,
        crop_x: int,
        crop_y: int,
        crop_width: int,
        crop_height: int,
        point_x: Optional[int] = None,
        point_y: Optional[int] = None,
        point_label: int = 1,
        box: Optional[List[int]] = None,
        simplify_tolerance: float = 2.0,
        rgb_bands: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Run SAM2 prediction on image crop
        
        Args:
            image_path: Path to raster image
            crop_x: Crop left pixel coordinate
            crop_y: Crop top pixel coordinate  
            crop_width: Crop width in pixels
            crop_height: Crop height in pixels
            point_x: Point X in crop coordinates (optional)
            point_y: Point Y in crop coordinates (optional)
            point_label: 1=positive, 0=negative
            box: Bounding box [x_min, y_min, x_max, y_max] in crop coords (optional)
            simplify_tolerance: Polygon simplification tolerance
            rgb_bands: RGB band indices (1-based), e.g. [1,2,3]
        
        Returns:
            Response with polygon coordinates in crop space
        """
        try:
            payload = {
                "image_path": str(image_path),
                "crop_x": crop_x,
                "crop_y": crop_y,
                "crop_width": crop_width,
                "crop_height": crop_height,
                "point_label": point_label,
                "simplify_tolerance": simplify_tolerance,
            }
            
            # Add optional parameters
            if point_x is not None:
                payload["point_x"] = point_x
            if point_y is not None:
                payload["point_y"] = point_y
            if box is not None:
                payload["box"] = box
            if rgb_bands is not None:
                payload["rgb_bands"] = rgb_bands
            
            response = requests.post(
                f"{self.base_url}/annotate/sam2/predict",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"SAM2 predict failed: {e}")
            raise

    def sam2_unload(self) -> Dict[str, Any]:
        """
        Unload SAM2 model and free GPU memory
        
        Returns:
            Response with status
        """
        try:
            response = requests.post(
                f"{self.base_url}/annotate/sam2/unload",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"SAM2 unload failed: {e}")
            raise
    
    # =========================================================================
    # SAM2 Session operations (incremental refinement)
    # =========================================================================
    
    def sam2_session_start(
        self,
        image_path: str,
        crop_x: int,
        crop_y: int,
        crop_width: int,
        crop_height: int,
        simplify_tolerance: float = 2.0,
        rgb_bands: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Start an interactive SAM2 session - encodes the crop once
        
        Args:
            image_path: Path to raster image
            crop_x: Crop left pixel coordinate
            crop_y: Crop top pixel coordinate
            crop_width: Crop width in pixels
            crop_height: Crop height in pixels
            simplify_tolerance: Polygon simplification tolerance
            rgb_bands: RGB band indices (1-based)
        
        Returns:
            Response with status
        """
        try:
            payload = {
                "image_path": str(image_path),
                "crop_x": crop_x,
                "crop_y": crop_y,
                "crop_width": crop_width,
                "crop_height": crop_height,
                "simplify_tolerance": simplify_tolerance,
            }
            if rgb_bands is not None:
                payload["rgb_bands"] = rgb_bands
            
            response = requests.post(
                f"{self.base_url}/annotate/sam2/session/start",
                headers=self.headers,
                json=payload,
                timeout=60  # Encoding can take time
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"SAM2 session start failed: {e}")
            raise
    
    def sam2_session_add_point(
        self,
        point_x: int,
        point_y: int,
        is_positive: bool = True
    ) -> Dict[str, Any]:
        """
        Add a point to the current session and get refined mask
        
        Args:
            point_x: Point X in crop coordinates
            point_y: Point Y in crop coordinates
            is_positive: True for positive point, False for negative
        
        Returns:
            Response with polygon and point count
        """
        try:
            response = requests.post(
                f"{self.base_url}/annotate/sam2/session/add_point",
                headers=self.headers,
                json={
                    "point_x": point_x,
                    "point_y": point_y,
                    "is_positive": is_positive
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"SAM2 add point failed: {e}")
            raise
    
    def sam2_session_preview(
        self,
        point_x: int,
        point_y: int,
        is_positive: bool = True
    ) -> Dict[str, Any]:
        """
        Preview mask with additional point WITHOUT committing
        
        Args:
            point_x: Point X in crop coordinates
            point_y: Point Y in crop coordinates
            is_positive: True for positive point, False for negative
        
        Returns:
            Response with polygon (session state unchanged)
        """
        try:
            response = requests.post(
                f"{self.base_url}/annotate/sam2/session/preview",
                headers=self.headers,
                json={
                    "point_x": point_x,
                    "point_y": point_y,
                    "is_positive": is_positive
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"SAM2 preview failed: {e}")
            raise
    
    def sam2_session_undo(self) -> Dict[str, Any]:
        """
        Remove last point and recompute mask
        
        Returns:
            Response with polygon and updated point count
        """
        try:
            response = requests.post(
                f"{self.base_url}/annotate/sam2/session/undo",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"SAM2 undo failed: {e}")
            raise
    
    def sam2_session_update_simplification(self, tolerance: float) -> Dict[str, Any]:
        """
        Update simplification tolerance and get recomputed polygon
        
        Args:
            tolerance: New simplification tolerance
        
        Returns:
            Response with polygon using new tolerance
        """
        try:
            response = requests.post(
                f"{self.base_url}/annotate/sam2/session/update_simplification",
                headers=self.headers,
                json={"tolerance": tolerance},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"SAM2 update simplification failed: {e}")
            raise
    
    def sam2_session_end(self) -> Dict[str, Any]:
        """
        End the current interactive session
        
        Returns:
            Response with status
        """
        try:
            response = requests.post(
                f"{self.base_url}/annotate/sam2/session/end",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"SAM2 session end failed: {e}")
            raise
    
    def sam2_predict_point(
        self,
        image_path: str,
        crop_x: int,
        crop_y: int,
        crop_width: int,
        crop_height: int,
        point_x: int,
        point_y: int,
        point_label: int = 1,
        simplify_tolerance: float = 2.0,
        rgb_bands: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Single point prediction without session (for hover preview)
        
        Returns:
            Response with polygon
        """
        try:
            payload = {
                "image_path": str(image_path),
                "crop_x": crop_x,
                "crop_y": crop_y,
                "crop_width": crop_width,
                "crop_height": crop_height,
                "point_x": point_x,
                "point_y": point_y,
                "point_label": point_label,
                "simplify_tolerance": simplify_tolerance,
            }
            if rgb_bands is not None:
                payload["rgb_bands"] = rgb_bands
            
            response = requests.post(
                f"{self.base_url}/annotate/sam2/predict_point",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"SAM2 predict point failed: {e}")
            raise
    
    def sam2_predict_box(
        self,
        image_path: str,
        crop_x: int,
        crop_y: int,
        crop_width: int,
        crop_height: int,
        box: List[int],
        simplify_tolerance: float = 2.0,
        rgb_bands: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Bounding box prediction
        
        Args:
            box: [x_min, y_min, x_max, y_max] in crop coordinates
        
        Returns:
            Response with polygon
        """
        try:
            payload = {
                "image_path": str(image_path),
                "crop_x": crop_x,
                "crop_y": crop_y,
                "crop_width": crop_width,
                "crop_height": crop_height,
                "box": box,
                "simplify_tolerance": simplify_tolerance,
            }
            if rgb_bands is not None:
                payload["rgb_bands"] = rgb_bands
            
            response = requests.post(
                f"{self.base_url}/annotate/sam2/predict_box",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"SAM2 predict box failed: {e}")
            raise

    # =========================================================================
    # Mask Export operations
    # =========================================================================
    
    def check_mask_export_available(self) -> bool:
        """
        Check if PNG mask export is available on server.
        
        Returns:
            True if available, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/mask_export/status",
                headers=self.headers,
                timeout=5
            )
            response.raise_for_status()
            result = response.json()
            return result.get('available', False)
        except RequestException:
            return False
    
    def create_png_mask(
        self,
        width: int,
        height: int,
        features: List[Dict[str, Any]],
        x_offset: float,
        y_offset: float,
        pixel_width: float,
        pixel_height: float
    ) -> Dict[str, Any]:
        """
        Create a PNG mask from vector polygons via server.
        
        Args:
            width: Mask width in pixels
            height: Mask height in pixels
            features: List of polygon features with:
                - class_id: int
                - exterior_points: List[List[float]] (list of [x, y])
                - holes: List[List[List[float]]] (optional, list of rings)
            x_offset: Geographic X offset (left edge)
            y_offset: Geographic Y offset (top edge)
            pixel_width: Pixel width in geographic units
            pixel_height: Pixel height in geographic units
        
        Returns:
            Response dictionary with:
                - success: bool
                - mask_base64: str (base64-encoded PNG)
                - has_annotations: bool
                - unique_classes: List[int]
                - shape: List[int] ([height, width])
        """
        try:
            payload = {
                "width": width,
                "height": height,
                "features": features,
                "x_offset": x_offset,
                "y_offset": y_offset,
                "pixel_width": pixel_width,
                "pixel_height": pixel_height
            }
            
            response = requests.post(
                f"{self.base_url}/api/create_png_mask",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"PNG mask creation failed: {e}")
            raise

def get_qgeoai_client(**kwargs) -> QGeoAIClient:
    """
    Factory function to get a QGeoAI client instance
    Useful for QGIS plugins to get a configured client
    
    Returns:
        Configured QGeoAIClient instance
    """
    return QGeoAIClient(**kwargs)
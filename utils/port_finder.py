"""
Port finder utility
Finds available ports for the QGeoAI server
"""

import socket
import logging

logger = logging.getLogger(__name__)


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """
    Check if a port is available for binding
    
    Args:
        port: Port number to check
        host: Host address (default: 127.0.0.1)
    
    Returns:
        True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
    except OSError:
        return False


def find_free_port(start_port: int = 8765, max_attempts: int = 10, host: str = "127.0.0.1") -> int | None:
    """
    Find an available port starting from start_port
    
    Args:
        start_port: Starting port number (default: 8765)
        max_attempts: Maximum number of ports to try (default: 10)
        host: Host address (default: 127.0.0.1)
    
    Returns:
        Available port number or None if no port found
    """
    for offset in range(max_attempts):
        port = start_port + offset
        if is_port_available(port, host):
            logger.info(f"Found available port: {port}")
            return port
        else:
            logger.debug(f"Port {port} is not available, trying next...")
    
    logger.error(f"Could not find available port in range {start_port}-{start_port + max_attempts - 1}")
    return None


def get_server_port_from_file(config_dir) -> int | None:
    """
    Read the server port from the config file
    
    Args:
        config_dir: Path to config directory
    
    Returns:
        Port number or None if file doesn't exist
    """
    from pathlib import Path
    
    port_file = Path(config_dir) / 'server.port'
    if port_file.exists():
        try:
            return int(port_file.read_text().strip())
        except (ValueError, IOError) as e:
            logger.error(f"Failed to read port from {port_file}: {e}")
            return None
    return None

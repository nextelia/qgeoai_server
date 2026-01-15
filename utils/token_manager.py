"""
Token manager for security
Generates and manages authentication tokens
"""

import secrets
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TokenManager:
    """Manages security tokens for server authentication"""
    
    def __init__(self, config_dir: Path):
        """
        Initialize token manager
        
        Args:
            config_dir: Path to configuration directory
        """
        self.config_dir = Path(config_dir)
        self.token_file = self.config_dir / 'server.token'
        self._token = None
    
    def generate_token(self) -> str:
        """
        Generate a new random token and save it to file
        
        Returns:
            Generated token string
        """
        self._token = secrets.token_hex(32)
        
        try:
            self.token_file.write_text(self._token)
            # Make token file readable only by owner (Unix systems)
            if hasattr(self.token_file, 'chmod'):
                self.token_file.chmod(0o600)
            logger.info(f"Token generated and saved to {self.token_file}")
        except IOError as e:
            logger.error(f"Failed to save token to {self.token_file}: {e}")
            raise
        
        return self._token
    
    def get_token(self) -> str | None:
        """
        Get the current token (from memory or file)
        
        Returns:
            Token string or None if not available
        """
        if self._token:
            return self._token
        
        if self.token_file.exists():
            try:
                self._token = self.token_file.read_text().strip()
                return self._token
            except IOError as e:
                logger.error(f"Failed to read token from {self.token_file}: {e}")
                return None
        
        return None
    
    def verify_token(self, token: str) -> bool:
        """
        Verify if provided token matches the current token
        
        Args:
            token: Token to verify
        
        Returns:
            True if token is valid, False otherwise
        """
        current_token = self.get_token()
        if not current_token:
            logger.warning("No token available for verification")
            return False
        
        # Use secrets.compare_digest for timing-attack resistance
        return secrets.compare_digest(token, current_token)
    
    def cleanup(self):
        """Clean up token file on shutdown"""
        if self.token_file.exists():
            try:
                self.token_file.unlink()
                logger.info(f"Token file removed: {self.token_file}")
            except IOError as e:
                logger.error(f"Failed to remove token file: {e}")

import yaml
import os
from typing import Any, Dict
from logs.logging_setup import logger

class ConfigurationManager:
    """
    handles runtime loading and retrieval of project configuration.
    """
    
    def __init__(self, file_name: str = "config/mcp_config.yaml"):
        """
        initializes the configuration manager.

        args:
            file_name (str): the name of the yaml configuration file.
        """
        self._path = os.path.join(os.path.dirname(__file__), file_name)
        self._registry: Dict[str, Any] = self._initialize_registry()

    def _initialize_registry(self) -> Dict[str, Any]:
        """
        loads configuration from the filesystem into memory.

        returns:
            dict[str, any]: the loaded configuration registry.
        """
        if not os.path.exists(self._path):
            logger.warning(f"configuration source missing: {self._path}")
            return {}

        try:
            with open(self._path, "r") as stream:
                data = yaml.safe_load(stream)
                logger.info(f"configuration synchronized from {os.path.basename(self._path)}")
                return data if data is not None else {}
        except Exception as error:
            logger.error(f"synchronization failure: {error}")
            return {}

    def fetch(self, query: str, fallback: Any = None) -> Any:
        """
        retrieves a value from the registry using dot notation.

        args:
            query (str): the dot-notated path to the value.
            fallback (any): the default value if the key is not found.

        returns:
            any: the retrieved value or fallback.
        """
        segments = query.split('.')
        cursor = self._registry
        
        for segment in segments:
            if isinstance(cursor, dict):
                cursor = cursor.get(segment)
            else:
                return fallback
            
            if cursor is None:
                return fallback
                
        return cursor

    @property
    def registry(self) -> Dict[str, Any]:
        """
        returns the full configuration registry.

        returns:
            dict[str, any]: the complete registry.
        """
        return self._registry


server_config = ConfigurationManager("config/mcp_config.yaml")
tools_config = server_config   # copying needs fix

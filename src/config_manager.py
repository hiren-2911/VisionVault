# config_manager.py
import json
import logging
import logging.config
from typing import Dict, Any, Union
from pathlib import Path



class ConfigManager:
    """Manages application configuration"""
    
    def __init__(self, config_path: str, logger_config_path:str, application_name: str, console_log:bool=False, log_dir:str = None):
        self.config_path = Path(config_path).as_posix()
        self.logger_config_path = Path(logger_config_path).as_posix()
        self.application_name = application_name
        self.console_log = console_log

        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._load_logger()
        self.logger = logging.getLogger(__name__)
        self._config = self._load_config()
        

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    def _load_logger(self):
        try:
            with open(self.logger_config_path, 'r') as f:
                logger_conf = json.load(f)
                if self.console_log:
                    for key, val in logger_conf['loggers'].items():
                        val['handlers'].append('console')
                if 'handlers' in logger_conf:
                    for handler_name, handler_config in logger_conf['handlers'].items():
                        if 'filename' in handler_config:
                            # Get the original filename
                            original_filename = handler_config['filename']
                            # Create full path with custom log directory
                            full_path = self.log_dir / original_filename
                            handler_config['filename'] = str(full_path)
                logging.config.dictConfig(logger_conf)
        except Exception as e:
            self.logger.error(f'error while loading config:{e}')
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value with application-specific override"""
        try:
            applications = self._config.get('applications', [])
            
            # Look for application-specific config
            for app in applications:
                if app.get('application_name') == self.application_name:
                    app_config = app.get(key, {})
                    global_config = self._config.get(key, {})
                    
                    if isinstance(app_config, dict) and isinstance(global_config, dict):
                        return {**global_config, **app_config}
                    elif app_config:
                        return app_config
                    elif global_config:
                        return global_config
                    else:
                        return default
            
            # Fallback to global config
            return self._config.get(key, default)
        except Exception as e:
            self.logger.error(f"Error getting config for key '{key}': {e}")
            return default
    
    @property
    def use_gpu(self) -> bool:
        return self.get_config("use_gpu", False)
    
    @property
    def yolo_threshold(self) -> float:
        return self.get_config("yolo_threshold", 0.4)
    
    @property
    def device(self) -> str:
        return 'cuda' if self.use_gpu else 'cpu'

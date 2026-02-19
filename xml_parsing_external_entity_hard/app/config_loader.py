#!/usr/bin/env python3
"""
Configuration Loader Module

This module handles loading and parsing XML configuration files at service startup.
It provides functions to load configuration from XML files and retrieve specific settings.

WARNING: This code is intentionally vulnerable for demonstration purposes.
"""

import xml.dom.minidom
import logging
import os
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration cache
CONFIG_CACHE: Dict[str, Any] = {}


def load_config(config_file_path: str) -> Dict[str, Any]:
    """
    Load configuration from an XML file.
    
    This function parses an XML configuration file and extracts settings
    into a dictionary format for easy access throughout the application.
    
    Args:
        config_file_path (str): Path to the XML configuration file
        
    Returns:
        Dict[str, Any]: Dictionary containing configuration key-value pairs
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        xml.parsers.expat.ExpatError: If the XML is malformed
    """
    global CONFIG_CACHE
    
    logger.info(f"Loading configuration from: {config_file_path}")
    
    # Check if file exists
    if not os.path.exists(config_file_path):
        logger.error(f"Configuration file not found: {config_file_path}")
        raise FileNotFoundError(f"Config file not found: {config_file_path}")
    
    try:
        # Parse the XML file using minidom (VULNERABLE to XXE)
        # This allows DTD processing and external entity resolution
        dom = xml.dom.minidom.parse(config_file_path)
        
        config_dict = {}
        
        # Get the root element
        root = dom.documentElement
        
        # Extract configuration settings from XML elements
        for setting_node in root.getElementsByTagName('setting'):
            if setting_node.hasAttribute('name'):
                name = setting_node.getAttribute('name')
                
                # Get the text content of the setting
                value = ""
                for child in setting_node.childNodes:
                    if child.nodeType == child.TEXT_NODE:
                        value += child.data
                
                value = value.strip()
                
                # Try to convert to appropriate type
                if value.isdigit():
                    value = int(value)
                elif value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                
                config_dict[name] = value
                logger.debug(f"Loaded setting: {name} = {value}")
        
        # Cache the configuration
        CONFIG_CACHE = config_dict
        
        logger.info(f"Successfully loaded {len(config_dict)} configuration settings")
        return config_dict
        
    except xml.parsers.expat.ExpatError as e:
        logger.error(f"XML parsing error in config file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def get_setting(config_dict: Dict[str, Any], setting_name: str, default: Any = None) -> Any:
    """
    Retrieve a specific setting from the configuration dictionary.
    
    Args:
        config_dict (Dict[str, Any]): Configuration dictionary
        setting_name (str): Name of the setting to retrieve
        default (Any, optional): Default value if setting not found. Defaults to None.
        
    Returns:
        Any: The setting value or default if not found
    """
    value = config_dict.get(setting_name, default)
    
    if value is None and default is None:
        logger.warning(f"Setting '{setting_name}' not found and no default provided")
    
    return value


def reload_config(config_file_path: str) -> Dict[str, Any]:
    """
    Reload configuration from file, updating the cache.
    
    Args:
        config_file_path (str): Path to the XML configuration file
        
    Returns:
        Dict[str, Any]: Updated configuration dictionary
    """
    logger.info("Reloading configuration...")
    return load_config(config_file_path)


def get_cached_config() -> Dict[str, Any]:
    """
    Get the currently cached configuration.
    
    Returns:
        Dict[str, Any]: Cached configuration dictionary
    """
    return CONFIG_CACHE.copy()


if __name__ == "__main__":
    # Example usage
    try:
        config = load_config("config.xml")
        print(f"Loaded configuration: {config}")
        
        # Example of retrieving specific settings
        db_host = get_setting(config, "database_host", "localhost")
        db_port = get_setting(config, "database_port", 5432)
        timeout = get_setting(config, "timeout", 30)
        
        print(f"Database Host: {db_host}")
        print(f"Database Port: {db_port}")
        print(f"Timeout: {timeout}")
        
    except Exception as e:
        print(f"Error: {e}")
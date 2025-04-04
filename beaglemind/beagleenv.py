import os
import logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

class BeagleEnv:
    """
    Utility class for managing environment variables for BeagleMind applications.
    Provides methods to load, access, and validate environment variables from .env files.
    """
    
    _env_loaded = False
    
    @classmethod
    def load_env_file(cls, env_path=None):
        """
        Load environment variables from a .env file.
        
        Args:
            env_path (str, optional): Path to the .env file. If None, will search for .env files.
        
        Returns:
            bool: True if environment variables were loaded successfully, False otherwise.
        """
        if cls._env_loaded:
            logging.info("Environment variables already loaded")
            return True
            
        try:
            # If path is provided, use it; otherwise find .env file
            if env_path:
                env_file = Path(env_path)
                if not env_file.exists():
                    logging.warning(f"Specified .env file not found at {env_path}")
                    return False
                env_path = str(env_file)
            else:
                env_path = find_dotenv(usecwd=True)
                if not env_path:
                    logging.warning("No .env file found in current directory or parent directories")
                    return False
            
            # Load the environment variables
            loaded = load_dotenv(env_path)
            if loaded:
                logging.info(f"Environment variables loaded from {env_path}")
                cls._env_loaded = True
                return True
            else:
                logging.warning(f"Failed to load environment variables from {env_path}")
                return False
                
        except Exception as e:
            logging.error(f"Error loading environment variables: {e}")
            return False
    
    @classmethod
    def get_env(cls, var_name, default=None):
        """
        Get an environment variable with optional default value.
        
        Args:
            var_name (str): Name of the environment variable to retrieve
            default: Default value to return if the variable is not found
            
        Returns:
            The value of the environment variable or the default value
        """
        if not cls._env_loaded:
            cls.load_env_file()
            
        return os.environ.get(var_name, default)
    
    @classmethod
    def validate_required_env(cls, required_vars):
        """
        Validate that all required environment variables are present.
        
        Args:
            required_vars (list): List of required environment variable names
            
        Returns:
            tuple: (bool, list) - Success status and list of missing variables
        """
        if not cls._env_loaded:
            cls.load_env_file()
            
        missing = []
        for var in required_vars:
            if not cls.get_env(var):
                missing.append(var)
                
        return len(missing) == 0, missing
    
    @classmethod
    def get_env_or_raise(cls, var_name):
        """
        Get an environment variable or raise an exception if it's not found.
        
        Args:
            var_name (str): Name of the environment variable to retrieve
            
        Returns:
            str: The value of the environment variable
            
        Raises:
            ValueError: If the environment variable is not found
        """
        value = cls.get_env(var_name)
        if value is None:
            raise ValueError(f"Required environment variable '{var_name}' not found")
        return value
import logging
import os
from dotenv import load_dotenv


class BeagleEnv:
    """Centralized environment variable management"""
    
    @staticmethod
    def load_env_file(env_path: str = '.env'):
        """
        Load environment variables from a specified .env file.
        
        Args:
            env_path (str): Path to the .env file. Defaults to '.env'
        
        Raises:
            ValueError: If required keys are missing
        """
        # Load environment variables
        load_dotenv(env_path)
        
        # Validate critical environment variables
        required_keys = [
            'GROQ_API_KEY', 
            'OPENAI_API_KEY', 
            'OPENAI_AZURE_ENDPOINT',
            'OPENAI_API_VERSION'
        ]
        
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        
        if missing_keys:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
        
        logging.info("Environment variables loaded successfully")
    
    @staticmethod
    def get_env(key: str, default: str = None) -> str:
        """
        Safely retrieve an environment variable.
        
        Args:
            key (str): Environment variable name
            default (str, optional): Default value if key is not found
        
        Returns:
            str: Environment variable value
        """
        value = os.getenv(key, default)
        if value is None:
            logging.warning(f"Environment variable {key} not found")
        return value
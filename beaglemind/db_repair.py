import os
import logging
import shutil
from pathlib import Path
from typing import Optional

class VectorDBRepair:
    """
    Utility class for diagnosing and repairing Chroma vector database issues.
    """
    
    @staticmethod
    def repair_database(db_path: str) -> bool:
        """
        Repair a corrupted Chroma vector database.
        
        Args:
            db_path: Path to the vector database directory
            
        Returns:
            bool: True if repair was successful, False otherwise
        """
        db_path = Path(db_path)
        if not db_path.exists():
            logging.warning(f"Database directory does not exist at {db_path}")
            try:
                db_path.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created new database directory at {db_path}")
                return True
            except Exception as e:
                logging.error(f"Failed to create database directory: {e}")
                return False
        
        # Backup existing database files if they exist
        backup_path = db_path.with_name(f"{db_path.name}_backup_{VectorDBRepair._get_timestamp()}")
        try:
            if db_path.exists() and any(db_path.iterdir()):
                shutil.copytree(db_path, backup_path)
                logging.info(f"Created database backup at {backup_path}")
        except Exception as e:
            logging.warning(f"Failed to create backup: {e}")
        
        # Remove problematic SQLite files
        try:
            sqlite_file = db_path / "chroma.sqlite3"
            if sqlite_file.exists():
                sqlite_file.unlink()
                logging.info(f"Removed corrupted SQLite file: {sqlite_file}")
            
            journal_file = db_path / "chroma.sqlite3-journal"
            if journal_file.exists():
                journal_file.unlink()
                logging.info(f"Removed SQLite journal file: {journal_file}")
            
            # Also check for and remove WAL and SHM files which can cause issues
            wal_file = db_path / "chroma.sqlite3-wal"
            if wal_file.exists():
                wal_file.unlink()
                logging.info(f"Removed SQLite WAL file: {wal_file}")
                
            shm_file = db_path / "chroma.sqlite3-shm"
            if shm_file.exists():
                shm_file.unlink()
                logging.info(f"Removed SQLite SHM file: {shm_file}")
            
            return True
        except Exception as e:
            logging.error(f"Error during database repair: {e}")
            return False
        
    @staticmethod
    def clean_install(db_path: str) -> bool:
        """
        Perform a clean installation by removing the entire database directory.
        Use with caution as this will delete all existing data.
        
        Args:
            db_path: Path to the vector database directory
            
        Returns:
            bool: True if clean install was successful, False otherwise
        """
        db_path = Path(db_path)
        if not db_path.exists():
            db_path.mkdir(parents=True, exist_ok=True)
            return True
            
        try:
            # Create backup before clean install with more unique timestamp
            from datetime import datetime
            import uuid
            
            # Use microseconds and a UUID to ensure uniqueness
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            unique_id = uuid.uuid4().hex[:8]
            backup_name = f"{db_path.name}_backup_{timestamp}_{unique_id}"
            backup_path = db_path.with_name(backup_name)
            
            # Make sure the backup path doesn't exist
            counter = 0
            original_backup_path = backup_path
            while backup_path.exists():
                counter += 1
                backup_path = original_backup_path.with_name(f"{original_backup_path.name}_{counter}")
            
            if db_path.exists() and any(db_path.iterdir()):
                shutil.copytree(db_path, backup_path)
                logging.info(f"Created database backup at {backup_path}")
            
            # Remove entire directory and recreate it
            shutil.rmtree(db_path)
            db_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Performed clean installation of database at {db_path}")
            return True
        except Exception as e:
            logging.error(f"Clean installation failed: {e}")
            return False
    
    @staticmethod
    def validate_database(db_path: str) -> bool:
        """
        Validate if the database directory and structure appear to be valid.
        
        Args:
            db_path: Path to the vector database directory
            
        Returns:
            bool: True if database appears valid, False otherwise
        """
        db_path = Path(db_path)
        if not db_path.exists():
            logging.warning(f"Database directory does not exist at {db_path}")
            return False
            
        sqlite_file = db_path / "chroma.sqlite3"
        if not sqlite_file.exists():
            logging.warning(f"SQLite file not found at {sqlite_file}")
            return False
            
        # Check if file is empty
        if sqlite_file.stat().st_size == 0:
            logging.warning(f"SQLite file exists but is empty at {sqlite_file}")
            return False
            
        return True
    
    @staticmethod
    def _get_timestamp() -> str:
        """Get a timestamp string for backup naming with microseconds for uniqueness"""
        from datetime import datetime
        import uuid
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        unique_id = uuid.uuid4().hex[:8]
        return f"{timestamp}_{unique_id}"
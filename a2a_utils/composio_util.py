import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class TemporaryComposioCwd:
    """
    A context manager to temporarily change the current working directory (CWD)
    to a writable temporary path (e.g., /tmp) for operations that might
    create lock files (like .composio.lock) in the CWD.

    This is useful in read-only file system environments like Vercel.
    """
    def __init__(self, temp_dir: str = "/tmp", lock_file_name: str = ".composio.lock"):
        self.temp_dir = temp_dir
        self.lock_file_name = lock_file_name
        self.original_cwd: Optional[str] = None
        self.pid = os.getpid() # For logging

    def __enter__(self):
        # Only change CWD if the current CWD is not writable and temp_dir is.
        current_process_cwd = os.getcwd()
        if not os.access(current_process_cwd, os.W_OK) and \
           os.path.isdir(self.temp_dir) and \
           os.access(self.temp_dir, os.W_OK):
            self.original_cwd = current_process_cwd
            os.chdir(self.temp_dir)
            logger.info(
                f"[PID-{self.pid}] ContextManager: Temporarily changed CWD from {self.original_cwd} to {self.temp_dir} "
                f"for potential '{self.lock_file_name}' creation."
            )
        else:
            logger.info(
                f"[PID-{self.pid}] ContextManager: CWD ({current_process_cwd}) is already writable or temp_dir ({self.temp_dir}) is not suitable. CWD not changed."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_cwd:
            # Check if lock file was created in temp_dir (for logging)
            # Current CWD should be self.temp_dir if it was changed
            lock_file_path_in_temp_dir = os.path.join(os.getcwd(), self.lock_file_name)
            if os.path.exists(lock_file_path_in_temp_dir): # Check in current CWD (which should be temp_dir)
                logger.info(
                    f"[PID-{self.pid}] ContextManager: Lock file '{self.lock_file_name}' "
                    f"found in {os.getcwd()} (which was temp_dir)."
                )
            else:
                logger.warning(
                    f"[PID-{self.pid}] ContextManager: Lock file '{self.lock_file_name}' "
                    f"NOT found in {os.getcwd()} (which was temp_dir). This might be okay."
                )
            
            os.chdir(self.original_cwd)
            logger.info(f"[PID-{self.pid}] ContextManager: Restored CWD to {self.original_cwd}.")
        
        # Do not suppress exceptions, let them propagate
        return False

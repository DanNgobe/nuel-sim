"""
Centralized warning suppression for RLLib and Ray warnings
"""
import os
import sys
import warnings
import logging
from io import StringIO

# Set environment variables before any imports
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1" 
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_SILENCE_IMPORT_WARNING"] = "1"

# Suppress all warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

# Set logging levels to ERROR to reduce output
root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR)

# Set specific loggers to ERROR
for logger_name in ["ray", "ray.rllib", "ray.tune", "ray.worker", "gymnasium"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

def suppress_ray_warnings():
    """Suppress Ray-specific warning outputs by redirecting stderr temporarily"""
    
    class FilteredOutput:
        def __init__(self, original_stream):
            self.original_stream = original_stream
            self.buffer = StringIO()
            
        def write(self, text):
            # Filter out specific warning patterns
            text_lower = text.lower()
            
            # Define patterns to filter
            filter_patterns = [
                'warning', 'deprecated', 'deprecationwarning', 'userwarning',
                'api stack', 'unifiedlogger', 'jsonlogger', 'csvlogger', 'tbxlogger',
                'overriding environment', 'rlmodule(config', 'install gputil',
                'you are running ppo on the new api', 'this api is deprecated',
                'the `', 'interface is deprecated'
            ]
            
            # If it's a warning line, don't write it
            if any(pattern in text_lower for pattern in filter_patterns):
                return len(text)  # Return length as if written
            
            # Write to original stream
            return self.original_stream.write(text)
            
        def flush(self):
            self.original_stream.flush()
            
        def __getattr__(self, name):
            # Delegate all other attributes to the original stream
            return getattr(self.original_stream, name)
    
    # Only apply filtering if not already applied
    if not hasattr(sys.stderr, '_original_stream'):
        sys.stderr = FilteredOutput(sys.stderr)
        
    # Also override the warning function
    original_warn = warnings.warn
    def null_warn(*args, **kwargs):
        pass
    warnings.warn = null_warn

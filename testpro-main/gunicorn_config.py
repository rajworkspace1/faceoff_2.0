import multiprocessing 
import os 
# Basic Settings 
bind = "0.0.0.0:8000"  # Bind to all interfaces on port 8000 
# Adjust the number of workers (based on your CPU cores and load) 
workers = 60  # Typically CPU cores * 2 + 1, adjust based on performance 
timeout = 600  # Long timeout for GPU-based model processing 
preload_app = True
graceful_timeout = 650  # Ensures longer tasks have time to finish 
max_requests = 1000  # Restart workers periodically to prevent memory leaks 
max_requests_jitter = 50  # Randomize max_requests to avoid simultaneous restarts 
# Environment Variables for CUDA Optimization 
raw_env = [ 
"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",  # Reduces memory fragmentation 
#"CUDA_VISIBLE_DEVICES=0"  # Use the first GPU, change if needed 
] 
# Logging (optional) 
#accesslog = "-"  # Log access to stdout 
#errorlog = "-"   # Log errors to stdout 
#loglevel = "info"  # Set log level to "debug" for more detailed logs 
# Worker class for better async handling (recommended for IO-heavy or long processes) 
worker_class = "gthread" 
threads = 4  # Adjust based on your application's concurrency needs
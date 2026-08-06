import os
from deepagents.backends import LocalShellBackend

backend = LocalShellBackend(
    root_dir=".",
    virtual_mode=True,
    env={"PATH": r"C:\Users\Sreekar\Desktop\amazon-review-agent\.venv\Scripts;C:\Windows\System32;C:\Windows"},
)

# source file
local_file = r"C:\Users\Sreekar\Desktop\amazon-review-agent\data\processed\All_Beauty.parquet"

# validate source file
if not os.path.exists(local_file):
    raise FileNotFoundError(f"File not found: {local_file}")

# read file contents
with open(local_file, "rb") as f:
    parquet_bytes = f.read()

# upload dataset
backend.upload_files([
    ("/root/data/All_Beauty.parquet", parquet_bytes)
])

print("Upload completed successfully!")

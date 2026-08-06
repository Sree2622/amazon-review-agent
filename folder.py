from pathlib import Path

# Root project folder
root = Path("amazon-review-agent")

# List of folders to create
folders = [
    "app",
    "configs",
    "data/raw",
    "data/interim",
    "data/processed",
    "data/train",
    "data/validation",
    "data/test",
    "preprocessing",
    "ml",
    "llm",
    "embeddings",
    "database",
    "memory",
    "agent",
    "tools",
    "tests",
    "docs",
    "logs",
]

# Create folders
for folder in folders:
    (root / folder).mkdir(parents=True, exist_ok=True)

# Create empty files
files = [
    "pyproject.toml",
    "README.md",
    ".gitignore",
]

for file in files:
    (root / file).touch(exist_ok=True)

print(f"Project structure '{root}' created successfully!")
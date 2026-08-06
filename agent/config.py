import os

# dataset paths
PRODUCT_PATH = r"C:\Users\Sreekar\Desktop\amazon-review-agent\data\processed\All_Beauty_products.parquet"
REVIEW_PATH = r"C:\Users\Sreekar\Desktop\amazon-review-agent\data\processed\All_Beauty.parquet"

# persistent state paths
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH = os.path.join(_BASE_DIR, "data", "memory.json")
DASHBOARD_PATH = os.path.join(_BASE_DIR, "data", "dashboard.json")

# dataset columns
PRODUCT_COLS = ["parent_asin", "title", "store", "average_rating", "price"]
REVIEW_COLS = ["parent_asin", "rating", "text", "title"]

# model settings
MODEL_NAME = "gemini-3.5-flash-lite"
MAX_NEW_TOKENS = 320

# agent prompt settings
INCLUDE_DF_IN_PROMPT = False
AGENT_HEAD_ROWS = 3
AGENT_MAX_ITERATIONS = 8
AGENT_MAX_EXECUTION_TIME = 170

# debug mode
AGENT_DEBUG = os.environ.get("AGENT_DEBUG") == "1"

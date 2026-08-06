"""
Run with: python -m agent.diagnose

Checks device placement and raw generation speed, isolated from the agent loop.
"""
import time
from .model import build_chat_llm

print("Building model...")
t0 = time.time()
chat_llm = build_chat_llm()
print(f"Model built in {time.time() - t0:.1f}s")

model = chat_llm.llm.pipeline.model
devices = {str(p.device) for p in model.parameters()}
print(f"Parameter devices in use: {devices}")
if devices == {"cpu"}:
    print("WARNING: model is running entirely on CPU. This is almost certainly your slowdown.")

print("\nGenerating a short chat response...")
t0 = time.time()
out = chat_llm.invoke("What is 2 + 2? Answer in one short sentence.")
elapsed = time.time() - t0
print(f"Output: {out.content!r}")
print(f"Single-call generation took {elapsed:.1f}s")

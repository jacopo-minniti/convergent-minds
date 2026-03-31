import re
import os
from pathlib import Path

# Path to a known problematic story
huth_dir = Path("/scratch/jacopo04/convergent-minds/.convminds-home/data/huth")
tg_path = huth_dir / "derivatives" / "TextGrids" / "exorcism.TextGrid"

print(f"--- Diagnosing: {tg_path} ---")

if not tg_path.exists():
    print(f"ERROR: File not found at {tg_path}")
    sys.exit(1)

try:
    # 1. Check raw bytes for encoding/BOM
    with open(tg_path, "rb") as f:
        raw = f.read(100)
        print(f"Raw bytes (first 100): {raw}")
    
    # 2. Try reading as UTF-8
    with open(tg_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
        print(f"Content length: {len(content)} chars")
        print("--- Head (first 500 chars) ---")
        print(content[:500])
        print("--- Tail (last 100 chars) ---")
        print(content[-100:])
        
    # 3. Test the regex split
    item_blocks = re.split(r'item\s*\[\s*\d+\s*\]\s*:?', content)
    print(f"\nRegex split results: {len(item_blocks)} blocks found.")
    
    # 4. Check for 'item' manually
    item_indices = [m.start() for m in re.finditer(r'item', content, re.IGNORECASE)]
    print(f"Manual 'item' search (case-insensitive): {len(item_indices)} occurrences at indices {item_indices[:10]}")
    
    # 5. Check if 'ooTextFile' is there
    if "ooTextFile" in content:
        print("Confirmed: 'ooTextFile' header found.")
    else:
        print("WARNING: 'ooTextFile' header NOT found. File might be binary or differently formatted.")

except Exception as e:
    print(f"Diagnostic failed: {e}")

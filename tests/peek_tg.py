import os
from pathlib import Path

# Path from the user's logs
huth_dir = Path("/scratch/jacopo04/convergent-minds/.convminds-home/data/huth")
tg_dir = huth_dir / "derivatives" / "TextGrids"

print(f"Checking TextGrids in {tg_dir}")
if tg_dir.exists():
    files = list(tg_dir.glob("*.TextGrid"))
    if files:
        f = files[0]
        print(f"File: {f}")
        try:
            with open(f, "r") as h:
                content = h.read(1000)
                print("Content Head:")
                print(content)
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print("No .TextGrid files found.")
else:
    print("Directory NOT found.")

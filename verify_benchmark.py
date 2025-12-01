
from brainscore import load_benchmark
try:
    load_benchmark('Pereira2018.243sentences-partialr2')
    print("Benchmark loaded successfully!")
except Exception as e:
    print(f"Failed to load benchmark: {e}")

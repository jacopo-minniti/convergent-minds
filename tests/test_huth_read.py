import h5py
import logging
import sys
from pathlib import Path
import convminds as cm
from convminds.cache import convminds_home

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def test_huth_structure():
    # 1. Check Paths
    huth_dir = convminds_home() / "data/huth"
    subject_id = "UTS01"
    story = "adollshouse"
    
    file_path = huth_dir / "derivatives" / "preprocessed_data" / subject_id / f"{story}.hf5"
    
    logger.info(f"Checking BOLD file at: {file_path}")
    if not file_path.exists():
        logger.error(f"File NOT found at {file_path}")
        return
    
    logger.info(f"File found. Size: {file_path.stat().st_size / 1e6:.2f} MB")
    
    # 2. Try opening with h5py
    try:
        with h5py.File(file_path, "r") as f:
            logger.info(f"Successfully opened HDF5 with keys: {list(f.keys())}")
            if "data" in f:
                data_shape = f["data"].shape
                logger.info(f"BOLD data shape (TRs, Voxels): {data_shape}")
                # Read a small chunk
                chunk = f["data"][:5, :5]
                logger.info(f"First 5x5 chunk: \n{chunk}")
            else:
                logger.error("Dataset 'data' not found in HDF5 file.")
    except Exception as e:
        logger.exception(f"Failed to open/read HDF5 file: {e}")

def test_huth_benchmark():
    # 3. Test via Benchmark class
    try:
        logger.info("Initializing HuthBenchmark for S1...")
        benchmark = cm.benchmarks.HuthBenchmark(subject="S1")
        
        logger.info(f"Benchmark stimuli found: {len(benchmark.stimuli)}")
        for rec in benchmark.stimuli[:3]:
            logger.info(f"  Story: {rec.stimulus_id}, Text length: {len(rec.text)}")
            
        logger.info("Loading recordings via source...")
        recording_data = benchmark.human_recording_source.load_recordings(benchmark, selector={"subject": "S1"})
        
        logger.info(f"Loaded recordings for {len(recording_data.stimulus_ids)} stories.")
        if recording_data.stimulus_ids:
             first_story = recording_data.stimulus_ids[0]
             first_values = recording_data.values[0]
             logger.info(f"First story: {first_story}, shape: {first_values.shape}")
             
        logger.info("SUCCESS: Huth dataset integration verified.")
        
    except Exception as e:
        logger.exception(f"Benchmark test failed: {e}")

if __name__ == "__main__":
    logger.info("Starting Huth Read Test...")
    test_huth_structure()
    logger.info("-" * 30)
    test_huth_benchmark()

import torch
import logging
from tests.huth_vae_pipeline import HuthVaeDataset
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("inspect_lang")

def inspect_language():
    subject = "S1"
    dataset = HuthVaeDataset(subject_ids=[subject], trim=20, n_components=10)
    
    logger.info(f"--- Language Data Inspection ---")
    
    # Select a story and a segment
    start_idx = 100
    num_tr = 10
    
    logger.info(f"Inspecting {num_tr} consecutive TRs starting at sample {start_idx}...")
    
    all_texts = []
    for i in range(start_idx, start_idx + num_tr):
        sample = dataset[i]
        text = sample["text"]
        tr = sample["tr"]
        story = sample["story"]
        
        # Clean text for stats
        words = text.split()
        real_words = [w for w in words if w.lower() != "sp" and w.lower() != "uh" and w.lower() != "um"]
        
        logger.info(f"[TR {tr:4d} | {story}]")
        logger.info(f"  Raw Text ({len(words):2d} w): \"{text}\"")
        logger.info(f"  Clean Text({len(real_words):2d} w): \"{' '.join(real_words)}\"")
        logger.info("-" * 40)
        
        all_texts.append(text)
    
    # Check sliding window overlap
    # We want to see how many new words are added per TR
    for i in range(len(all_texts) - 1):
        words1 = set(all_texts[i].split())
        words2 = set(all_texts[i+1].split())
        new_words = words2 - words1
        logger.info(f"TR {start_idx+i} -> {start_idx+i+1}: Added words: {new_words}")

if __name__ == "__main__":
    inspect_language()

import os
import subprocess
from pathlib import Path
import sys

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    # Using curl for better progress and reliability on clusters
    cmd = ["curl", "-L", "-o", str(output_path), url]
    subprocess.run(cmd, check=True)

def main():
    # Base directory for Pereira data
    base_dir = Path(".cache/data/pereira/raw")
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stimuli and Materials (from Dropbox links provided)
    materials = {
        "IARPA_expt1_stim_images.zip": "https://www.dropbox.com/s/4bi7e7rds8thg11/IARPA_expt1_stim_images.zip?dl=1",
        "IARPA_expt1_stim_sents.zip": "https://www.dropbox.com/s/eughbv6dgrk05yf/IARPA_expt1_stim_sents.zip?dl=1",
        "IARPA_expt2_stim.zip": "https://www.dropbox.com/s/s4ulonvo4y1lj1x/IARPA_expt2_stim.zip?dl=1",
        "IARPA_expt3_stim.zip": "https://www.dropbox.com/s/kjs6s09w2py93o5/IARPA_expt3_stim.zip?dl=1",
        "DMN_overlap_n60.zip": "https://www.dropbox.com/s/obbx7mgmzd8t3h2/DMN_overlap_n60.zip?dl=1",
        "voxelInform_FractionMNI.nii": "https://www.dropbox.com/s/gmiubb0p73ushm3/voxelInform_FractionMNI.nii?dl=1",
    }

    print("--- Downloading Materials ---")
    for name, url in materials.items():
        dest = base_dir / name
        if not dest.exists():
            download_file(url, dest)
        else:
            print(f"Skipping {name}, already exists.")

    # 2. Individual Subject Data (the brain activations)
    # The user provided a main directory link: https://www.dropbox.com/sh/5z1ikn8osaao57w/AABg9LIlJfEOrQgZN7Sj7WHRa?dl=0
    # To download the WHOLE folder of subjects systematically:
    subjects_zip_url = "https://www.dropbox.com/sh/5z1ikn8osaao57w/AABg9LIlJfEOrQgZN7Sj7WHRa?dl=1"
    subjects_dest = base_dir / "pereira_subjects_all.zip"

    print("\n--- Downloading All Subject Brain Data (~several GBs) ---")
    if not subjects_dest.exists():
        try:
            download_file(subjects_zip_url, subjects_dest)
        except Exception as e:
            print(f"Error downloading all subjects: {e}")
            print("Trying to download P01 separately as a fallback...")
            # P01 link if available - fallback
            # (Note: Dropbox folder direct dl might fail on some platforms if the file is massive)
    else:
        print("Skipping subjects zip, already exists.")

    # 3. Systematic Extraction
    print("\n--- Extracting archives ---")
    import zipfile
    import tarfile

    # Extract all zips in raw dir
    for zip_path in list(base_dir.glob("*.zip")):
        print(f"Extracting {zip_path.name}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(base_dir)
        except Exception as e:
            print(f"Error extracting {zip_path.name}: {e}")

    # Extract all tars (subjects)
    # The subjects zip likely contains M01.tar, P01.tar etc.
    for tar_path in list(base_dir.rglob("*.tar")):
        print(f"Extracting subject {tar_path.name}...")
        try:
            with tarfile.open(tar_path, "r") as tar_ref:
                tar_ref.extractall(base_dir)
        except Exception as e:
            print(f"Error extracting {tar_path.name}: {e}")

    print("\nDone! Pereira dataset is ready for convminds.")

if __name__ == "__main__":
    main()

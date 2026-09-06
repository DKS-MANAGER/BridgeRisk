"""
Download FHWA NBI data for Maine (ME), Hawaii (HI), and Delaware (DE) (2021–2025).

Author: Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur
GitHub: https://github.com/DKS-MANAGER
"""

import hashlib
import os
import shutil
import zipfile

import requests

STATES = ["ME", "HI", "DE"]
YEARS = [2021, 2022, 2023, 2024, 2025]
BASE_URL = "https://www.fhwa.dot.gov/bridge/nbi"

DOWNLOAD_DIR = "data/raw/downloads"
EXTRACT_DIR = "data/raw/extracted"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_zip(file_info):
    year = file_info["year"]
    state = file_info["state"]
    disclaimer_url = file_info["disclaimer_url"]
    zip_url = file_info["zip_url"]
    zip_filename = file_info["zip_filename"]
    inner_filename = file_info["inner_filename"]
    final_filename = file_info["final_filename"]

    download_path = os.path.join(DOWNLOAD_DIR, zip_filename)
    extract_dir_year = os.path.join(EXTRACT_DIR, str(year))

    print(f"\n--- {year} {state} NBI Data (ZIP) ---")
    print(f"Disclaimer URL: {disclaimer_url}")
    print(f"ZIP URL: {zip_url}")

    session = requests.Session()
    print("  Accessing disclaimer page to obtain session cookie...")
    resp = session.get(disclaimer_url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to access disclaimer page: HTTP {resp.status_code}")

    print("  Downloading ZIP...")
    resp = session.get(zip_url, stream=True, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download ZIP: HTTP {resp.status_code}")

    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    total = 0
    with open(download_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                total += len(chunk)

    size_mb = total / (1024 * 1024)
    checksum = sha256_file(download_path)
    print(f"  Saved to: {download_path}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  SHA256: {checksum}")

    os.makedirs(extract_dir_year, exist_ok=True)
    print(f"  Extracting {inner_filename}...")
    with zipfile.ZipFile(download_path, "r") as zf:
        if inner_filename in zf.namelist():
            zf.extract(inner_filename, extract_dir_year)
        else:
            # Case insensitive search
            matching = [n for n in zf.namelist() if n.lower() == inner_filename.lower()]
            if matching:
                zf.extract(matching[0], extract_dir_year)
                extracted_file = os.path.join(extract_dir_year, matching[0])
                final_file = os.path.join(extract_dir_year, final_filename)
                if extracted_file != final_file:
                    shutil.move(extracted_file, final_file)
            else:
                raise FileNotFoundError(f"{inner_filename} not found in {zip_filename}")

    extracted_file = os.path.join(extract_dir_year, inner_filename)
    final_file = os.path.join(extract_dir_year, final_filename)
    if extracted_file != final_file and os.path.exists(extracted_file):
        shutil.move(extracted_file, final_file)

    print(f"  Extracted to: {final_file}")
    return download_path, final_file


def download_delimited(file_info):
    year = file_info["year"]
    state = file_info["state"]
    disclaimer_url = file_info["disclaimer_url"]
    data_url = file_info["data_url"]
    filename = file_info["filename"]

    download_path = os.path.join(DOWNLOAD_DIR, filename)
    extract_path = os.path.join(EXTRACT_DIR, str(year), filename)

    print(f"\n--- {year} {state} NBI Data (Delimited) ---")
    print(f"Disclaimer URL: {disclaimer_url}")
    print(f"Data URL: {data_url}")

    session = requests.Session()
    print("  Accessing disclaimer page to obtain session cookie...")
    resp = session.get(disclaimer_url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to access disclaimer page: HTTP {resp.status_code}")

    print("  Downloading data...")
    resp = session.get(data_url, stream=True, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download data: HTTP {resp.status_code}")

    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    total = 0
    with open(download_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                total += len(chunk)

    size_mb = total / (1024 * 1024)
    checksum = sha256_file(download_path)
    print(f"  Saved to: {download_path}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  SHA256: {checksum}")

    os.makedirs(os.path.dirname(extract_path), exist_ok=True)
    shutil.copy(download_path, extract_path)
    print(f"  Extracted to: {extract_path}")
    return download_path, extract_path


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    for state in STATES:
        for year in YEARS:
            if year in [2021, 2022, 2023, 2024]:
                file_info = {
                    "year": year,
                    "state": state,
                    "disclaimer_url": f"{BASE_URL}/disclaim.cfm?nbiYear={year}del&nbiZip=zip",
                    "zip_url": f"{BASE_URL}/{year}del.zip",
                    "zip_filename": f"{year}del.zip",
                    "inner_filename": f"{state.lower()}{str(year)[-2:]}.txt",
                    "final_filename": f"{state}{str(year)[-2:]}.txt",
                }
                try:
                    download_zip(file_info)
                except Exception as e:
                    print(f"Error downloading {state} {year}: {e}")
            elif year == 2025:
                file_info = {
                    "year": year,
                    "state": state,
                    "disclaimer_url": f"{BASE_URL}/disclaim.cfm?nbiYear=2025/delimited&nbiState={state}25",
                    "data_url": f"{BASE_URL}/2025/delimited/{state}25.txt",
                    "filename": f"{state}25.txt",
                }
                try:
                    download_delimited(file_info)
                except Exception as e:
                    print(f"Error downloading {state} 2025: {e}")

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()

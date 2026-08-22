"""
Download FHWA NBI data for Maine (2023–2025).

Author: Divyansh Kumar Singh (DKS) · M.Tech Civil Engineering (Hydraulic), IIT Kanpur
GitHub: https://github.com/DKS-MANAGER
"""

import hashlib
import os
import shutil
import zipfile

import requests

STATE = "ME"
BASE_URL = "https://www.fhwa.dot.gov/bridge/nbi"

DOWNLOAD_DIR = "data/raw/downloads"
EXTRACT_DIR = "data/raw/extracted"

FILES = [
    {
        "year": 2023,
        "disclaimer_url": f"{BASE_URL}/disclaim.cfm?nbiYear=2023del&nbiZip=zip",
        "zip_url": f"{BASE_URL}/2023del.zip",
        "zip_filename": "2023del.zip",
        "inner_filename": "ME23.txt",
        "final_filename": "ME23.txt",
    },
    {
        "year": 2024,
        "disclaimer_url": f"{BASE_URL}/disclaim.cfm?nbiYear=2024del&nbiZip=zip",
        "zip_url": f"{BASE_URL}/2024del.zip",
        "zip_filename": "2024del.zip",
        "inner_filename": "ME24.txt",
        "final_filename": "ME24.txt",
    },
    {
        "year": 2025,
        "disclaimer_url": f"{BASE_URL}/disclaim.cfm?nbiYear=2025/delimited&nbiState=ME25",
        "data_url": f"{BASE_URL}/2025/delimited/ME25.txt",
        "filename": "ME25.txt",
    },
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_zip(file_info):
    year = file_info["year"]
    disclaimer_url = file_info["disclaimer_url"]
    zip_url = file_info["zip_url"]
    zip_filename = file_info["zip_filename"]
    inner_filename = file_info["inner_filename"]
    final_filename = file_info["final_filename"]

    download_path = os.path.join(DOWNLOAD_DIR, zip_filename)

    print(f"\n--- {year} Maine NBI Data (ZIP) ---")
    print(f"Disclaimer URL: {disclaimer_url}")
    print(f"ZIP URL: {zip_url}")

    session = requests.Session()
    print("  Accessing disclaimer page to obtain session cookie...")
    resp = session.get(disclaimer_url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to access disclaimer page: HTTP {resp.status_code}")

    print("  Downloading ZIP...")
    resp = session.get(zip_url, stream=True, timeout=300)
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
    print(f"  Saved ZIP to: {download_path}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  SHA256: {checksum}")

    print(f"  Extracting {inner_filename}...")
    with zipfile.ZipFile(download_path, "r") as z:
        if inner_filename not in z.namelist():
            raise RuntimeError(f"{inner_filename} not found in ZIP")
        z.extract(inner_filename, os.path.join(EXTRACT_DIR, str(year)))

    extracted_file = os.path.join(EXTRACT_DIR, str(year), inner_filename)
    final_file = os.path.join(EXTRACT_DIR, str(year), final_filename)
    if extracted_file != final_file:
        shutil.move(extracted_file, final_file)

    print(f"  Extracted to: {final_file}")
    return download_path, final_file


def download_delimited(file_info):
    year = file_info["year"]
    disclaimer_url = file_info["disclaimer_url"]
    data_url = file_info["data_url"]
    filename = file_info["filename"]

    download_path = os.path.join(DOWNLOAD_DIR, filename)
    extract_path = os.path.join(EXTRACT_DIR, str(year), filename)

    print(f"\n--- {year} Maine NBI Data (Delimited) ---")
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
    shutil.copy2(download_path, extract_path)
    print(f"  Extracted to: {extract_path}")

    return download_path, extract_path


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    for file_info in FILES:
        year = file_info["year"]

        if "zip_url" in file_info:
            download_path = os.path.join(DOWNLOAD_DIR, file_info["zip_filename"])
            if os.path.exists(download_path):
                print(f"\n--- {year} Maine NBI Data (ZIP) ---")
                print(f"WARNING: {download_path} already exists.")
                print("Skipping download. Delete the file manually to re-download.")
                continue
            try:
                download_zip(file_info)
            except Exception as e:
                print(f"ERROR downloading {year} data: {e}")
        else:
            download_path = os.path.join(DOWNLOAD_DIR, file_info["filename"])
            if os.path.exists(download_path):
                print(f"\n--- {year} Maine NBI Data (Delimited) ---")
                print(f"WARNING: {download_path} already exists.")
                print("Skipping download. Delete the file manually to re-download.")
                continue
            try:
                download_delimited(file_info)
            except Exception as e:
                print(f"ERROR downloading {year} data: {e}")


if __name__ == "__main__":
    main()

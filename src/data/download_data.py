from pathlib import Path
import zipfile
import requests
from tqdm import tqdm

ZENODO_RECORD_ID = "15619247"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = DATA_DIR / "TIDS.zip"


def get_zenodo_zip_url() -> str:
    r = requests.get(ZENODO_API, timeout=60)
    r.raise_for_status()
    data = r.json()

    files = data.get("files", [])
    if not files:
        raise RuntimeError("No files found in Zenodo record")

    zip_candidates = [f for f in files if f.get("key", "").lower().endswith(".zip")]
    if not zip_candidates:
        keys = [f.get("key") for f in files]
        raise RuntimeError(f"No .zip file found. Available files: {keys}")

    
    f0 = zip_candidates[0]
    
    url = f0.get("links", {}).get("self") or f0.get("links", {}).get("download")
    if not url:
        raise RuntimeError("No download link found for zip file")
    return url


def download_file(url: str, dest: Path):
    with requests.get(url, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status()

        total = int(r.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, "wb") as f, tqdm(
            desc=f"Downloading {dest.name}",
            total=total if total > 0 else None,
            unit="iB",
            unit_scale=True,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
                    if total > 0:
                        bar.update(len(chunk))


def is_zip_valid(zip_path: Path) -> bool:
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            bad = z.testzip()
            return bad is None
    except zipfile.BadZipFile:
        return False


def unzip_file(zip_path: Path, extract_to: Path):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    url = get_zenodo_zip_url()
    print(f"Found zip URL: {url}")

    if ZIP_PATH.exists():
        if is_zip_valid(ZIP_PATH):
            print("ZIP already exists and looks valid, skipping download")
        else:
            print("Existing ZIP is invalid, re-downloading")
            ZIP_PATH.unlink()
            download_file(url, ZIP_PATH)
    else:
        download_file(url, ZIP_PATH)

    if not is_zip_valid(ZIP_PATH):
        raise RuntimeError(
            "Downloaded file is not a valid zip. "
            "Most likely the URL did not return the dataset file."
        )

    print("Extracting dataset")
    unzip_file(ZIP_PATH, DATA_DIR)
    print("Done")
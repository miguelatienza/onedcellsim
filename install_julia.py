import os
import platform
import urllib.request
import tarfile
import zipfile
import shutil
import sys

def install_julia(version="1.6.7", install_dir="venv"):
    system = platform.system().lower()
    arch = platform.machine()

    # Map system and architecture to Julia binaries
    if system == "linux" and arch == "x86_64":
        filename = f"julia-{version}-linux-x86_64.tar.gz"
    elif system == "darwin" and arch == "x86_64":
        filename = f"julia-{version}-mac64.dmg"
    elif system == "darwin" and arch == "arm64":
        filename = f"julia-{version}-mac64-arm64.dmg"
    elif system == "windows" and arch in {"AMD64", "x86_64"}:
        filename = f"julia-{version}-win64.zip"
    else:
        raise RuntimeError(f"Unsupported platform: {system} {arch}")

    # Construct URL and paths
    base_url = "https://julialang-s3.julialang.org/bin"
    url = f"{base_url}/{system}/{arch}/1.6/{filename}"
    target_path = os.path.join(install_dir, filename)
    extract_path = os.path.join(install_dir, f"julia-{version}")

    # Create installation directory
    os.makedirs(install_dir, exist_ok=True)

    # Download Julia archive
    print(f"Downloading Julia {version} from {url}...")
    urllib.request.urlretrieve(url, target_path)
    print(f"Downloaded {filename} to {target_path}")

    # Extract archive
    print(f"Extracting {filename}...")
    if filename.endswith(".tar.gz"):
        with tarfile.open(target_path, "r:gz") as tar:
            tar.extractall(install_dir)
    elif filename.endswith(".zip"):
        with zipfile.ZipFile(target_path, "r") as zip_ref:
            zip_ref.extractall(install_dir)
    else:
        raise RuntimeError("Unsupported archive format. Install manually.")

    # Clean up
    os.remove(target_path)
    print(f"Extracted Julia to {extract_path}")

    # Add to PATH (optional, for current session)
    julia_bin = os.path.join(extract_path, "bin")
    if system == "windows":
        julia_bin = os.path.join(extract_path, "bin", "julia.exe")
    if os.path.exists(julia_bin):
        os.environ["PATH"] = f"{julia_bin};{os.environ['PATH']}" if system == "windows" else f"{julia_bin}:{os.environ['PATH']}"
        print(f"Added {julia_bin} to PATH.")
    else:
        print("Julia bin directory not found.")

    # Verify installation
    print("Verifying Julia installation...")
    if shutil.which("julia"):
        os.system("julia --version")
    else:
        print("Julia installation complete, but could not find 'julia' in PATH.")

# Entry point
if __name__ == "__main__":
    try:
        install_julia()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

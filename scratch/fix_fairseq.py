# scratch/fix_fairseq.py
import os
import subprocess
import shutil
import urllib.request

# Configuration
VERSION = "0.12.1"
TARBALL = f"fairseq-{VERSION}.tar.gz"
URL = f"https://files.pythonhosted.org/packages/source/f/fairseq/{TARBALL}"
EXTRACT_DIR = f"fairseq-{VERSION}"

def run(cmd):
    print(f"> Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

try:
    # 1. Download directly from PyPI Source URL
    print(f"Downloading fairseq {VERSION} from {URL}...")
    urllib.request.urlretrieve(URL, TARBALL)

    # 2. Extract
    print("Extracting...")
    run(f"tar -xvf {TARBALL}")

    # 3. Patch
    print("Patching fairseq (adding version.txt and removing C++ extensions)...")
    version_file = os.path.join(EXTRACT_DIR, "fairseq", "version.txt")
    with open(version_file, "w") as f:
        f.write(VERSION + "\n")

    # Remove C++ extensions from setup.py to bypass compiler errors
    setup_py = os.path.join(EXTRACT_DIR, "setup.py")
    with open(setup_py, "r") as f:
        content = f.read()
    
    # Nuclear Option: Replace setup.py with a minimal pure-Python version
    print("Nuclear Option: Replacing setup.py with a minimal version...")
    setup_py = os.path.join(EXTRACT_DIR, "setup.py")
    minimal_setup = f"""
from setuptools import setup, find_packages

setup(
    name="fairseq",
    version="{VERSION}",
    packages=find_packages(exclude=["scripts", "tests"]),
    description="Minimal pure-python fairseq for transliteration",
    python_requires=">=3.7",
    install_requires=[
        "cffi",
        "cython",
        "hydra-core>=1.0.7,<1.1",
        "omegaconf>=2.0.5,<2.1",
        "numpy",
        "regex",
        "sacrebleu>=1.4.12",
        "torch",
        "tqdm",
        "bitarray",
        "torchaudio>=0.8.0",
    ],
    zip_safe=False,
)
"""
    with open(setup_py, "w") as f:
        f.write(minimal_setup)

    # 4. Resolve Dependency Hell Manually
    print("Downgrading pip temporarily to bypass metadata validation...")
    venv_python = os.path.join(os.getcwd(), ".venv", "bin", "python")
    run(f"{venv_python} -m pip install 'pip<24.1'")

    print("Installing dependencies manually to bypass conflicts...")
    venv_pip = os.path.join(os.getcwd(), ".venv", "bin", "pip")
    
    # Install problematic dependencies first with --no-deps
    run(f"{venv_pip} install omegaconf==2.0.6 hydra-core==1.0.7 --no-deps")
    run(f"{venv_pip} install portalocker sacrebleu bitarray")

    # 5. Install patched fairseq with --no-deps (this bypasses the conflict check)
    print("Installing patched fairseq...")
    venv_python = os.path.join(os.getcwd(), ".venv", "bin", "python")
    run(f"{venv_python} -m pip install ./{EXTRACT_DIR} --no-deps")

    # 6. Finally, install ai4bharat
    print("Installing ai4bharat-transliteration...")
    run(f"{venv_pip} install ai4bharat-transliteration --no-deps")

    print("\n[SUCCESS] AI4Bharat Transliteration is now installed!")
    print("You can now run the experiment.")

except Exception as e:
    print(f"\n[ERROR] Failed to fix fairseq: {e}")
    print("Please ensure your terminal has internet access.")
finally:
    # Cleanup
    if os.path.exists(TARBALL): os.remove(TARBALL)

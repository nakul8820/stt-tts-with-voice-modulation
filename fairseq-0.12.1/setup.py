
from setuptools import setup, find_packages

setup(
    name="fairseq",
    version="0.12.1",
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

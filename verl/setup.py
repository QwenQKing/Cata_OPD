import os
from pathlib import Path

from setuptools import find_packages, setup

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "verl/version/version")) as f:
    __version__ = f.read().strip()

install_requires = [
    "accelerate",
    "codetiming",
    "datasets",
    "dill",
    "hydra-core",
    "numpy",
    "pandas",
    "peft",
    "pyarrow>=15.0.0",
    "pybind11",
    "pylatexenc",
    "ray[default]>=2.10",
    "torchdata",
    "tensordict<=0.6.2",
    "transformers",
    "wandb",
]

TEST_REQUIRES = ["pytest", "pre-commit", "py-spy"]
PRIME_REQUIRES = ["pyext"]
GEO_REQUIRES = ["mathruler"]
GPU_REQUIRES = ["liger-kernel", "flash-attn"]
MATH_REQUIRES = ["math-verify"]  
VLLM_REQUIRES = ["tensordict<=0.6.2", "vllm<=0.8.3"]
SGLANG_REQUIRES = [
    "tensordict<=0.6.2",
    "sglang[srt,openai]==0.4.6.post1",
    "torch-memory-saver>=0.0.5",
]

extras_require = {
    "test": TEST_REQUIRES,
    "prime": PRIME_REQUIRES,
    "geo": GEO_REQUIRES,
    "gpu": GPU_REQUIRES,
    "math": MATH_REQUIRES,
    "vllm": VLLM_REQUIRES,
    "sglang": SGLANG_REQUIRES,
}

this_directory = Path(__file__).parent
long_description = "Reinforcement learning utilities."

setup(
    name="verl",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where="."),
    description="Reinforcement learning utilities",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "": ["version/*"],
        "verl": ["trainer/config/*.yaml"],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)

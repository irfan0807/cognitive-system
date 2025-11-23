from setuptools import setup, find_packages

setup(
    name="cognitive-system",
    version="0.1.0",
    description="Autonomous AI with embodied cognition driven by neural networks and brain simulation",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "torch>=2.0.0",
        "opencv-python>=4.8.0",
        "sounddevice>=0.4.6",
        "librosa>=0.10.0",
    ],
    python_requires=">=3.8",
)

from setuptools import setup, find_packages

setup(
    name="cognitive-system",
    version="0.1.0",
    description="Autonomous AI with embodied cognition driven by neural networks and brain simulation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.8",
)

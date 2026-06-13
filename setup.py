from setuptools import setup, find_packages

setup(
    name="flower-aistron",
    version="0.1.0",
    packages=find_packages(),
    description="Amodal instance segmentation for flowers using AISFormer",
    license="Apache License 2.0",
    install_requires=["torch"],
)

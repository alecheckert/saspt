#!/usr/bin/env python
import setuptools
setuptools.setup(
    name="saspt",
    version="0.1",
    packages=["saspt"],
    url="https://github.com/alecheckert/saspt",
    author="Alec Heckert",
    author_email="alecheckert@gmail.com",
    description="State arrays for single particle tracking",
    license="MIT",
    install_requires=["numpy", "pandas", "dask", "tqdm",
        "matplotlib", "seaborn", "scipy", "scikit-image"],
    classifiers=[
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)

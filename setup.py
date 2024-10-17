import os

import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages


EIGEN_PATH = os.getenv("EIGEN_PATH", "/usr/include/eigen3")

with open("README.md", "r") as f:
    long_description = f.read()

extensions = [
    Extension(
        "py_bm25.bm25",
        sources=["py_bm25/bm25.pyx", "py_bm25/src/bm25.cpp"],
        language="c++",
        include_dirs=[np.get_include(), EIGEN_PATH],
        extra_compile_args=["-std=c++11", "-O3", "-fopenmp"],
        extra_link_args=["-std=c++11", "-fopenmp"]
    ),
    Extension(
        "py_bm25.convert.data.convert",
        sources=["py_bm25/convert/data/convert.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11", "-O3"],
        extra_link_args=["-std=c++11"]
    ),
    Extension(
        "py_bm25.convert.eigen.eigen",
        sources=["py_bm25/convert/eigen/eigen.pyx"],
        include_dirs=[EIGEN_PATH],
        language="c++",
        extra_compile_args=["-std=c++11", "-O3"],
        extra_link_args=["-std=c++11"]
    )
]

setup(
    name="py-bm25",
    version="0.0.10",
    description="High-Performance BM25 Ranking Functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NaughtyConstrictor/py-bm25",
    author="NaughtyConstrictor",
    author_email="naughtyconstrictor@gmail.com",
    license="MIT",
    packages=find_packages(),
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Operating System :: OS Independent",
    ],
    setup_requires=["numpy", "cython"],
    install_requires=["numpy", "cython"],
    extra_requires={
        "dev": ["pytest", "ir-datasets", "tox", "twine", "numpy"]
    },
    include_dirs=[np.get_include(), EIGEN_PATH],
    python_requires=">=3.7"
)

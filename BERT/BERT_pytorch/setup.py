from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import sys

__version__ = "0.0.1a4"

with open("requirements.txt") as f:
    require_packages = [line[:-1] if line[-1] == "\n" else line for line in f]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="bert_pytorch",
    version=__version__,
    author='Jay Shah',
    author_email='jayshah0726@gmail.com',
    packages=find_packages(),
    install_requires=require_packages,
    url="https://github.com/JS12540/Research-Paper-implementations",
    description="Google AI 2018 BERT pytorch implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'bert = bert_pytorch.__main__:train',
            'bert-vocab = bert_pytorch.dataset.vocab:build',
        ]
    }
)
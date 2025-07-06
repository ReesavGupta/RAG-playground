#!/usr/bin/env python3
"""
Setup script for the Hybrid Search Research Assistant.
"""

from setuptools import setup, find_packages

setup(
    name="research-assistant",
    version="1.0.0",
    description="A modular research assistant with hybrid search capabilities",
    author="Research Assistant Team",
    packages=find_packages(),
    install_requires=[
        "langchain==0.1.0",
        "langchain-community==0.0.10",
        "langchain-ollama==0.0.1",
        "chromadb==0.4.22",
        "pypdf2==3.0.1",
        "requests==2.31.0",
        "sentence-transformers==2.2.2",
        "rank-bm25==0.2.2",
        "numpy==1.24.3",
        "python-dotenv==1.0.0",
        "streamlit==1.29.0",
        "plotly==5.17.0",
        "pandas==2.0.3",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
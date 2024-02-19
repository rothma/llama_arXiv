# ArXiv Paper Downloader and LLaMA Fine-Tuner

This project provides a comprehensive toolkit for downloading arXiv papers, creating a structured dataset from them, and fine-tuning the LLaMA model for specific domains or interests. It culminates in the deployment of an interactive chatbot that leverages the fine-tuned model to provide insightful responses based on the arXiv dataset.

## Features

- **ArXiv Downloader**: Scripts to download papers from the arXiv platform.
- **Dataset Creation**: Tools for transforming downloaded papers into a structured dataset. The tool uses an LLM to create an appropriate instruction/response pattern.
- **Dataset Helpers**: Utility functions for manipulating and preparing the dataset.
- **LLaMA Fine-Tuning**: A Jupyter notebook to fine-tune the LLaMA model on the arXiv dataset.
- **Interactive Chatbot**: A Python script to deploy a chatbot using the fine-tuned model.

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed, along with Jupyter Notebook for running the fine-tuning notebook.

### Installation

1. Clone this repository to your local machine.
2. Install the required dependencies:

```bash
pip install -r requirements.txt

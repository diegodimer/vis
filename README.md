# Evaluating Pre-Training Bias on Severe Acute Respiratory Syndrome Dataset

## Description

This is the code used for the paper Evaluating Pre-Training Bias on Severe Acute Respiratory Syndrome Dataset for CMP273- 2023/2

## Installation

To set up the project, follow these steps:

1. Create a virtual environment using Python's venv:

    ```bash
    python -m venv env
    ```

2. Activate the virtual environment:

    - On Windows:
    
        ```bash
        .\env\Scripts\activate
        ```
    
    - On macOS and Linux:
    
        ```bash
        source env/bin/activate
        ```

3. Install the project dependencies:

    ```bash
    pip install -r requirements.txt
    ```

The experiment codes are under SRAG/main.py
The datasets can be found at https://opendatasus.saude.gov.br/dataset/srag-2021-a-2023 and should be placed under resources/datasets (to be found by the SRAG/data.py code)
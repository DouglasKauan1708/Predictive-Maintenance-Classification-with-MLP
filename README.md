# Predictive Maintenance Classification with MLP

Classification model using a Multilayer Perceptron (MLP) neural network on the **Machine Predictive Maintenance Classification** dataset.
The model predicts the type of failure based on sensor readings.

## Prerequisites

- Python 3.14 or later
- Git

## Installation

``` bash
# Clone the repository
git clone https://github.com/user/project.git
cd project

# Create and activate the virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python -m pipelines.run_pipeline
```

## Project Structure

```
project/
    config/             # Centralized configurations
    data/               # Cleaned data
    notebooks/          # Jupyter notebooks with explanations
    pipelines/          # Main pipeline script
    raw/                # Raw input data
    src/                # MLP functions
    .gitignore          
    README.md           
    requirements.txt    
```

## Authors

- Douglas Kauan Dutra de Oliveira [@DouglasKauan1708](https://github.com/DouglasKauan1708)

# Grain Processor

## Quickstart

```bash
pip install git+https://github.com/mipt-srf/grain-processor
```

```python
import GrainProcessor as px
GP = GrainProcessor(
    r"path\to\image.tif",
    cut_SEM=True,
    fft_filter=True,
)
GP.save_results()
```

## Installation

Using latest version from Github

```bash
pip install git+https://github.com/mipt-srf/grain-processor
```

## Contribution

Clone the repository

```bash
git clone https://github.com/mipt-srf/grain-processor
```

Open repository folder

```bash
cd grain-processor 
```

Install dependencies

- Using pip

    Create virtual environment

    ```bash
    python -m venv .venv
    ```

    Activate the environment

    ```bash
    .venv\Scripts\activate
    ```

    Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

- Using uv

    Create virtual environment with dependencies

    ```bash
    uv sync
    ```

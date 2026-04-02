# Grain Processor

## Quickstart

`pip install grain-processor`

```python
import GrainProcessor as px
GP = GrainProcessor(
    r"path\to\image.tif",
    cut_SEM=True,
    fft_filter=True,
)
GP.save_results()
```

See the [jupyter notebook]() and [documentation]() for more examples.

## Installation

<!-- Using latest release on PyPI

```bash
pip install grain-processor
``` -->

Using latest version from Github

```bash
pip install git+https://github.com/mipt-srf/grain-processor
```

## Contribution

Clone the repository

```bash
git clone https://github.com/mipt-srf/grain-processor
```

Install package locally

```bash
pip install -e /path/to/sphinx_copybutton
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

<!-- Run a script

- Using pip

    ```bash
    python grain_processor/process_all.py
    ```

- Using uv

    ```bash
    uv run grain_processor/process_all.py
    ``` -->

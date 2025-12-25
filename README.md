# PERCy

PER-Cy: Prioritized Experience Replay Buffer ported to Python.

This project is inspired by and ported from the original [PER-C repository](https://github.com/redzhepdx/PER-C). The original implementation is written in C, and this Python version aims to provide similar functionality with Python bindings.

## Features

- Python bindings for the PER-C library.
- Easy integration with Python-based reinforcement learning workflows.
- Includes examples and tests for quick setup and usage.

## Repository

For the original C implementation, visit the [PER-C repository](https://github.com/redzhepdx/PER-C).

## Getting Started

Experimental pip installation.

```bash
python3 -m pip install percy-rl
```

or if you want to install from source:

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/PERCy.git
   cd PERCy
   ```

2. Set up a virtual environment
    ```bash
    python3 -m venv percy_env
    source percy_env/bin/activate
    ```

3. Install it locally
    ```bash
    pip install -e . 
    ```

4. Run basic example
    ```bash
    python3 examples/examples_basic.py
    ```
```text
   ______                         ____  __              
  / ____/___  ____ ___  ____     / __ \/ /_  __  ___  __
 / /   / __ \/ __ `__ \/ __ \   / /_/ / __ \/ / / / |/_/
/ /___/ /_/ / / / / / / /_/ /  / ____/ / / / /_/ />  <  
\____/\____/_/ /_/ /_/ .___/  /_/   /_/ /_/\__, /_/|_|  
                    /_/                   /____/        
```


**compPhyx** is a modular computational physics library written in Python.  
It provides reusable numerical tools for:

- Finite difference methods
- Series expansions (e.g., Taylor series)
- Numerical differentiation and integration
- ODE and PDE solvers (in development)
- Linear algebra and stochastic methods (planned)

The goal of this project is to build a clean, extensible computational physics framework suitable for learning, research prototyping, and future expansion.

## Installation (development mode)

```bash
git clone https://github.com/BarlevR/compPhyx.git
cd compPhyx
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Structure
```text
compPhyx/
├── compPhyx/          # Core library package
│   ├── approx/        # Series expansions and approximations
│   ├── calculus/      # Numerical differentiation and integration
│   └── ...
├── examples/          # Example scripts
├── pyproject.toml     # Package configuration
└── README.md
```

## Requirements
Core dependencies are defined in pyproject.toml.

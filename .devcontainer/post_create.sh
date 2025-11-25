#!/usr/bin/env bash
set -euo pipefail


# 1. Update ~/.bashrc with Dev Container history settings
{
  echo ""
  echo "# Dev Container history settings"
  echo "HISTSIZE=100000"
  echo "HISTFILESIZE=200000"
  echo "HISTTIMEFORMAT=\"%F %T \""
  echo "PROMPT_COMMAND=\"history -a; history -n; \${PROMPT_COMMAND}\""
} >> ~/.bashrc


# 2. Install packages
python -m pip install -U pip
pip install -r .devcontainer/requirements-colab-like.txt


# 3. Print versions (your inline Python block)
python - <<'PY'
import sys, IPython, numpy, pandas, scipy, statsmodels, matplotlib, seaborn, yfinance
print('Python', sys.version.split()[0])
print('IPython', IPython.__version__)
print('numpy', numpy.__version__)
print('pandas', pandas.__version__)
print('scipy', scipy.__version__)
print('statsmodels', statsmodels.__version__)
print('matplotlib', matplotlib.__version__)
print('seaborn', seaborn.__version__)
print('yfinance', yfinance.__version__)
PY

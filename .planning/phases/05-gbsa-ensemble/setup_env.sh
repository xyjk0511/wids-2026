#!/bin/bash
# Phase 5 环境设置脚本 - 创建独立 scikit-survival 0.22.2 环境

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv_sksurv22"

echo "=========================================="
echo "Phase 5: GBSA Ensemble Environment Setup"
echo "=========================================="
echo ""
echo "Project root: ${PROJECT_ROOT}"
echo "Virtual env:  ${VENV_PATH}"
echo ""

# 检查 Python 版本
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: ${PYTHON_VERSION}"

if [[ ! "${PYTHON_VERSION}" =~ ^3\.(9|10|11|12) ]]; then
    echo "Warning: Python 3.9+ recommended, found ${PYTHON_VERSION}"
fi

# 创建虚拟环境
if [ -d "${VENV_PATH}" ]; then
    echo ""
    echo "Virtual environment already exists at ${VENV_PATH}"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        rm -rf "${VENV_PATH}"
    else
        echo "Keeping existing environment. Exiting."
        exit 0
    fi
fi

echo ""
echo "Creating virtual environment..."
python -m venv "${VENV_PATH}"

# 激活环境
echo "Activating environment..."
if [ -f "${VENV_PATH}/Scripts/activate" ]; then
    source "${VENV_PATH}/Scripts/activate"
elif [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "Error: Cannot find activation script"
    exit 1
fi

# 升级 pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# 安装核心依赖
echo ""
echo "Installing core dependencies..."
echo "  - scikit-survival==0.22.2 (GBSA support)"
echo "  - scikit-learn (required by sksurv)"
echo "  - pandas, numpy (data processing)"
echo "  - lifelines (Cox/AFT models)"
echo ""

pip install \
    scikit-survival==0.22.2 \
    scikit-learn \
    pandas \
    numpy \
    lifelines

# 可选: 安装其他依赖
echo ""
read -p "Install optional dependencies (lightgbm, catboost, xgboost)? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing optional dependencies..."
    pip install lightgbm catboost xgboost
fi

# 记录版本快照
echo ""
echo "Saving dependency snapshot..."
SNAPSHOT_FILE="${PROJECT_ROOT}/.planning/phases/05-gbsa-ensemble/requirements_sksurv22.txt"
pip freeze > "${SNAPSHOT_FILE}"
echo "Snapshot saved to: ${SNAPSHOT_FILE}"

# 验证关键包
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "
import sys
import sksurv
import sklearn
import pandas as pd
import numpy as np
import lifelines

print(f'Python:           {sys.version.split()[0]}')
print(f'scikit-survival:  {sksurv.__version__}')
print(f'scikit-learn:     {sklearn.__version__}')
print(f'pandas:           {pd.__version__}')
print(f'numpy:            {np.__version__}')
print(f'lifelines:        {lifelines.__version__}')

# 验证 GBSA 可用
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
print('')
print('✓ GradientBoostingSurvivalAnalysis available')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate this environment:"
echo "  source ${VENV_PATH}/Scripts/activate  # Git Bash"
echo "  ${VENV_PATH}\\Scripts\\activate.bat    # CMD"
echo ""
echo "To run quick validation:"
echo "  python scripts/exp34_gbsa_ensemble.py --mode quick"
echo ""

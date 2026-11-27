#!/usr/bin/env bash
set -euo pipefail
if [[ "${CONDA_DEFAULT_ENV:-}" != fsdp-med3d ]]; then echo 'Activate fsdp-med3d first.' >&2; exit 2; fi
# C8uses the numerical ESCNN backend; no Fortran/py3nj compiler is required.
if ! python -c 'from escnn import gspaces, nn; gspaces.rot2dOnR2(N=8)' ; then
  python -m pip install 'numpy<2' joblib 'pymanopt==2.2.1' 'autograd==1.8.0'
  python -m pip install --no-build-isolation 'lie-learn==0.0.1.post1'
  python -m pip install --no-deps 'escnn==1.0.11'
fi
python -c 'from escnn import gspaces, nn; import importlib.metadata as m; print("ESCNN", m.version("escnn")); print(gspaces.rot2dOnR2(N=8))'

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$env:EDGEMESH_DATA_DIR = Join-Path $taskRoot '.da3-validation'
$env:EDGEMESH_DA3_PYTHON = Join-Path $env:EDGEMESH_DATA_DIR 'runtimes\depth-anything-3\Scripts\python.exe'
$env:HF_HUB_CACHE = Join-Path $env:EDGEMESH_DATA_DIR 'cache\huggingface'
& 'D:\Documents\Code\GitHub\EdgeMesh\.venv\Scripts\python.exe' (Join-Path $taskRoot 'edge_mesh.py')

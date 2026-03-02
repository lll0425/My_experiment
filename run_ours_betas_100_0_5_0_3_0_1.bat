@echo off
setlocal

REM Ensure working directory is this script's folder.
cd /d "%~dp0"

REM You can edit these defaults if needed.
set DEVICE_ID=0
set TASK=label_skew
set DATASET=fl_fashionmnist
set ATTACK_TYPE=backdoor
set OPTIM=fedfish
set SERVER=Ours
echo Running Ours for Dirichlet beta: 100.0, 0.5, 0.3, 0.1
echo Seeds: 0, 1, 2

for %%B in (100.0 0.5 0.3 0.1) do (
  for %%S in (0 1 2) do (
    echo.
    echo ===== Start beta %%B seed %%S =====
    python main.py --device_id %DEVICE_ID% --task %TASK% --dataset %DATASET% --attack_type %ATTACK_TYPE% --optim %OPTIM% --server %SERVER% --seed %%S --csv_name beta%%B_seed%%S DATASET.beta %%B
    if errorlevel 1 (
      echo.
      echo Beta %%B seed %%S failed. Stop remaining runs.
      exit /b 1
    )
    echo ===== Beta %%B seed %%S done =====
  )
)

echo.
echo All beta x seed runs finished successfully.
exit /b 0

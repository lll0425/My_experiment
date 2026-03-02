@echo off
setlocal

REM Ensure working directory is this script's folder.
cd /d "%~dp0"

echo Running 3 rounds: seed 0, 1, 2

for %%S in (0 1 2) do (
  echo.
  echo ===== Start seed %%S =====
  python main.py --device_id 0 --task label_skew --dataset fl_fashionmnist --attack_type backdoor --optim fedfish --server Ours --seed %%S --csv_name beta0.2_seed%%S attack.noise_data_rate 0.5
  if errorlevel 1 (
    echo.
    echo Seed %%S failed. Stop remaining runs.
    exit /b 1
  )
  echo ===== Seed %%S done =====
)

echo.
echo All seeds finished successfully.
exit /b 0


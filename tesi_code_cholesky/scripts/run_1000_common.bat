@echo off
rem Run of 1000 wires (side 95), 100 networks, K = 1 and then K = 10,
rem with V = V* x Vref, the SAME Vref for all the networks (mode "common").
rem Vref is the mean of the Vth of the networks, read from the caches of the
rem mode "own": RUN run_1000_own.bat FIRST, otherwise this script stops with
rem the message "No Vth found".
rem Start it from the Miniforge Prompt:  scripts\run_1000_common.bat
rem If you stop it, start it again: it goes on from the networks already done.
cd /d "%~dp0.."
rem Solver: chol (Cholesky, needs scikit-sparse) or lu (default fast LU)
set SOLVER=chol
rem Number of processes at the same time (more processes use more RAM)
set PROCS=1
set RATIO=1000
set K=1
python scripts\run_conductance_curves_common.py 1000:95:100
python scripts\plot_conductance_curves_common.py
set K=10
python scripts\run_conductance_curves_common.py 1000:95:100
python scripts\plot_conductance_curves_common.py
echo.
echo Finished: mode common, 1000 wires, K = 1 and K = 10.
pause

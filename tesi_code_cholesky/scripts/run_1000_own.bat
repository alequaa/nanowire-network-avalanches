@echo off
rem Run of 1000 wires (side 95), 100 networks, K = 1 and then K = 10,
rem with V = V* x Vth of EACH network (mode "own").
rem Start it from the Miniforge Prompt, in any folder:  scripts\run_1000_own.bat
rem (from the main folder). The Vth of each network is computed by bisection,
rem so K = 10 is slow (about 1 minute per network in my tests, one process).
rem If you stop it, start it again: it goes on from the networks already done.
cd /d "%~dp0.."
rem Solver: chol (Cholesky, needs scikit-sparse) or lu (default fast LU)
set SOLVER=chol
rem Number of processes at the same time (more processes use more RAM)
set PROCS=1
set RATIO=1000
set K=1
python scripts\run_conductance_curves.py 1000:95:100
python scripts\plot_conductance_curves.py
set K=10
python scripts\run_conductance_curves.py 1000:95:100
python scripts\plot_conductance_curves.py
echo.
echo Finished: mode own, 1000 wires, K = 1 and K = 10.
pause

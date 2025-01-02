@echo off
REM Get the directory of this script
set script_dir=%~dp0

REM Path to Julia
set PATH_TO_JULIA=%script_dir%venv\julia-1.6.7\bin\julia.exe

REM Activate the Python virtual environment
call %script_dir%venv\Scripts\activate

REM Run the Python viewer script
python %script_dir%onedcellsim\simulators\multistability\viewer.py %PATH_TO_JULIA%
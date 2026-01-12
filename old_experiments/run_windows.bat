@echo off
echo Installing dependencies for Nano FSD...
echo ---------------------------------------
pip install torch torchvision opencv-python numpy scikit-learn

echo.
echo Starting Nano FSD Demo...
echo ---------------------------------------
python fsd_laptop_demo.py

pause

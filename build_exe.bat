@echo off
chcp 65001 >nul
REM ===========================================================================
REM  ParallelSkreyper.exe ni yig'ish (PyInstaller)
REM  Loyiha papkasida, virtual muhit faollashtirilgan holda ishga tushiring:
REM      .venv\Scripts\activate
REM      build_exe.bat          (GPU qismi bilan)
REM      build_exe.bat nogpu    (GPU siz, kichikroq)
REM  Natija: dist\ParallelSkreyper\ParallelSkreyper.exe
REM ===========================================================================
set BUNDLE_CUDA=1
if /I "%1"=="nogpu" set BUNDLE_CUDA=0

echo [1/3] PyInstaller o'rnatilmoqda...
python -m pip install --upgrade pyinstaller pyinstaller-hooks-contrib psutil || goto :err

echo [2/3] Yig'ilmoqda (bir necha daqiqa)...
python -m PyInstaller --noconfirm --clean ParallelSkreyper.spec || goto :err

echo [3/3] Tekshirish: exe buyruq qatori rejimida ishga tushadimi...
"dist\ParallelSkreyper\ParallelSkreyper.exe" --test > dist\exe_test.txt 2>&1
type dist\exe_test.txt | findstr /C:"test o'tdi"

echo.
echo Tayyor: dist\ParallelSkreyper\ParallelSkreyper.exe
echo Tarqatish uchun butun dist\ParallelSkreyper papkasini ZIP qiling.
pause
exit /b 0

:err
echo XATO: yig'ish to'xtadi. Yuqoridagi xabarni nusxalab yuboring.
pause
exit /b 1

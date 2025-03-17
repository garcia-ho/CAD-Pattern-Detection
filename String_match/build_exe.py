import PyInstaller.__main__
import os
import shutil

# Clean previous build artifacts
for folder in ['build', 'dist']:
    if os.path.exists(folder):
        shutil.rmtree(folder)

# Package the app with PyInstaller
PyInstaller.__main__.run([
    'string_detect_gui.py',             # Your script file
    '--name=CADStringDetector',         # Name of the executable
    '--onefile',                        # Create a single executable file
    '--windowed',                       # Don't show console window (for Windows)
    '--add-data=target.txt;.',          # Include target.txt file
    '--icon=icon.ico',                  # Add an icon (you'll need to create/provide this)
    '--hidden-import=PyQt5',
    '--hidden-import=cv2',
    '--hidden-import=numpy',
    '--hidden-import=pytesseract',
    '--hidden-import=PIL',
    '--hidden-import=fitz'
])

print("Build completed. Check the 'dist' folder for the executable.")
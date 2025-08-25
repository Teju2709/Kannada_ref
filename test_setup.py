#!/usr/bin/env python3
"""
Test Setup Script for Kannada OCR Pipeline
Checks if all dependencies and files are properly set up
"""

import sys
import os
import importlib
from pathlib import Path
# ...existing code...
import api  # This will set up the API key
import google.generativeai as genai

# Now you can use genai as needed
# ...existing code...

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_required_files():
    """Check if all required Python files exist"""
    print("\nChecking required files...")
    required_files = [
        'kannada_ocr.py',
        'kannada_postprocessor.py', 
        'gemini_kannada_refiner.py',
        'integrated_pipeline.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} - Found")
        else:
            print(f"✗ {file} - Missing")
            missing_files.append(file)
    
    return len(missing_files) == 0

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\nChecking Python dependencies...")
    
    required_packages = [
        ('cv2', 'opencv-python'),
        ('pytesseract', 'pytesseract'),
        ('pdf2image', 'pdf2image'),
        ('PIL', 'Pillow'),
        ('docx', 'python-docx'),
        ('google.generativeai', 'google-generativeai'),
        ('numpy', 'numpy'),
        ('requests', 'requests')
    ]
    
    missing_packages = []
    
    for package, install_name in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {install_name} - Installed")
        except ImportError:
            print(f"✗ {install_name} - Missing")
            missing_packages.append(install_name)
    
    if missing_packages:
        print(f"\nTo install missing packages, run:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_tesseract():
    """Check if Tesseract is available"""
    print("\nChecking Tesseract OCR...")
    
    try:
        import pytesseract
        
        # Try to get tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract {version} - Found")
        
        # Check for Kannada language support
        try:
            langs = pytesseract.get_languages()
            if 'kan' in langs:
                print("✓ Kannada language support - Available")
            else:
                print("✗ Kannada language support - Missing")
                print("  Install with: sudo apt-get install tesseract-ocr-kan (Linux)")
                print("  Or download language pack for Windows/Mac")
                return False
        except:
            print("? Kannada language support - Unable to verify")
        
        return True
        
    except Exception as e:
        print(f"✗ Tesseract - Not found ({e})")
        print("  Install from: https://github.com/tesseract-ocr/tesseract")
        return False

def test_imports():
    """Test importing all modules"""
    print("\nTesting module imports...")
    
    modules = [
        ('kannada_ocr', 'KannadaPDFOCR'),
        ('kannada_postprocessor', 'KannadaPostProcessor'),
        ('gemini_kannada_refiner', 'GeminiKannadaRefiner')
    ]
    
    for module_name, class_name in modules:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            print(f"✓ {module_name}.{class_name} - Import successful")
        except Exception as e:
            print(f"✗ {module_name}.{class_name} - Import failed: {e}")
            return False
    
    return True

def create_requirements_txt():
    """Create requirements.txt file"""
    requirements = """opencv-python>=4.8.0
pytesseract>=0.3.10
pdf2image>=1.16.0
Pillow>=10.0.0
python-docx>=0.8.11
google-generativeai>=0.3.0
numpy>=1.24.0
requests>=2.31.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print(f"✓ Created requirements.txt")

def main():
    """Run all setup checks"""
    print("="*50)
    print("KANNADA OCR PIPELINE SETUP CHECK")
    print("="*50)
    
    checks = []
    
    # Run all checks
    checks.append(check_python_version())
    checks.append(check_required_files())
    checks.append(check_dependencies())
    checks.append(check_tesseract())
    checks.append(test_imports())
    
    # Summary
    print("\n" + "="*50)
    print("SETUP CHECK SUMMARY")
    print("="*50)
    
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print("🎉 ALL CHECKS PASSED!")
        print("Your setup is ready for Kannada OCR processing.")
        print("\nNext steps:")
        print("1. Get Google AI API key from: https://makersuite.google.com/app/apikey")
        print("2. Test with: python integrated_pipeline.py --help")
    else:
        print(f"❌ {total - passed} checks failed out of {total}")
        print("Please fix the issues above before proceeding.")
        
        # Create requirements.txt if dependencies are missing
        create_requirements_txt()
        print("\nTo install missing Python packages:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
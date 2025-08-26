# Environment Setup Guide

## Prerequisites & Requirements

### Technical Requirements
- **Python**: 3.10+ (verify with `python --version`)
- **Hardware**: 8GB RAM minimum, 16GB recommended
- **Network**: Stable internet connection for Azure OpenAI calls
- **IDE**: VS Code (recommended) or Jupyter Notebook

### Required Accounts & Keys
- Azure subscription with OpenAI service access
- Azure OpenAI API key and endpoint
- GitHub account (for code repositories)

## Installation Steps

### Step 1: Verify Python Installation
```bash
# Verify Python version
python --version
# Should show Python 3.10 or higher
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

### Step 3: Install Dependencies
```bash
# Install required packages
pip install semantic-kernel[azure]
pip install streamlit
pip install python-dotenv
```

## Configuration

### Environment Variables Setup
Create `.env` file in your project root:
```bash
# Required for Azure OpenAI
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-01

# Optional for enhanced features
LOGGING_LEVEL=INFO
PERFORMANCE_MONITORING=true
```

### .env.example Template
Copy this template for your `.env` file:
```bash
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-01

# Optional Settings
LOGGING_LEVEL=INFO
PERFORMANCE_MONITORING=true
```

## Verification Steps

### Test Your Setup
```python
# Run the setup verification
python -c "
import asyncio
from dotenv import load_dotenv
load_dotenv()
print('✅ Environment ready for workshop!')
"
```

### Verify Dependencies
```python
# Test imports
python -c "
import semantic_kernel
import streamlit
from dotenv import load_dotenv
print('✅ All dependencies installed correctly!')
"
```

## Troubleshooting

### Common Setup Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Python Version Error** | `python --version` shows < 3.10 | Install Python 3.10+ from python.org |
| **Virtual Environment Issues** | `venv` command not found | Use `python -m venv` instead |
| **Package Installation Fails** | pip install errors | Upgrade pip: `python -m pip install --upgrade pip` |
| **Import Errors** | `ModuleNotFoundError` | Verify virtual environment is activated |
| **Azure OpenAI Connection** | 401/403 errors | Check API key and endpoint in `.env` file |

### Debug Script
```python
import os
import sys
import subprocess
from dotenv import load_dotenv

def debug_setup():
    """Debug workshop setup"""
    
    print("Setup Debugging")
    print("=" * 30)
    
    # Check Python version
    python_version = sys.version
    print(f"Python Version: {python_version}")
    
    # Check virtual environment
    venv = os.getenv('VIRTUAL_ENV')
    print(f"Virtual Environment: {venv if venv else 'Not activated'}")
    
    # Check required environment variables
    required_vars = [
        'AZURE_OPENAI_API_KEY',
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_DEPLOYMENT_NAME',
        'AZURE_OPENAI_API_VERSION'
    ]
    
    print("\nEnvironment Variables:")
    for var in required_vars:
        value = os.getenv(var)
        status = "✅ Set" if value else "❌ Missing"
        print(f"  {var}: {status}")
    
    # Check package installations
    packages = ['semantic-kernel', 'streamlit', 'python-dotenv']
    print("\nPackage Installation:")
    
    for package in packages:
        try:
            result = subprocess.run([sys.executable, '-c', f'import {package.replace("-", "_")}'], 
                                  capture_output=True, text=True)
            status = "✅ Installed" if result.returncode == 0 else "❌ Missing"
            print(f"  {package}: {status}")
        except Exception as e:
            print(f"  {package}: ❌ Error - {e}")

if __name__ == "__main__":
    debug_setup()
```

## Next Steps

Once your environment is set up successfully:

1. Proceed to [Insurance Domain Context](insurance-context.md)
2. Start with [Module 1: SK Fundamentals](../modules/01-sk-fundamentals.md)
3. Review the [Complete Workshop Guide](../README.md)

---

*For additional help, refer to [Troubleshooting Guide](troubleshooting.md)*
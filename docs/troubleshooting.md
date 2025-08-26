# Troubleshooting Guide

This comprehensive troubleshooting guide covers common issues encountered during the Semantic Kernel Insurance Bootcamp.

## Quick Diagnostic Checklist

### Before Starting Troubleshooting
Run this quick diagnostic to identify the general area of the problem:

```python
import os
import sys
import subprocess
import asyncio
from dotenv import load_dotenv

def quick_diagnostic():
    """Run quick diagnostic to identify common issues"""
    
    print("Quick Diagnostic Check")
    print("=" * 40)
    
    issues_found = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 10):
        issues_found.append("? Python version too old")
        print(f"? Python: {python_version.major}.{python_version.minor} (Need 3.10+)")
    else:
        print(f"? Python: {python_version.major}.{python_version.minor}")
    
    # Check virtual environment
    venv = os.getenv('VIRTUAL_ENV')
    if venv:
        print(f"? Virtual Environment: Active")
    else:
        issues_found.append("Virtual environment not detected")
        print(f"Virtual Environment: Not detected")
    
    # Check environment variables
    load_dotenv()
    required_vars = [
        'AZURE_OPENAI_API_KEY',
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_DEPLOYMENT_NAME',
        'AZURE_OPENAI_API_VERSION'
    ]
    
    print("\nEnvironment Variables:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"? {var}: Set")
        else:
            issues_found.append(f"? {var}: Missing")
            print(f"? {var}: Missing")
    
    # Check package installations
    packages = ['semantic-kernel', 'streamlit', 'python-dotenv']
    print("\nPackage Installation:")
    
    for package in packages:
        try:
            module_name = package.replace('-', '_')
            result = subprocess.run([sys.executable, '-c', f'import {module_name}'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"? {package}: Installed")
            else:
                issues_found.append(f"? {package}: Missing")
                print(f"? {package}: Missing")
        except Exception as e:
            issues_found.append(f"? {package}: Error")
            print(f"? {package}: Error - {e}")
    
    print("\n" + "=" * 40)
    if issues_found:
        print("Issues Found:")
        for issue in issues_found:
            print(f"  {issue}")
        print("\nRefer to specific sections below for solutions.")
    else:
        print("? All checks passed!")
    
    return len(issues_found) == 0

# Run diagnostic
if __name__ == "__main__":
    quick_diagnostic()
```

---

## Python and Environment Issues

### Issue: Python Version Too Old
**Symptoms**: `SyntaxError` with modern Python features, import errors
**Solution**:
1. Install Python 3.10+ from [python.org](https://python.org)
2. Update your PATH to use the new version
3. Recreate your virtual environment:
```bash
python -m venv .venv --clear
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

### Issue: Virtual Environment Problems
**Symptoms**: Packages not found, permission errors, wrong Python version
**Solution**:
```bash
# Delete existing environment
rm -rf .venv  # Windows: rmdir /s .venv

# Create new environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Verify activation
which python  # Windows: where python
python --version

# Install packages
pip install --upgrade pip
pip install semantic-kernel[azure] streamlit python-dotenv
```

### Issue: Package Installation Failures
**Symptoms**: `pip install` fails, dependency conflicts
**Solutions**:

**For network/proxy issues**:
```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org semantic-kernel[azure]
```

**For dependency conflicts**:
```bash
pip install --upgrade pip setuptools wheel
pip install --no-cache-dir semantic-kernel[azure]
```

**For permission issues**:
```bash
# Ensure virtual environment is activated first
pip install --user semantic-kernel[azure]
```

---

## Azure OpenAI Configuration Issues

### Issue: API Key Authentication Failures
**Symptoms**: 401 Unauthorized, 403 Forbidden errors
**Diagnosis**:
```python
import os
from dotenv import load_dotenv

def test_azure_config():
    load_dotenv()
    
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    
    print(f"API Key exists: {bool(api_key)}")
    print(f"API Key length: {len(api_key) if api_key else 0}")
    print(f"Endpoint: {endpoint}")
    
    if api_key:
        print(f"API Key starts with: {api_key[:10]}...")
    
test_azure_config()
```

**Solutions**:
1. **Verify API key format**: Should be 32 characters, alphanumeric
2. **Check endpoint format**: Should be `https://your-resource.openai.azure.com/`
3. **Verify resource exists**: Check Azure portal
4. **Regenerate keys**: In Azure portal, regenerate keys if suspicious of compromise

### Issue: Deployment Name Errors
**Symptoms**: Model not found, deployment errors
**Solution**:
```python
# Test deployment access
import asyncio
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

async def test_deployment():
    try:
        service = AzureChatCompletion(
            service_id="test",
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        print("? Service created successfully")
        return True
    except Exception as e:
        print(f"? Service creation failed: {e}")
        return False

# asyncio.run(test_deployment())
```

### Issue: API Version Compatibility
**Symptoms**: Unsupported API version errors
**Solution**: Use supported API versions:
```bash
# In .env file, use one of these versions:
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_API_VERSION=2023-12-01-preview
AZURE_OPENAI_API_VERSION=2023-05-15
```

---

## Semantic Kernel Issues

### Issue: Kernel Creation Failures
**Symptoms**: Kernel initialization errors, service registration fails
**Debug Script**:
```python
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

async def debug_kernel_creation():
    try:
        print("1. Creating kernel...")
        kernel = Kernel()
        print("? Kernel created")
        
        print("2. Creating service...")
        service = AzureChatCompletion(
            service_id="test",
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        print("? Service created")
        
        print("3. Adding service to kernel...")
        kernel.add_service(service)
        print("? Service added")
        
        print("4. Testing service retrieval...")
        retrieved_service = kernel.get_service("test")
        print("? Service retrieved")
        
        return True
        
    except Exception as e:
        print(f"? Failed at step: {e}")
        import traceback
        traceback.print_exc()
        return False

# asyncio.run(debug_kernel_creation())
```

### Issue: Function Registration Problems
**Symptoms**: Function not found errors, KeyError exceptions
**Solution**:
```python
def debug_function_registration():
    """Debug function registration issues"""
    
    # Check function creation
    try:
        from semantic_kernel.functions import kernel_function
        
        @kernel_function(name="test_function", description="Test function")
        def test_func(input: str) -> str:
            return f"Processed: {input}"
        
        print("? Function decorator works")
        
        # Check kernel function addition
        kernel = Kernel()
        kernel.add_function(
            function_name="test_function",
            plugin_name="TestPlugin", 
            prompt="Test prompt: {{$input}}",
            prompt_execution_settings=None
        )
        print("? Function added to kernel")
        
        # Check function retrieval
        func = kernel.get_function("TestPlugin", "test_function")
        print("? Function retrieved")
        
        return True
        
    except Exception as e:
        print(f"? Function registration failed: {e}")
        return False
```

### Issue: Prompt Template Errors
**Symptoms**: Variable substitution fails, template parsing errors
**Common Template Issues**:

1. **Incorrect variable syntax**:
```python
# ? Wrong
"Name: {name}"
"Name: $name"

# ? Correct
"Name: {{$name}}"
```

2. **Missing variables**:
```python
# Template expects {{$name}} but no name provided
result = await kernel.invoke(function, input="test")  # Missing name parameter
```

3. **Special characters**:
```python
# ? Problematic
template = "Price: {{$amount}}$"  # $ conflicts with variable syntax

# ? Fixed
template = "Price: ${{$amount}}"
```

---

## Agent-Specific Issues

### Issue: Agent Not Responding
**Symptoms**: No output, hanging, empty responses
**Debugging Steps**:
```python
async def debug_agent_response():
    """Debug agent response issues"""
    
    try:
        # Test basic kernel functionality
        kernel = await create_kernel()
        print("? Kernel works")
        
        # Test simple function call
        simple_prompt = "Respond with 'Hello World'"
        execution_settings = OpenAIChatPromptExecutionSettings(
            service_id="azure_openai",
            max_tokens=100,
            temperature=0
        )
        
        simple_function = kernel.add_function(
            function_name="simple_test",
            plugin_name="TestPlugin",
            prompt=simple_prompt,
            prompt_execution_settings=execution_settings
        )
        
        result = await kernel.invoke(simple_function)
        print(f"? Simple response: {result}")
        
        return True
        
    except Exception as e:
        print(f"? Agent debugging failed: {e}")
        return False
```

### Issue: Memory/Context Problems
**Symptoms**: Agent forgets previous interactions, context not maintained
**Solution**:
```python
class DebugAgent:
    """Agent with enhanced debugging for memory issues"""
    
    def __init__(self):
        self.conversation_history = []
        self.context = {}
    
    def debug_memory(self):
        """Debug memory and context"""
        print(f"Conversation History ({len(self.conversation_history)} items):")
        for i, item in enumerate(self.conversation_history):
            print(f"  {i+1}. {item}")
        
        print(f"Context ({len(self.context)} items):")
        for key, value in self.context.items():
            print(f"  {key}: {value}")
    
    async def process_with_debug(self, query: str):
        """Process query with memory debugging"""
        
        print(f"Processing: {query}")
        
        # Add to history
        self.conversation_history.append(f"User: {query}")
        
        # Debug before processing
        self.debug_memory()
        
        # Your normal processing here...
        
        return {"message": "Processed with debug info"}
```

### Issue: JSON Parsing Failures
**Symptoms**: JSON decode errors, malformed responses
**Solution**:
```python
import json
import re

def robust_json_extraction(response_text: str) -> dict:
    """Robustly extract JSON from AI responses"""
    
    try:
        # Try direct parsing first
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in response
    json_patterns = [
        r'\{[^{}]*\{[^{}]*\}[^{}]*\}',  # Nested objects
        r'\{[^{}]*\}',                  # Simple objects
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # Fallback to structured response
    return {
        "analysis": "Failed to parse structured response",
        "message": response_text,
        "recommendations": [],
        "error": "JSON parsing failed"
    }

# Test the function
test_response = """
Here's my analysis: The customer needs coverage.
{
    "analysis": "Customer needs basic coverage",
    "recommendations": [
        {"policy_type": "life", "premium": 100}
    ]
}
Additional text here.
"""

result = robust_json_extraction(test_response)
print(result)
```

---

## Plugin Development Issues

### Issue: Plugin Functions Not Called
**Symptoms**: Functions available but not invoked by AI
**Solution**:
```python
from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior

# Ensure function calling is enabled
execution_settings = OpenAIChatPromptExecutionSettings(
    service_id="azure_openai",
    max_tokens=1000,
    temperature=0.7,
    function_call_behavior=FunctionCallBehavior.EnableFunctions(
        auto_invoke=True,  # Important!
        filters={}
    )
)

# Make functions easily discoverable
@kernel_function(
    name="calculate_premium",
    description="Calculate insurance premium for given age, coverage amount, and policy type. Use this when customer asks about costs or pricing."
)
def calculate_premium(age: str, coverage: str, policy_type: str) -> str:
    """Clear, descriptive function"""
    # Implementation here
    pass
```

### Issue: Plugin Registration Errors
**Symptoms**: Plugin not found, function registration fails
**Debug Script**:
```python
def debug_plugin_registration():
    """Debug plugin registration process"""
    
    try:
        # Test class-based plugin
        class TestPlugin:
            @kernel_function(name="test_func", description="Test function")
            def test_function(self, input: str) -> str:
                return f"Test: {input}"
        
        kernel = Kernel()
        plugin = TestPlugin()
        
        # Register plugin
        kernel.add_plugin(plugin, plugin_name="TestPlugin")
        print("? Plugin registered")
        
        # Test function existence
        func = kernel.get_function("TestPlugin", "test_func")
        print("? Function found")
        
        # List all functions
        print("Available functions:")
        for plugin_name, functions in kernel.plugins.items():
            print(f"  Plugin: {plugin_name}")
            for func_name, func in functions.items():
                print(f"    Function: {func_name}")
        
        return True
        
    except Exception as e:
        print(f"? Plugin registration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

debug_plugin_registration()
```

---

## Performance Issues

### Issue: Slow Response Times
**Symptoms**: Responses take >30 seconds, timeouts
**Diagnosis**:
```python
import time
import asyncio

async def performance_test():
    """Test response performance"""
    
    start_time = time.time()
    
    try:
        # Your kernel/agent code here
        kernel = await create_kernel()
        
        setup_time = time.time()
        print(f"Setup time: {setup_time - start_time:.2f}s")
        
        # Test simple function
        result = await kernel.invoke(simple_function, input="test")
        
        end_time = time.time()
        print(f"Execution time: {end_time - setup_time:.2f}s")
        print(f"Total time: {end_time - start_time:.2f}s")
        
        if end_time - start_time > 10:
            print("Performance is slow")
        else:
            print("? Performance is good")
            
    except Exception as e:
        print(f"? Performance test failed: {e}")

# asyncio.run(performance_test())
```

**Solutions**:
1. **Reduce max_tokens**: Lower values = faster responses
2. **Optimize prompts**: Shorter, more focused prompts
3. **Check network**: Ensure stable internet connection
4. **Use caching**: Cache static responses when possible

### Issue: Memory Usage Problems
**Symptoms**: High RAM usage, system slowdown
**Solution**:
```python
import psutil
import gc

def monitor_memory_usage():
    """Monitor memory usage during execution"""
    
    process = psutil.Process()
    
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    print(f"Memory percent: {process.memory_percent():.2f}%")
    
    # Force garbage collection
    gc.collect()
    
    print(f"After GC: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Call periodically during execution
monitor_memory_usage()
```

---

## File and Import Issues

### Issue: Module Import Errors
**Symptoms**: `ModuleNotFoundError`, `ImportError`
**Solutions**:

1. **Check Python path**:
```python
import sys
print("Python path:")
for path in sys.path:
    print(f"  {path}")
```

2. **Verify package installation**:
```bash
pip list | grep semantic-kernel
pip show semantic-kernel
```

3. **Reinstall packages**:
```bash
pip uninstall semantic-kernel
pip install semantic-kernel[azure]
```

### Issue: File Path Problems
**Symptoms**: `.env` not found, file access errors
**Solution**:
```python
import os
from pathlib import Path

def debug_file_paths():
    """Debug file path issues"""
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {__file__ if '__file__' in globals() else 'Unknown'}")
    
    # Check .env file
    env_path = Path(".env")
    print(f".env exists: {env_path.exists()}")
    
    if env_path.exists():
        print(f".env size: {env_path.stat().st_size} bytes")
        with open(env_path) as f:
            lines = f.readlines()
            print(f".env lines: {len(lines)}")
            for i, line in enumerate(lines[:3]):  # Show first 3 lines
                print(f"  Line {i+1}: {line.strip()[:20]}...")
    
    # List current directory
    print("Current directory contents:")
    for item in Path(".").iterdir():
        print(f"  {item}")

debug_file_paths()
```

---

## Quick Fixes

### Complete Environment Reset
If all else fails, start fresh:
```bash
# 1. Remove virtual environment
rm -rf .venv

# 2. Create new environment
python -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install packages fresh
pip install semantic-kernel[azure] streamlit python-dotenv

# 5. Verify installation
python -c "import semantic_kernel; print('? SK installed')"
```

### Reset Azure OpenAI Connection
```python
# Test script to verify Azure OpenAI setup
import os
from dotenv import load_dotenv

# Load fresh environment
load_dotenv(override=True)

# Print configuration (safely)
print("Configuration check:")
print(f"API Key set: {bool(os.getenv('AZURE_OPENAI_API_KEY'))}")
print(f"Endpoint set: {bool(os.getenv('AZURE_OPENAI_ENDPOINT'))}")
print(f"Deployment set: {bool(os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'))}")
print(f"API Version set: {bool(os.getenv('AZURE_OPENAI_API_VERSION'))}")

if os.getenv('AZURE_OPENAI_ENDPOINT'):
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    if not endpoint.startswith('https://'):
        print("Endpoint should start with https://")
    if not endpoint.endswith('/'):
        print("Endpoint should end with /")
```

---

## Getting Additional Help

### Workshop Resources
- **Setup Guide**: [docs/setup.md](setup.md)
- **Insurance Context**: [docs/insurance-context.md](insurance-context.md)
- **Module Documentation**: [modules/](../modules/)

### Microsoft Documentation
- [Semantic Kernel Python Documentation](https://learn.microsoft.com/en-us/semantic-kernel/)
- [Azure OpenAI Service Documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/)

### Community Support
- [Semantic Kernel GitHub Issues](https://github.com/microsoft/semantic-kernel/issues)
- [Semantic Kernel Discord](https://aka.ms/sk-discord)

### Creating a Bug Report
When reporting issues, include:
1. **Environment details**: Python version, OS, package versions
2. **Complete error message**: Full traceback
3. **Minimal reproduction**: Smallest code that demonstrates the issue
4. **Expected vs actual behavior**
5. **Configuration**: Redacted .env contents

```python
# Bug report template
def generate_bug_report():
    """Generate information for bug reports"""
    
    import sys
    import platform
    
    print("=== BUG REPORT INFORMATION ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    
    try:
        import semantic_kernel
        print(f"Semantic Kernel version: {semantic_kernel.__version__}")
    except:
        print("Semantic Kernel: Not installed or version unknown")
    
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")

# Run when creating a bug report
generate_bug_report()
```

---

*This troubleshooting guide is continuously updated based on common workshop issues. If you encounter a problem not covered here, please contribute by documenting the solution.*
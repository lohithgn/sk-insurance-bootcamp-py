# Module 1: Semantic Kernel Fundamentals

**Duration**: 45 minutes  
**Objective**: Build foundational understanding of Semantic Kernel by creating a kernel, configuring AI services, and testing basic prompts.

## Learning Outcomes
By completing this module, you will be able to:
- Create and configure a Semantic Kernel instance
- Add Azure OpenAI services to the kernel
- Write and execute semantic functions
- Understand prompt templates and variables
- Test basic AI interactions
- Integrate native functions with semantic functions

## Prerequisites
- Completed [Environment Setup](../docs/setup.md)
- Understanding of [Insurance Domain Context](../docs/insurance-context.md)
- Azure OpenAI credentials configured in `.env` file

---

## Part 1: Kernel Creation and Configuration

### Step 1: Create Your First Kernel

```python
import asyncio
import os
import json
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

# Load environment variables
load_dotenv()

async def create_kernel():
    """Create and configure a Semantic Kernel instance"""
    
    # Initialize the kernel
    kernel = Kernel()
    
    # Add Azure OpenAI Chat Completion service
    service = AzureChatCompletion(
        service_id="azure_openai",
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    kernel.add_service(service)
    
    print("✅ Kernel created successfully!")
    return kernel


async def main():
    """Main function to test kernel creation"""
    kernel = await create_kernel()

if __name__ == "__main__":
    asyncio.run(main())
```

**Your Task**: Run this code to create your first kernel. You should see the success message printed.

### Step 2: Test Basic AI Connection

Now let's test that our kernel can actually communicate with Azure OpenAI.

```python
async def test_ai_connection(kernel):
    """Test basic AI connectivity"""
    
    # Create execution settings
    execution_settings = OpenAIChatPromptExecutionSettings(
        service_id="azure_openai",
        max_tokens=500,
        temperature=0.7
    )
    
    # Create a simple chat history
    history = ChatHistory()
    history.add_user_message("Hello! Can you help me with insurance?")
    
    # Get the AI service
    ai_service = kernel.get_service("azure_open_ai")
    
    # Get response
    response = await ai_service.get_chat_message_contents(
        chat_history=history,
        settings=execution_settings
    )
    
    print(f"AI Response: {response[0].content}")
    return response
```

**Your Task**: Update the `main()` function to also test the AI connection by adding the test function call:

```python
async def main():
    """Main function to test kernel creation and AI connection"""
    kernel = await create_kernel()
    await test_ai_connection(kernel)
```

Run the updated code. You should now see:
1. ✅ Kernel created successfully!
2. AI Response with an insurance-related greeting

This confirms your Semantic Kernel is properly configured and can communicate with Azure OpenAI.

---

## Part 2: Creating Semantic Functions

### Step 3: Your First Semantic Function

```python
async def create_insurance_assistant(kernel):
    """Create a basic insurance assistant semantic function"""
    
    # Define the prompt template
    insurance_prompt = """
    You are an expert insurance advisor with 20+ years of experience.
    
    Customer Question: {{$input}}
    
    Instructions:
    - Provide helpful, accurate insurance advice
    - Keep responses professional but friendly  
    - If you need more information, ask clarifying questions
    - Always prioritize customer's best interests
    
    Response:
    """
    
    # Create execution settings
    execution_settings = OpenAIChatPromptExecutionSettings(
        service_id="azure_openai",
        max_tokens=500,
        temperature=0.7
    )
    
    # Create the semantic function
    insurance_assistant = kernel.add_function(
        function_name="insurance_assistant",
        plugin_name="InsurancePlugin",
        prompt=insurance_prompt,
        prompt_execution_settings=execution_settings
    )
    
    return insurance_assistant

async def test_semantic_function(kernel, function):
    """Test the semantic function with sample questions"""
    
    test_questions = [
        "What type of life insurance is best for a 30-year-old with a family?",
        "How much auto insurance coverage do I need?",
        "What's the difference between term and whole life insurance?"
    ]
    
    for question in test_questions:
        print(f"\n? Question: {question}")
        
        result = await kernel.invoke(function, input=question)
        print(f"Answer: {result}")
        print("-" * 80)
```

**Your Task**: Now let's test semantic functions. Update your `main()` function to remove the AI connection test and instead test the semantic function:

```python
async def main():
    """Main function to test semantic functions"""
    kernel = await create_kernel()
    function = await create_insurance_assistant(kernel)
    await test_semantic_function(kernel, function)
```

Run the updated code. You should see:
1. ✅ Kernel created successfully!
2. Three insurance questions being answered by your AI assistant

This demonstrates how semantic functions use prompt templates to create specialized AI behaviors.

---

## Part 3: Advanced Prompt Templates

### Step 4: Multi-Variable Prompt Templates

```python
async def create_personalized_advisor(kernel):
    """Create a personalized insurance advisor with multiple variables"""
    
    personalized_prompt = """
    You are a personalized insurance advisor analyzing a customer profile.
    
    Customer Details:
    - Name: {{$name}}
    - Age: {{$age}} 
    - Annual Income: ${{$income}}
    - Family Status: {{$family_status}}
    - Current Coverage: {{$current_coverage}}
    - Specific Need: {{$need}}
    
    Instructions:
    1. Address the customer by name
    2. Consider their age and income for affordability
    3. Factor in their family situation
    4. Account for existing coverage to avoid duplication
    5. Provide 2-3 specific recommendations
    6. Include estimated premium ranges
    
    Personalized Recommendation:
    """
    
    # Create execution settings
    execution_settings = OpenAIChatPromptExecutionSettings(
        service_id="azure_openai",
        max_tokens=800,
        temperature=0.7
    )
    
    # Create the function
    personalized_advisor = kernel.add_function(
        function_name="personalized_advisor", 
        plugin_name="InsurancePlugin",
        prompt=personalized_prompt,
        prompt_execution_settings=execution_settings
    )
    
    return personalized_advisor

async def test_personalized_function(kernel, function):
    """Test personalized advisor with sample customer data"""
    
    sample_customer = {
        "name": "Sarah Johnson",
        "age": "32",
        "income": "85000", 
        "family_status": "Married with 1 child",
        "current_coverage": "Basic health insurance through employer",
        "need": "Need comprehensive family protection including life insurance"
    }
    
    print("🧪 Testing Personalized Advisor:")
    print(f"Customer: {sample_customer['name']}")
    
    result = await kernel.invoke(function, **sample_customer)
    print(f"\nPersonalized Recommendation:\n{result}")
```

**Your Task**: Now let's test multi-variable prompt templates. Update your `main()` function to comment out the step 3 code and test the personalized advisor instead:

```python
async def main():
    """Main function to test personalized advisor"""
    kernel = await create_kernel()
    
    # Step 3 code (comment out for now)
    # function = await create_insurance_assistant(kernel)
    # await test_semantic_function(kernel, function)
    
    # Step 4 code
    function = await create_personalized_advisor(kernel)
    await test_personalized_function(kernel, function)
```

Run the updated code. You should see:
1. ✅ Kernel created successfully!
2. A personalized recommendation for Sarah Johnson with her specific details

This demonstrates how to use multiple variables in prompt templates for personalized AI responses.

---

## Part 4: Native Functions Integration

### Step 5: Create Native Functions

```python
class InsurancePremiumCalculator:
    """Native function class for premium calculations"""
    
    @kernel_function(
        name="calculate_life_premium",
        description="Calculate life insurance premium based on age and coverage amount"
    )
    def calculate_life_premium(self, age: str, coverage_amount: str) -> str:
        """Calculate basic life insurance premium"""
        try:
            age_int = int(age)
            coverage_int = int(coverage_amount)
            
            # Simple premium calculation (real-world would be much more complex)
            base_rate = 0.5  # $0.50 per $1000 of coverage
            age_multiplier = 1 + (age_int - 25) * 0.02  # 2% increase per year after 25
            
            annual_premium = (coverage_int / 1000) * base_rate * age_multiplier
            monthly_premium = annual_premium / 12
            
            result = {
                "coverage_amount": coverage_int,
                "age": age_int,
                "annual_premium": round(annual_premium, 2),
                "monthly_premium": round(monthly_premium, 2)
            }
            
            return json.dumps(result)
            
        except ValueError:
            return json.dumps({"error": "Invalid age or coverage amount"})
    
    @kernel_function(
        name="get_policy_types",
        description="Get available insurance policy types and descriptions"
    )
    def get_policy_types(self, category: str = "all") -> str:
        """Return available policy types"""
        
        policies = {
            "life": {
                "Term Life": "Temporary coverage for specific period",
                "Whole Life": "Permanent coverage with cash value",
                "Universal Life": "Flexible permanent coverage"
            },
            "health": {
                "Individual Health": "Coverage for single person",
                "Family Health": "Coverage for family members", 
                "HSA Compatible": "High-deductible health plan"
            },
            "auto": {
                "Liability": "Basic legal requirement coverage",
                "Comprehensive": "Full coverage including theft/damage",
                "Collision": "Coverage for vehicle damage"
            }
        }
        
        if category == "all":
            return json.dumps(policies)
        else:
            return json.dumps(policies.get(category, {}))

async def test_native_functions(kernel):
    """Test native functions integration"""
    
    # Add the calculator plugin
    calculator = InsurancePremiumCalculator()
    kernel.add_plugin(calculator, plugin_name="Calculator")
    
    # Test premium calculation
    print("Testing Premium Calculator:")
    result = await kernel.invoke(
        kernel.get_function("Calculator", "calculate_life_premium"),
        age="35",
        coverage_amount="500000"
    )
    print(f"Premium Calculation: {result}")
    
    # Test policy types
    print("\nTesting Policy Types:")
    result = await kernel.invoke(
        kernel.get_function("Calculator", "get_policy_types"),
        category="life"
    )
    print(f"Life Insurance Types: {result}")
```

**Your Task**: Now let's test native functions. Update your `main()` function to comment out the step 4 code and test the native functions instead:

```python
async def main():
    """Main function to test native functions"""
    kernel = await create_kernel()
    
    # Step 4 code (comment out for now)
    # function = await create_personalized_advisor(kernel)
    # await test_semantic_function(kernel, function)
    
    # Step 5 code
    await test_native_functions(kernel)
```

Run the updated code. You should see:
1. ✅ Kernel created successfully!
2. Premium calculation results showing JSON data
3. Policy types information for life insurance

This demonstrates how native functions provide precise calculations and structured data that complement AI responses.

---

## Part 5: Combining Semantic and Native Functions

### Step 6: Create Hybrid Function Workflow

```python
async def create_complete_advisor(kernel):
    """Create advisor that combines semantic AI with native calculations"""
    
    hybrid_prompt = """
    You are an insurance advisor with access to premium calculation tools.
    
    Customer Request: {{$request}}
    
    Available Tools:
    - calculate_life_premium: For life insurance premium calculations
    - get_policy_types: For policy information
    
    Instructions:
    1. Understand the customer's request
    2. If they need premium calculations, use the appropriate tools
    3. Provide comprehensive advice combining AI insights with calculated data
    4. Present information in a customer-friendly format
    
    Response:
    """
    
    # Create execution settings
    execution_settings = OpenAIChatPromptExecutionSettings(
        service_id="azure_openai",
        max_tokens=1000,
        temperature=0.7,
        function_call_behavior=FunctionChoiceBehavior.EnableFunctions(
            auto_invoke=True, filters={}
        )
    )
    
    advisor = kernel.add_function(
        function_name="complete_advisor",
        plugin_name="HybridPlugin", 
        prompt=hybrid_prompt,
        prompt_execution_settings=execution_settings
    )
    
    return advisor

async def test_complete_workflow():
    """Test the complete SK fundamentals workflow with automatic function calling"""
    
    print("Testing Complete Semantic Kernel Workflow")
    print("=" * 60)
    
    # 1. Create kernel
    kernel = await create_kernel()
    
    # 2. Add native functions
    calculator = InsurancePremiumCalculator()
    kernel.add_plugin(calculator, plugin_name="Calculator")
    
    # 3. Create hybrid advisor
    advisor = await create_complete_advisor(kernel)
    
    # 4. Test with complex request - let AI automatically use tools
    customer_request = """
    I'm 35 years old and want to buy $500,000 in life insurance for my family. 
    What are my options and how much would each cost approximately?
    """
    
    print(f"Customer Request: {customer_request}")
    print("\nAI Advisor Response (with automatic function calling):")
    
    # Let the AI advisor automatically determine what tools to use
    final_response = await kernel.invoke(advisor, request=customer_request)
    print(final_response)
```

**Your Task**: Now let's test the complete hybrid workflow that combines AI reasoning with precise calculations. Update your `main()` function to comment out the step 5 code and test the complete workflow instead:

```python
async def main():
    """Main function to test complete hybrid workflow"""
    kernel = await create_kernel()
    
    # Step 5 code (comment out for now)
    # await test_native_functions(kernel)
    
    # Step 6 code
    await test_complete_workflow()
```

Run the updated code. You should see:
1. ✅ Kernel created successfully!
2. A complete workflow demonstration showing how AI uses native function results
3. Premium calculations and policy information combined with AI reasoning
4. A comprehensive response to a complex customer request

This demonstrates the power of combining semantic AI functions with native calculations for intelligent, data-driven responses.

---

## Exercise: Build Your Own Insurance Function

**Your Task**: Create a semantic function that helps customers understand insurance deductibles.

### Complete Template:
```python
async def create_deductible_advisor(kernel):
    """Create a deductible advisor function with automatic tool calling"""
    
    prompt = """
    You are an insurance expert specializing in deductibles with access to calculation tools.
    
    Customer Question: {{$question}}
    Insurance Type: {{$insurance_type}}
    Current Premium: {{$current_premium}}
    
    Available Tools:
    - calculate_premium_savings: Calculate potential premium savings with different deductible amounts
    
    Instructions:
    1. Explain deductible concepts in simple terms
    2. Provide examples specific to their insurance type
    3. Use the calculation tool to show precise premium savings with different deductibles
    4. Give clear recommendations based on their situation and calculations
    
    Response:
    """
    
    execution_settings = OpenAIChatPromptExecutionSettings(
        service_id="azure_openai",
        max_tokens=600,
        temperature=0.7,
        function_call_behavior=FunctionChoiceBehavior.Auto()
    )
    
    return kernel.add_function(
        function_name="deductible_advisor",
        plugin_name="InsurancePlugin",
        prompt=prompt,
        prompt_execution_settings=execution_settings
    )

class DeductibleCalculator:
    """Calculate deductible impacts"""
    
    @kernel_function(
        name="calculate_premium_savings",
        description="Calculate premium savings with different deductible amounts"
    )
    def calculate_premium_savings(self, current_premium: str, deductible_change: str) -> str:
        """Calculate premium impact of deductible changes"""
        try:
            premium = float(current_premium)
            deductible_diff = float(deductible_change)
            
            # Simple calculation: higher deductible = lower premium
            savings_percent = min(deductible_diff / 1000 * 0.05, 0.3)  # Max 30% savings
            new_premium = premium * (1 - savings_percent)
            annual_savings = (premium - new_premium) * 12
            
            result = {
                "current_annual_premium": premium * 12,
                "new_annual_premium": new_premium * 12,
                "annual_savings": round(annual_savings, 2),
                "deductible_increase": deductible_diff
            }
            
            return json.dumps(result)
            
        except ValueError:
            return json.dumps({"error": "Invalid premium or deductible values"})

async def test_deductible_advisor():
    """Test the deductible advisor with sample questions"""
    kernel = await create_kernel()
    
    # Add calculator
    calc = DeductibleCalculator()
    kernel.add_plugin(calc, plugin_name="DeductibleCalc")
    
    # Create advisor
    advisor = await create_deductible_advisor(kernel)
    
    # Test questions
    test_cases = [
        {
            "question": "What's a deductible and how does it affect my premium?",
            "insurance_type": "auto",
            "current_premium": "150"
        },
        {
            "question": "Should I choose a high or low deductible for my auto insurance?",
            "insurance_type": "auto", 
            "current_premium": "200"
        }
    ]
    
    for case in test_cases:
        print(f"\n? Question: {case['question']}")
        result = await kernel.invoke(advisor, **case)
        print(f"Answer: {result}")
```

**Your Task**: Now let's test your deductible advisor function. Update your `main()` function to comment out the step 6 code and test your deductible advisor instead:

```python
async def main():
    """Main function to test deductible advisor"""
    
    # Step 6 code (comment out for now)
    # await test_complete_workflow()
    
    # Exercise code
    await test_deductible_advisor()
```

Run the updated code. You should see:
1. ✅ Kernel created successfully!
2. Two deductible-related questions being answered with explanations
3. Recommendations based on the customer's premium and insurance type

This demonstrates how you can create your own specialized insurance functions using Semantic Kernel patterns.

---

## Troubleshooting

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Kernel Creation Failed** | ImportError or connection timeout | Check .env file and Azure OpenAI credentials |
| **Function Not Found** | KeyError when invoking functions | Verify function name and plugin registration |
| **Prompt Template Error** | Variable substitution fails | Check {{$variable}} syntax in templates |
| **AI Service Error** | 401/403 HTTP errors | Validate API key and endpoint configuration |
| **Import Error** | ModuleNotFoundError | Run `pip install semantic-kernel[azure]` and verify installation |
| **Async Error** | RuntimeError: no running event loop | Ensure using `asyncio.run()` for top-level calls |

### Complete Debugging Setup

```python
import logging
import sys

def setup_debugging():
    """Setup comprehensive debugging"""
    
    # Enable detailed logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('sk_debug.log')
        ]
    )
    
    # Test environment
    required_env_vars = [
        'AZURE_OPENAI_API_KEY',
        'AZURE_OPENAI_ENDPOINT', 
        'AZURE_OPENAI_DEPLOYMENT_NAME',
        'AZURE_OPENAI_API_VERSION'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"? Missing environment variables: {missing_vars}")
        return False
    
    print("? All environment variables found")
    return True

async def run_complete_test():
    """Run all tests in sequence"""
    
    if not setup_debugging():
        return
        
    try:
        print("Testing kernel creation...")
        kernel = await create_kernel()
        
        print("Testing AI connection...")
        await test_ai_connection(kernel)
        
        print("Testing semantic functions...")
        function = await create_insurance_assistant(kernel)
        await test_semantic_function(kernel, function)
        
        print("Testing native functions...")
        await test_native_functions(kernel)
        
        print("Testing complete workflow...")
        await test_complete_workflow()
        
        print("All tests passed!")
        
    except Exception as e:
        print(f"? Test failed: {e}")
        logging.error(f"Test failure: {e}", exc_info=True)

# Run comprehensive test
if __name__ == "__main__":
    asyncio.run(run_complete_test())
```

---

## Success Criteria

- Kernel successfully created and configured
- Azure OpenAI service properly integrated
- Basic semantic function responds correctly
- Multi-variable prompt templates work
- Native functions integrated and callable
- Hybrid workflow combining AI and calculations functions
- All code blocks execute without errors
- Proper error handling and debugging setup

---

## Next Steps

**Congratulations!** You now have a solid foundation in Semantic Kernel fundamentals. 

**Next Module**: [Lab 1: Plugin Creation](../labs/lab1-policy-retriever.md)

**Resources**:
- [Troubleshooting Guide](../docs/troubleshooting.md)
- [Insurance Context Reference](../docs/insurance-context.md)
- [Complete Workshop Guide](../README.md)
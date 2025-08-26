# Lab 1: Policy Data Retriever Plugin

**Duration**: 45 minutes  
**Objective**: Create a plugin that fetches insurance policy details based on customer profile.

## Learning Outcomes
By completing this lab, you will be able to:
- Understand semantic vs. native functions
- Create reusable plugins for insurance data
- Implement policy data simulation
- Handle structured data in Semantic Kernel
- Build comprehensive policy recommendation systems

## Prerequisites
- Completed [Module 1: SK Fundamentals](../modules/01-sk-fundamentals.md)
- Understanding of [Insurance Domain Context](../docs/insurance-context.md)
- Working Semantic Kernel environment

---

## Sample Customer Profiles (Test Data)

```python
SAMPLE_CUSTOMERS = [
    {
        "id": "CUST001",
        "age": 35,
        "income": 80000,
        "family_size": 4,
        "existing_policies": [],
        "risk_factors": ["none"]
    },
    {
        "id": "CUST002", 
        "age": 28,
        "income": 65000,
        "family_size": 2,
        "existing_policies": ["auto"],
        "risk_factors": ["smoker"]
    },
    {
        "id": "CUST003",
        "age": 45,
        "income": 120000,
        "family_size": 3,
        "existing_policies": ["home", "auto"],
        "risk_factors": ["high-risk-job"]
    }
]
```

---

## Step 1: Semantic Function with Prompt Template

### Create Policy Recommendation Function

```python
import asyncio
import json
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import kernel_function
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

async def create_policy_retriever_plugin():
    """Create the complete Policy Data Retriever plugin"""
    
    # Create kernel
    kernel = Kernel()
    service = AzureChatCompletion(
        service_id="azure_openai",
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    kernel.add_service(service)
    
    # Semantic function prompt template
    policy_recommendation_prompt = """
    Given a customer profile, analyze their needs and recommend suitable insurance policies.

    Customer Profile:
    - Age: {{$age}}
    - Income: ${{$income}}
    - Family Size: {{$family_size}}
    - Existing Policies: {{$existing_policies}}
    - Risk Factors: {{$risk_factors}}

    Available Policy Types:
    - Life Insurance (term, whole, universal)
    - Health Insurance (individual, family, HSA)
    - Auto Insurance (liability, comprehensive, collision)
    - Home Insurance (homeowners, renters, condo)
    - Disability Insurance (short-term, long-term)

    Provide recommendations in JSON format with policy type, coverage amount, estimated premium, and reasoning.
    Format your response as valid JSON with this structure:
    {
        "recommendations": [
            {
                "policy_type": "string",
                "coverage_amount": number,
                "estimated_premium": number,
                "reasoning": "string"
            }
        ]
    }
    """
    
    # Create execution settings
    execution_settings = OpenAIChatPromptExecutionSettings(
        service_id="azure_openai",
        max_tokens=800,
        temperature=0.3
    )
    
    # Add semantic function
    policy_recommender = kernel.add_function(
        function_name="recommend_policies",
        plugin_name="PolicyRetriever",
        prompt=policy_recommendation_prompt,
        prompt_execution_settings=execution_settings
    )
    
    return kernel, policy_recommender
```

---

## Step 2: Native Function for Policy Database Simulation

### Create Policy Database Class

```python
class PolicyDatabase:
    """Native function class for policy database simulation"""
    
    @kernel_function(
        name="get_policy_details",
        description="Retrieve detailed policy information from database"
    )
    def get_policy_details(self, policy_type: str) -> str:
        """Simulate database lookup for policy details"""
        
        policy_db = {
            "life": {
                "base_premium": 50,
                "max_coverage": 1000000,
                "min_age": 18,
                "max_age": 80,
                "available_types": ["term", "whole", "universal"]
            },
            "health": {
                "base_premium": 400,
                "max_coverage": 50000,
                "min_age": 18,
                "max_age": 65,
                "available_types": ["individual", "family", "hsa"]
            },
            "auto": {
                "base_premium": 150,
                "max_coverage": 100000,
                "min_age": 16,
                "max_age": 85,
                "available_types": ["liability", "comprehensive", "collision"]
            },
            "home": {
                "base_premium": 120,
                "max_coverage": 500000,
                "min_age": 18,
                "max_age": 99,
                "available_types": ["homeowners", "renters", "condo"]
            },
            "disability": {
                "base_premium": 80,
                "max_coverage": 200000,
                "min_age": 18,
                "max_age": 65,
                "available_types": ["short-term", "long-term"]
            }
        }
        
        result = policy_db.get(policy_type.lower(), {"error": "Policy type not found"})
        return json.dumps(result, indent=2)
    
    @kernel_function(
        name="calculate_premium_estimate",
        description="Calculate estimated premium based on customer profile"
    )
    def calculate_premium_estimate(self, policy_type: str, age: str, income: str, coverage_amount: str) -> str:
        """Calculate premium estimate based on customer factors"""
        
        try:
            age_int = int(age) if age.isdigit() else 35
            income_int = int(income) if income.isdigit() else 50000
            coverage_int = int(coverage_amount) if coverage_amount.isdigit() else 100000
            
            # Base rates per policy type
            base_rates = {
                "life": 0.5,
                "health": 0.08,
                "auto": 0.02,
                "home": 0.003,
                "disability": 0.015
            }
            
            base_rate = base_rates.get(policy_type.lower(), 0.01)
            
            # Age factor
            if age_int < 30:
                age_factor = 0.8
            elif age_int < 50:
                age_factor = 1.0
            else:
                age_factor = 1.5
            
            # Income factor (higher income = lower risk in some cases)
            if income_int > 100000:
                income_factor = 0.9
            elif income_int > 50000:
                income_factor = 1.0
            else:
                income_factor = 1.1
            
            annual_premium = coverage_int * base_rate * age_factor * income_factor
            monthly_premium = annual_premium / 12
            
            result = {
                "policy_type": policy_type,
                "annual_premium": round(annual_premium, 2),
                "monthly_premium": round(monthly_premium, 2),
                "factors_applied": {
                    "age_factor": age_factor,
                    "income_factor": income_factor,
                    "base_rate": base_rate
                }
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Invalid input: {str(e)}"})
```

---

## Step 3: Complete Plugin Integration and Testing

### Test Policy Retriever Plugin

```python
async def test_policy_retriever_plugin():
    """Test the Policy Data Retriever plugin with sample customers"""
    
    print("Testing Policy Data Retriever Plugin")
    print("=" * 50)
    
    # Create plugin
    kernel, policy_recommender = await create_policy_retriever_plugin()
    
    # Add native functions
    policy_db = PolicyDatabase()
    kernel.add_plugin(policy_db, plugin_name="PolicyDB")
    
    # Test with sample customers
    for i, customer in enumerate(SAMPLE_CUSTOMERS, 1):
        print(f"\nTesting Customer {i} (ID: {customer['id']})")
        print(f"Profile: {customer['age']} years old, ${customer['income']:,} income, {customer['family_size']} family members")
        
        try:
            # Get policy recommendations using semantic function
            recommendations = await kernel.invoke(
                policy_recommender,
                age=str(customer['age']),
                income=str(customer['income']),
                family_size=str(customer['family_size']),
                existing_policies=str(customer['existing_policies']),
                risk_factors=str(customer['risk_factors'])
            )
            
            print(f"AI Recommendations: {recommendations}")
            
            # Test native functions
            print(f"\nTesting database lookup for life insurance:")
            life_details = await kernel.invoke(
                kernel.get_function("PolicyDB", "get_policy_details"),
                policy_type="life"
            )
            print(f"Database Details: {life_details}")
            
            # Test premium calculation
            print(f"\nTesting premium calculation:")
            premium_estimate = await kernel.invoke(
                kernel.get_function("PolicyDB", "calculate_premium_estimate"),
                policy_type="life",
                age=str(customer['age']),
                income=str(customer['income']),
                coverage_amount="500000"
            )
            print(f"Premium Estimate: {premium_estimate}")
            
        except Exception as e:
            print(f"? Error processing customer {customer['id']}: {str(e)}")
        
        print("-" * 50)

# Extended exercise implementation
async def test_extended_features():
    """Test extended plugin features with filters"""
    
    print("\nTesting Extended Features")
    print("=" * 40)
    
    kernel, _ = await create_policy_retriever_plugin()
    policy_db = PolicyDatabase()
    kernel.add_plugin(policy_db, plugin_name="PolicyDB")
    
    # Test age bracket filtering
    age_brackets = [25, 35, 45, 55, 65]
    
    for age in age_brackets:
        print(f"\nAge Bracket: {age}")
        
        # Test different policy types for this age
        for policy_type in ["life", "health", "disability"]:
            try:
                estimate = await kernel.invoke(
                    kernel.get_function("PolicyDB", "calculate_premium_estimate"),
                    policy_type=policy_type,
                    age=str(age),
                    income="75000",
                    coverage_amount="300000"
                )
                
                estimate_data = json.loads(estimate)
                monthly = estimate_data.get('monthly_premium', 0)
                print(f"  {policy_type.title()}: ${monthly:.2f}/month")
                
            except Exception as e:
                print(f"  {policy_type.title()}: Error - {str(e)}")

# Main execution function
async def run_lab1():
    """Run complete Lab 1 workflow"""
    
    try:
        # Test basic plugin functionality
        await test_policy_retriever_plugin()
        
        # Test extended features
        await test_extended_features()
        
        print("\n? Lab 1 completed successfully!")
        
    except Exception as e:
        print(f"? Lab 1 failed: {str(e)}")
        import traceback
        traceback.print_exc()

# Run the lab
if __name__ == "__main__":
    asyncio.run(run_lab1())
```

---

## Extended Exercise: Add Risk Assessment Features

### Challenge: Enhance your plugin with risk assessment capabilities

```python
class RiskAssessmentPlugin:
    """Enhanced plugin with risk assessment features"""
    
    @kernel_function(
        name="assess_customer_risk",
        description="Assess customer risk factors for insurance underwriting"
    )
    def assess_customer_risk(self, age: str, health_status: str, occupation: str, lifestyle: str) -> str:
        """Assess risk factors and return risk score"""
        
        try:
            age_int = int(age) if age.isdigit() else 35
            risk_score = 100  # Base score (lower is better)
            
            # Age-based risk
            if age_int < 25:
                risk_score += 20  # Young drivers higher risk
            elif age_int > 65:
                risk_score += 30  # Senior health risks
            
            # Health-based risk
            health_risks = {
                "excellent": -10,
                "good": 0,
                "fair": 15,
                "poor": 40
            }
            risk_score += health_risks.get(health_status.lower(), 0)
            
            # Occupation-based risk
            high_risk_occupations = ["pilot", "construction", "mining", "police"]
            if any(job in occupation.lower() for job in high_risk_occupations):
                risk_score += 25
            
            # Lifestyle-based risk
            if "smoker" in lifestyle.lower():
                risk_score += 50
            if "extreme_sports" in lifestyle.lower():
                risk_score += 30
            
            # Determine risk category
            if risk_score <= 90:
                category = "Low Risk"
                premium_multiplier = 0.8
            elif risk_score <= 120:
                category = "Standard Risk"
                premium_multiplier = 1.0
            elif risk_score <= 150:
                category = "Moderate Risk"
                premium_multiplier = 1.3
            else:
                category = "High Risk"
                premium_multiplier = 1.8
            
            result = {
                "risk_score": risk_score,
                "risk_category": category,
                "premium_multiplier": premium_multiplier,
                "factors_considered": {
                    "age": age_int,
                    "health_status": health_status,
                    "occupation": occupation,
                    "lifestyle": lifestyle
                }
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Risk assessment failed: {str(e)}"})

# Test the enhanced plugin
async def test_risk_assessment():
    """Test risk assessment functionality"""
    
    print("Testing Risk Assessment Plugin")
    print("=" * 40)
    
    kernel, _ = await create_policy_retriever_plugin()
    
    # Add risk assessment plugin
    risk_plugin = RiskAssessmentPlugin()
    kernel.add_plugin(risk_plugin, plugin_name="RiskAssessment")
    
    # Test different risk profiles
    risk_profiles = [
        {
            "name": "Low Risk Customer",
            "age": "30",
            "health_status": "excellent",
            "occupation": "teacher",
            "lifestyle": "non-smoker, moderate exercise"
        },
        {
            "name": "High Risk Customer",
            "age": "45",
            "health_status": "fair",
            "occupation": "construction worker",
            "lifestyle": "smoker, extreme sports"
        },
        {
            "name": "Senior Customer",
            "age": "68",
            "health_status": "good",
            "occupation": "retired accountant",
            "lifestyle": "non-smoker, golf"
        }
    ]
    
    for profile in risk_profiles:
        print(f"\n{profile['name']}")
        
        risk_result = await kernel.invoke(
            kernel.get_function("RiskAssessment", "assess_customer_risk"),
            age=profile["age"],
            health_status=profile["health_status"],
            occupation=profile["occupation"],
            lifestyle=profile["lifestyle"]
        )
        
        print(f"Risk Assessment: {risk_result}")
        
        # Calculate adjusted premium
        risk_data = json.loads(risk_result)
        base_premium = 100  # Base monthly premium
        adjusted_premium = base_premium * risk_data.get("premium_multiplier", 1.0)
        
        print(f"Base Premium: ${base_premium}/month")
        print(f"Risk-Adjusted Premium: ${adjusted_premium:.2f}/month")
        print(f"Risk Category: {risk_data.get('risk_category', 'Unknown')}")
        
        print("-" * 40)

# Add to main function
async def run_extended_lab1():
    """Run Lab 1 with extended features"""
    
    try:
        # Run basic lab
        await run_lab1()
        
        # Run risk assessment tests
        await test_risk_assessment()
        
        print("\nExtended Lab 1 completed successfully!")
        
    except Exception as e:
        print(f"? Extended Lab 1 failed: {str(e)}")
        import traceback
        traceback.print_exc()

# Uncomment to run extended version: asyncio.run(run_extended_lab1())
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|--------|----------|
| **API Key Error** | Invalid Azure OpenAI credentials | Verify `.env` file configuration |
| **Import Error** | Missing semantic-kernel package | Run `pip install semantic-kernel[azure]` |
| **JSON Format Error** | AI not returning valid JSON | Improve prompt template specificity |
| **Function Not Found** | Plugin not registered correctly | Check `kernel.add_plugin()` call |
| **Premium Calculation Error** | Invalid numeric inputs | Add input validation and error handling |

### Debug Helper

```python
def debug_lab1():
    """Debug Lab 1 issues"""
    
    print("Debugging Lab 1...")
    
    # Check environment
    required_vars = ['AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_DEPLOYMENT_NAME']
    for var in required_vars:
        if os.getenv(var):
            print(f"? {var}: Set")
        else:
            print(f"? {var}: Not set")
    
    # Test JSON parsing
    try:
        test_json = '{"test": "value"}'
        json.loads(test_json)
        print("? JSON parsing: OK")
    except:
        print("? JSON parsing: Failed")
    
    # Test async execution
    try:
        asyncio.get_event_loop()
        print("? Async environment: OK")
    except:
        print("? Async environment: Failed")

# Uncomment to run debug: debug_lab1()
```

---

## ? Success Criteria

- Plugin successfully processes all 3 sample customers
- Output includes policy type, premium estimate, and reasoning
- Native function properly simulates database lookup
- Extended features (age brackets, income tiers) work correctly
- Error handling prevents crashes with invalid input
- All code blocks execute without syntax or runtime errors
- Risk assessment functionality works (if implementing extended exercise)

---

## Next Steps

**Congratulations!** You've successfully created a comprehensive policy data retriever plugin.

**Next Lab**: [Lab 2: Single Agent - PolicyAdvisorAgent](lab2-policy-advisor.md)

**Key Takeaways**:
- Semantic functions excel at natural language processing and reasoning
- Native functions provide deterministic calculations and data access
- Combining both creates powerful, flexible plugins
- Proper error handling is crucial for production-ready plugins

**Resources**:
- [Troubleshooting Guide](../docs/troubleshooting.md)
- [Module 2: Plugin Development](../modules/02-plugin-development.md)
- [Insurance Context Reference](../docs/insurance-context.md)
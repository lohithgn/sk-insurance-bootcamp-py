# Module 2: Plugin Development Patterns

**Duration**: 45 minutes  
**Objective**: Master advanced plugin development patterns and best practices for insurance applications.

## Learning Outcomes
By completing this module, you will be able to:
- Design reusable plugin architectures
- Implement advanced semantic and native function patterns
- Handle complex data structures and API integrations
- Apply plugin composition and orchestration techniques
- Build production-ready insurance plugins

## Prerequisites
- Completed [Module 1: SK Fundamentals](01-sk-fundamentals.md)
- Successfully completed [Lab 1: Policy Retriever Plugin](../labs/lab1-policy-retriever.md)
- Understanding of [Insurance Domain Context](../docs/insurance-context.md)

---

## Plugin Architecture Patterns

### Single Responsibility Plugins
Each plugin should focus on one specific domain or function:

```python
# Good - Focused plugin
class PremiumCalculatorPlugin:
    """Dedicated to premium calculations only"""
    
    @kernel_function(name="calculate_life_premium")
    def calculate_life_premium(self, age: str, coverage: str) -> str:
        # Implementation
        pass
    
    @kernel_function(name="calculate_auto_premium") 
    def calculate_auto_premium(self, age: str, vehicle_type: str) -> str:
        # Implementation
        pass

# Avoid - Mixed responsibilities
class InsuranceEverythingPlugin:
    """Tries to do too much"""
    def calculate_premium(self): pass
    def send_email(self): pass  # Should be separate
    def generate_pdf(self): pass  # Should be separate
```

### Plugin Composition Pattern

```python
class InsurancePluginComposer:
    """Compose multiple specialized plugins"""
    
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        
        # Add specialized plugins
        self.kernel.add_plugin(PremiumCalculatorPlugin(), "PremiumCalc")
        self.kernel.add_plugin(RiskAssessmentPlugin(), "RiskAssess")
        self.kernel.add_plugin(PolicyDataPlugin(), "PolicyData")
        self.kernel.add_plugin(DocumentGeneratorPlugin(), "DocGen")
    
    async def get_complete_quote(self, customer_profile: dict) -> dict:
        """Orchestrate multiple plugins for complete quote"""
        
        # Step 1: Assess risk
        risk_result = await self.kernel.invoke(
            self.kernel.get_function("RiskAssess", "assess_customer_risk"),
            **customer_profile
        )
        
        # Step 2: Calculate premium based on risk
        premium_result = await self.kernel.invoke(
            self.kernel.get_function("PremiumCalc", "calculate_adjusted_premium"),
            risk_data=risk_result,
            coverage_amount=customer_profile.get("coverage_amount")
        )
        
        # Step 3: Generate policy documents
        policy_docs = await self.kernel.invoke(
            self.kernel.get_function("DocGen", "generate_policy_documents"),
            customer_profile=json.dumps(customer_profile),
            premium_data=premium_result
        )
        
        return {
            "risk_assessment": risk_result,
            "premium_calculation": premium_result,
            "policy_documents": policy_docs
        }
```

---

## Advanced Native Function Patterns

### Data Validation and Sanitization

```python
class DataValidationPlugin:
    """Advanced data validation for insurance inputs"""
    
    @kernel_function(
        name="validate_customer_profile",
        description="Validate and sanitize customer profile data"
    )
    def validate_customer_profile(self, profile_json: str) -> str:
        """Validate customer profile with comprehensive checks"""
        
        try:
            profile = json.loads(profile_json)
            errors = []
            sanitized = {}
            
            # Age validation
            age = profile.get("age")
            if age is None:
                errors.append("Age is required")
            elif not isinstance(age, (int, str)) or not str(age).isdigit():
                errors.append("Age must be a number")
            elif int(age) < 0 or int(age) > 120:
                errors.append("Age must be between 0 and 120")
            else:
                sanitized["age"] = int(age)
            
            # Income validation
            income = profile.get("income")
            if income is None:
                errors.append("Income is required")
            else:
                try:
                    # Handle various income formats
                    income_str = str(income).replace(",", "").replace("$", "")
                    if income_str.lower().endswith("k"):
                        income_val = float(income_str[:-1]) * 1000
                    else:
                        income_val = float(income_str)
                    
                    if income_val < 0:
                        errors.append("Income cannot be negative")
                    elif income_val > 10000000:  # $10M cap
                        errors.append("Income exceeds maximum allowed")
                    else:
                        sanitized["income"] = int(income_val)
                except ValueError:
                    errors.append("Invalid income format")
            
            # Email validation (if provided)
            email = profile.get("email")
            if email and not self._is_valid_email(email):
                errors.append("Invalid email format")
            elif email:
                sanitized["email"] = email.lower().strip()
            
            # State validation
            state = profile.get("state")
            if state:
                valid_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
                              "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                              "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                              "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                              "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
                
                if state.upper() not in valid_states:
                    errors.append(f"Invalid state code: {state}")
                else:
                    sanitized["state"] = state.upper()
            
            result = {
                "valid": len(errors) == 0,
                "errors": errors,
                "sanitized_profile": sanitized
            }
            
            return json.dumps(result, indent=2)
            
        except json.JSONDecodeError:
            return json.dumps({
                "valid": False,
                "errors": ["Invalid JSON format"],
                "sanitized_profile": {}
            })
    
    def _is_valid_email(self, email: str) -> bool:
        """Basic email validation"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

# Usage example
async def test_validation():
    kernel = Kernel()
    # Add service...
    
    validator = DataValidationPlugin()
    kernel.add_plugin(validator, "Validator")
    
    test_profile = {
        "age": "35",
        "income": "$80,000",
        "email": "test@example.com",
        "state": "ca"
    }
    
    result = await kernel.invoke(
        kernel.get_function("Validator", "validate_customer_profile"),
        profile_json=json.dumps(test_profile)
    )
    
    print(f"Validation result: {result}")
```

### External API Integration Pattern

```python
import aiohttp
from typing import Optional

class ExternalAPIPlugin:
    """Pattern for integrating external APIs securely"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("EXTERNAL_API_KEY")
        self.base_url = "https://api.example.com/v1"
    
    @kernel_function(
        name="get_credit_score",
        description="Retrieve credit score from external service"
    )
    async def get_credit_score(self, ssn: str, consent: str = "false") -> str:
        """Get credit score with proper consent and security"""
        
        # Verify consent
        if consent.lower() != "true":
            return json.dumps({
                "error": "Customer consent required for credit check",
                "credit_score": None
            })
        
        # Validate SSN format (basic)
        if not self._is_valid_ssn(ssn):
            return json.dumps({
                "error": "Invalid SSN format",
                "credit_score": None
            })
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "ssn": self._encrypt_ssn(ssn),  # Always encrypt sensitive data
                    "purpose": "insurance_underwriting"
                }
                
                async with session.post(
                    f"{self.base_url}/credit-score",
                    headers=headers,
                    json=payload,
                    timeout=10
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        return json.dumps({
                            "credit_score": data.get("score"),
                            "score_range": data.get("range"),
                            "factors": data.get("factors", []),
                            "timestamp": data.get("timestamp")
                        })
                    else:
                        return json.dumps({
                            "error": f"API error: {response.status}",
                            "credit_score": None
                        })
                        
        except aiohttp.ClientTimeout:
            return json.dumps({
                "error": "Credit service timeout",
                "credit_score": None
            })
        except Exception as e:
            return json.dumps({
                "error": f"Credit service error: {str(e)}",
                "credit_score": None
            })
    
    def _is_valid_ssn(self, ssn: str) -> bool:
        """Validate SSN format"""
        import re
        # Basic SSN format validation (XXX-XX-XXXX)
        pattern = r'^\d{3}-\d{2}-\d{4}$'
        return re.match(pattern, ssn) is not None
    
    def _encrypt_ssn(self, ssn: str) -> str:
        """Encrypt SSN for transmission (simplified example)"""
        # In production, use proper encryption
        import hashlib
        return hashlib.sha256(ssn.encode()).hexdigest()
```

---

## Advanced Semantic Function Patterns

### Context-Aware Semantic Functions

```python
async def create_context_aware_advisor(kernel: Kernel):
    """Create semantic function that maintains and uses context"""
    
    context_aware_prompt = """
    You are an insurance advisor with access to comprehensive customer context.
    
    Current Customer Context:
    {{$customer_context}}
    
    Previous Conversation:
    {{$conversation_history}}
    
    Current Query: {{$query}}
    
    Available Tools and Data:
    - Credit score information
    - Policy database
    - Risk assessment tools
    - Premium calculators
    
    Instructions:
    1. Review the customer's complete context and history
    2. Consider how this query relates to previous discussions
    3. Use available tools when specific calculations or data lookups are needed
    4. Provide personalized advice that builds on the existing relationship
    5. Reference previous conversations naturally when relevant
    6. Ask follow-up questions to deepen understanding
    
    Response Format (JSON):
    {
        "analysis": "Analysis considering full context",
        "recommendations": [
            {
                "type": "insurance_product",
                "product": "specific product name",
                "reasoning": "why this fits their complete profile",
                "priority": "high/medium/low",
                "next_action": "specific next step"
            }
        ],
        "context_references": ["reference to previous discussions"],
        "follow_up_questions": ["questions to deepen relationship"],
        "tools_used": ["list of tools/functions called"]
    }
    
    Remember: You're building a long-term advisory relationship, not just answering isolated questions.
    """
    
    execution_settings = OpenAIChatPromptExecutionSettings(
        service_id="azure_openai",
        max_tokens=1200,
        temperature=0.4,  # Slightly higher for more personalized responses
        function_call_behavior=FunctionCallBehavior.EnableFunctions(
            auto_invoke=True, filters={}
        )
    )
    
    return kernel.add_function(
        function_name="context_aware_advisor",
        plugin_name="AdvancedAdvisor",
        prompt=context_aware_prompt,
        prompt_execution_settings=execution_settings
    )
```

### Multi-Step Reasoning Pattern

```python
async def create_multi_step_analyzer(kernel: Kernel):
    """Create semantic function that performs multi-step analysis"""
    
    multi_step_prompt = """
    You are performing a comprehensive insurance needs analysis. Break this down into clear steps.
    
    Customer Information: {{$customer_info}}
    Analysis Type: {{$analysis_type}}
    
    Perform this analysis in the following steps:
    
    STEP 1: SITUATION ANALYSIS
    - Current life stage and circumstances
    - Financial situation assessment
    - Existing coverage evaluation
    - Risk exposure identification
    
    STEP 2: NEEDS IDENTIFICATION
    - Protection needs (what risks to cover)
    - Coverage amount requirements
    - Budget constraints
    - Timeline considerations
    
    STEP 3: PRODUCT MATCHING
    - Suitable insurance products
    - Coverage options comparison
    - Cost-benefit analysis
    - Alternative solutions
    
    STEP 4: PRIORITIZATION
    - Most critical needs first
    - Budget allocation recommendations
    - Implementation timeline
    - Review schedule
    
    STEP 5: ACTION PLAN
    - Immediate actions required
    - Information needed from customer
    - Next appointment scheduling
    - Follow-up requirements
    
    Format your response as JSON with each step clearly labeled:
    {
        "step_1_situation": {
            "life_stage": "",
            "financial_assessment": "",
            "existing_coverage": "",
            "risk_exposure": ""
        },
        "step_2_needs": {
            "protection_needs": [],
            "coverage_amounts": {},
            "budget_constraints": "",
            "timeline": ""
        },
        "step_3_products": [
            {
                "product_type": "",
                "coverage_options": [],
                "cost_benefit": "",
                "alternatives": []
            }
        ],
        "step_4_priority": [
            {
                "priority_level": 1,
                "need": "",
                "reasoning": "",
                "budget_allocation": ""
            }
        ],
        "step_5_action_plan": {
            "immediate_actions": [],
            "information_needed": [],
            "next_steps": [],
            "follow_up_schedule": ""
        }
    }
    
    Be thorough, logical, and ensure each step builds on the previous ones.
    """
    
    execution_settings = OpenAIChatPromptExecutionSettings(
        service_id="azure_openai",
        max_tokens=2000,  # More tokens for comprehensive analysis
        temperature=0.2   # Lower temperature for structured analysis
    )
    
    return kernel.add_function(
        function_name="multi_step_analyzer",
        plugin_name="AdvancedAdvisor",
        prompt=multi_step_prompt,
        prompt_execution_settings=execution_settings
    )
```

---

## Plugin Orchestration Patterns

### Pipeline Pattern

```python
class InsurancePipeline:
    """Process customer through a series of plugin stages"""
    
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.stages = []
    
    def add_stage(self, plugin_name: str, function_name: str, stage_name: str):
        """Add a stage to the pipeline"""
        self.stages.append({
            "plugin": plugin_name,
            "function": function_name,
            "name": stage_name
        })
    
    async def process(self, initial_data: dict) -> dict:
        """Process data through all pipeline stages"""
        
        results = {
            "initial_data": initial_data,
            "stage_results": {},
            "final_result": None
        }
        
        current_data = initial_data.copy()
        
        for stage in self.stages:
            print(f"Processing stage: {stage['name']}")
            
            try:
                # Get the function
                func = self.kernel.get_function(stage["plugin"], stage["function"])
                
                # Process current data
                stage_result = await self.kernel.invoke(func, **current_data)
                
                # Store stage result
                results["stage_results"][stage["name"]] = stage_result
                
                # Parse result if JSON and update current_data
                try:
                    parsed_result = json.loads(str(stage_result))
                    if isinstance(parsed_result, dict):
                        current_data.update(parsed_result)
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, store as string result
                    current_data[f"{stage['name']}_result"] = str(stage_result)
                
            except Exception as e:
                print(f"Error in stage {stage['name']}: {e}")
                results["stage_results"][stage["name"]] = f"Error: {e}"
                # Continue to next stage despite error
        
        results["final_result"] = current_data
        return results

# Usage example
async def create_insurance_pipeline():
    """Create a comprehensive insurance processing pipeline"""
    
    kernel = await create_kernel()
    
    # Add all necessary plugins
    kernel.add_plugin(DataValidationPlugin(), "Validator")
    kernel.add_plugin(RiskAssessmentPlugin(), "RiskAssess")
    kernel.add_plugin(PremiumCalculatorPlugin(), "PremiumCalc")
    kernel.add_plugin(PolicyRecommendationPlugin(), "PolicyRec")
    
    # Create pipeline
    pipeline = InsurancePipeline(kernel)
    
    # Add stages in order
    pipeline.add_stage("Validator", "validate_customer_profile", "validation")
    pipeline.add_stage("RiskAssess", "assess_customer_risk", "risk_assessment")
    pipeline.add_stage("PremiumCalc", "calculate_premium_range", "premium_calculation")
    pipeline.add_stage("PolicyRec", "recommend_policies", "policy_recommendation")
    
    return pipeline

# Test the pipeline
async def test_pipeline():
    pipeline = await create_insurance_pipeline()
    
    customer_data = {
        "age": "35",
        "income": "80000",
        "family_size": "3",
        "state": "CA",
        "occupation": "software_engineer"
    }
    
    results = await pipeline.process(customer_data)
    
    print("Pipeline Results:")
    print(json.dumps(results, indent=2, default=str))
```

---

## Exercise: Build a Comprehensive Plugin System

### Challenge: Create a Multi-Plugin Insurance System

Your task is to build a comprehensive plugin system that demonstrates all the patterns covered:

```python
class ComprehensiveInsuranceSystem:
    """Complete insurance system using advanced plugin patterns"""
    
    def __init__(self):
        self.kernel = None
        self.pipeline = None
        self.context_store = {}
    
    async def initialize(self):
        """Initialize the complete system"""
        
        # Create kernel and add services
        self.kernel = Kernel()
        service = AzureChatCompletion(
            service_id="azure_openai",
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        self.kernel.add_service(service)
        
        # Add all plugins
        self.kernel.add_plugin(DataValidationPlugin(), "Validator")
        self.kernel.add_plugin(RiskAssessmentPlugin(), "RiskAssess")
        self.kernel.add_plugin(PremiumCalculatorPlugin(), "PremiumCalc")
        self.kernel.add_plugin(PolicyRecommendationPlugin(), "PolicyRec")
        self.kernel.add_plugin(ExternalAPIPlugin(), "ExternalAPI")
        
        # Add advanced semantic functions
        await create_context_aware_advisor(self.kernel)
        await create_multi_step_analyzer(self.kernel)
        
        # Create pipeline
        self.pipeline = InsurancePipeline(self.kernel)
        self.pipeline.add_stage("Validator", "validate_customer_profile", "validation")
        self.pipeline.add_stage("RiskAssess", "assess_customer_risk", "risk_assessment")
        self.pipeline.add_stage("PremiumCalc", "calculate_premium_range", "premium_calculation")
        
        print("? Comprehensive Insurance System initialized!")
    
    async def process_new_customer(self, customer_data: dict) -> dict:
        """Process a new customer through the complete system"""
        
        print("Processing new customer through comprehensive system")
        
        # Step 1: Run through pipeline
        pipeline_results = await self.pipeline.process(customer_data)
        
        # Step 2: Perform multi-step analysis
        analysis_result = await self.kernel.invoke(
            self.kernel.get_function("AdvancedAdvisor", "multi_step_analyzer"),
            customer_info=json.dumps(customer_data),
            analysis_type="comprehensive_needs_analysis"
        )
        
        # Step 3: Generate context-aware recommendations
        context_data = {
            "customer_profile": customer_data,
            "pipeline_results": pipeline_results,
            "analysis": analysis_result
        }
        
        recommendations = await self.kernel.invoke(
            self.kernel.get_function("AdvancedAdvisor", "context_aware_advisor"),
            customer_context=json.dumps(context_data),
            conversation_history="New customer - initial consultation",
            query="Provide comprehensive insurance recommendations based on complete analysis"
        )
        
        # Store context for future interactions
        customer_id = customer_data.get("id", "anonymous")
        self.context_store[customer_id] = {
            "profile": customer_data,
            "pipeline_results": pipeline_results,
            "analysis": analysis_result,
            "recommendations": recommendations,
            "conversation_history": ["Initial comprehensive analysis completed"]
        }
        
        return {
            "customer_id": customer_id,
            "pipeline_results": pipeline_results,
            "multi_step_analysis": analysis_result,
            "final_recommendations": recommendations,
            "context_stored": True
        }
    
    async def handle_follow_up_query(self, customer_id: str, query: str) -> dict:
        """Handle follow-up queries with full context"""
        
        if customer_id not in self.context_store:
            return {"error": "Customer context not found"}
        
        context = self.context_store[customer_id]
        
        # Add query to conversation history
        context["conversation_history"].append(f"Customer: {query}")
        
        # Use context-aware advisor
        response = await self.kernel.invoke(
            self.kernel.get_function("AdvancedAdvisor", "context_aware_advisor"),
            customer_context=json.dumps(context),
            conversation_history="\n".join(context["conversation_history"]),
            query=query
        )
        
        # Update conversation history
        context["conversation_history"].append(f"Advisor: {response}")
        
        return {
            "customer_id": customer_id,
            "query": query,
            "response": response,
            "context_updated": True
        }

# Test the comprehensive system
async def test_comprehensive_system():
    """Test the complete comprehensive system"""
    
    system = ComprehensiveInsuranceSystem()
    await system.initialize()
    
    # Test new customer processing
    customer_data = {
        "id": "CUST_TEST_001",
        "age": "35",
        "income": "85000",
        "family_size": "3",
        "state": "CA",
        "occupation": "software_engineer",
        "existing_coverage": ["basic_health"],
        "risk_factors": ["none"]
    }
    
    print("\nTesting new customer processing...")
    initial_result = await system.process_new_customer(customer_data)
    print("Initial processing completed!")
    
    # Test follow-up queries
    follow_up_queries = [
        "What if I want to increase my life insurance coverage?",
        "How much would it cost to add my spouse to the policy?",
        "What happens if I change jobs?"
    ]
    
    for query in follow_up_queries:
        print(f"\nFollow-up query: {query}")
        result = await system.handle_follow_up_query("CUST_TEST_001", query)
        print("Response generated with full context!")
    
    return system

# Run the comprehensive test
if __name__ == "__main__":
    asyncio.run(test_comprehensive_system())
```

---

## ? Success Criteria

By completing this module, you should be able to:

- Design and implement single-responsibility plugins
- Create plugin composition and orchestration patterns  
- Implement advanced data validation and sanitization
- Integrate external APIs securely
- Build context-aware semantic functions
- Create multi-step reasoning workflows
- Implement pipeline processing patterns
- Build comprehensive plugin systems
- Handle error scenarios gracefully
- Maintain customer context across interactions

---

## Next Steps

**Congratulations!** You've mastered advanced plugin development patterns.

**Next Module**: [Module 3: Single Agent Architecture](03-single-agent.md)

**Key Takeaways**:
- Plugin composition enables complex functionality through simple building blocks
- Context awareness dramatically improves user experience  
- Multi-step reasoning provides more thorough analysis
- Pipeline patterns ensure consistent processing
- Error handling and validation are critical for production systems

**Resources**:
- [Lab 2: PolicyAdvisorAgent](../labs/lab2-policy-advisor.md)
- [Advanced Topics Guide](../docs/advanced-topics.md)
- [Troubleshooting Guide](../docs/troubleshooting.md)
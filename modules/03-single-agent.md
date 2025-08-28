# Module 3: Single Agent Architecture with Semantic Kernel

**Duration**: 45 minutes  
**Objective**: Learn how to create intelligent conversational agents using Semantic Kernel's ChatCompletionAgent framework.

## Learning Outcomes
By completing this module, you will be able to:
- Create a ChatCompletionAgent that can hold conversations and use tools
- Configure agent personality through system instructions  
- Integrate plugins with agents for enhanced functionality
- Understand how agents automatically invoke functions based on conversation context
- Handle multi-turn conversations with built-in memory

## Prerequisites
- Completed [Module 2: Plugin Development](02-plugin-development.md)
- Understanding of SK fundamentals (Kernel, plugins, functions)

---

## Agent vs Function: The Key Difference

### ? Traditional Function Approach
```python
# Stateless, one-shot interactions
async def get_policy_info(policy_type: str) -> str:
    # Returns policy info but no conversation context
    return "Term Life: $50/month for $500K coverage"

# Each call is independent - no memory or conversation flow
result1 = await get_policy_info("life")  
result2 = await get_policy_info("health") 
# Agent doesn't remember first call when processing second
```

### ? Agent Approach - Conversational & Context-Aware
```python
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.contents.chat_history import ChatHistory

# Agent maintains conversation, uses tools automatically, remembers context
agent = ChatCompletionAgent(
    service_id="azure_openai",
    kernel=kernel,  # Kernel with plugins attached
    name="PolicyAgent", 
    instructions="You are a helpful insurance advisor..."
)

# Conversational flow with memory
chat_history = ChatHistory()
chat_history.add_user_message("I'm 35 and need life insurance")
response = await agent.invoke(chat_history)  # Agent remembers this context

chat_history.add_user_message("What would that cost me?") 
response = await agent.invoke(chat_history)  # Agent knows "that" refers to life insurance for 35-year-old
```

---

## Building a Multi-Step Policy Recommendation Agent

Let's build an agent that demonstrates a systematic, outcome-focused 5-step process for policy recommendations (the agent decides which tools to use for each step):

```python
import asyncio
import os
import json
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import (kernel_function, KernelArguments)
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

load_dotenv()

class SimpleInsuranceAgent:
    """A focused agent that recommends insurance policies using SK framework"""
    
    def __init__(self):
        self.kernel = None
        self.agent = None
        self.chat_history = ChatHistory()
    
    async def initialize(self):
        """Set up the kernel, plugins, and agent"""
        
        # 1. Create kernel with AI service
        self.kernel = Kernel()
        service = AzureChatCompletion(
            service_id="azure_openai",
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        self.kernel.add_service(service)
        
        # 2. Add plugin (tools) with realistic data for the agent
        policy_plugin = PolicyAdvisorPlugin()
        self.kernel.add_plugin(policy_plugin, plugin_name="PolicyTools")
        
        # 3. Create the agent with a clear, outcome-focused 5-step process
        self.agent = ChatCompletionAgent(
            kernel=self.kernel,
            name="InsuranceAdvisor",
            instructions="""You are a professional insurance advisor. For every policy recommendation, follow this systematic process:

STEP 1 ➜ UNDERSTAND THE CUSTOMER
Announce: "Step 1: Understanding your situation..."
Goal: Gather a complete picture of their profile and needs (age, income, family, goals)

STEP 2 ➜ IDENTIFY OPTIONS
Announce: "Step 2: Identifying suitable options..."  
Goal: Find policies that match their profile and constraints

STEP 3 ➜ DETERMINE COVERAGE
Announce: "Step 3: Calculating appropriate coverage..."
Goal: Determine how much insurance they need based on standard methods

STEP 4 ➜ ESTIMATE INVESTMENT
Announce: "Step 4: Estimating your investment..."
Goal: Estimate premiums for the recommended coverage

STEP 5 ➜ RECOMMEND SOLUTION
Announce: "Step 5: My recommendation..."
Goal: Provide clear advice with concise reasoning

Use the appropriate tools at your disposal to complete each step thoroughly. Keep responses concise and user-friendly.

Policies and defaults:
- Do Step 1 once per conversation. Build an internal CustomerProfile {age, income, dependents, debts, mortgage, goals, health_class, preferred_term} from chat history; keep it updated silently.
- Ask at most one non-redundant question per turn, only if it materially changes the outcome. Never ask for details already provided.
- If required information is missing, proceed with best-effort defaults and disclose under "Assumptions":
  - debts: 0; mortgage: 0; goals: income replacement (+ children's education if dependents > 0); health_class: Standard; preferred_term: 20 years if age < 40, otherwise 20–30 years.
- If the user requests a later step (e.g., premiums), answer that step immediately using the current profile + assumptions; do not return to Step 1.
- Start with the requested answer, then add brief context. Label any optional clarification "Optional".
- Keep steps concise and do not re-announce Step 1 after it has been completed.""",
            arguments=KernelArguments(settings = OpenAIChatPromptExecutionSettings(
                max_tokens=1000,
                temperature=0.7,
                function_choice_behavior=FunctionChoiceBehavior.Auto()
            ))
        )
        
        print("Simple Insurance Agent ready with 5-step process!")
    
    async def chat(self, user_message: str) -> str:
        """Have a conversation with the agent"""
        
        # Add user message and get agent response
        self.chat_history.add_user_message(user_message)
        response = await self.agent.get_response(self.chat_history)
        
        # Return the agent's response
        return response.content if response else "I didn't understand that. Could you try again?"

class PolicyAdvisorPlugin:
    """Plugin with realistic insurance data and calculations to support the 5-step process"""

    @kernel_function(
        name="search_available_policies",
        description="Search policies available for a customer's age and category (life, auto, home). Returns structured data."
    )
    def search_available_policies(self, age: int, category: str = "life") -> str:
        """Return realistic policy options as JSON"""
        policy_catalog = {
            "life": [
                {
                    "id": "TL20-2025",
                    "name": "SecureLife Term 20",
                    "type": "Term Life (20-year level)",
                    "min_age": 18,
                    "max_age": 65,
                    "min_coverage": 100_000,
                    "max_coverage": 5_000_000,
                    "features": ["Level premiums", "Convertible options", "Accelerated benefits rider"],
                    "best_for": "Young families needing affordable protection",
                },
                {
                    "id": "TL30-2025",
                    "name": "SecureLife Term 30",
                    "type": "Term Life (30-year level)",
                    "min_age": 18,
                    "max_age": 55,
                    "min_coverage": 250_000,
                    "max_coverage": 3_000_000,
                    "features": ["Level premiums", "Renewable", "Living benefits rider"],
                    "best_for": "Longer-term obligations like a mortgage",
                },
                {
                    "id": "WL-2025",
                    "name": "WholeLife Plus",
                    "type": "Whole Life",
                    "min_age": 18,
                    "max_age": 75,
                    "min_coverage": 50_000,
                    "max_coverage": 2_000_000,
                    "features": ["Cash value accumulation", "Dividend eligible"],
                    "best_for": "Lifetime coverage and savings component",
                },
                {
                    "id": "UL-2025",
                    "name": "FlexLife Universal",
                    "type": "Universal Life",
                    "min_age": 20,
                    "max_age": 70,
                    "min_coverage": 100_000,
                    "max_coverage": 1_500_000,
                    "features": ["Flexible premiums", "Adjustable death benefit"],
                    "best_for": "Flexible coverage with potential growth",
                },
            ],
            "auto": [
                {"id": "AUTO-STD", "name": "SafeDrive Standard", "type": "Full Coverage", "min_age": 18, "max_age": 85},
                {"id": "AUTO-PRM", "name": "SafeDrive Premium", "type": "Premium Coverage", "min_age": 25, "max_age": 85},
            ],
            "home": [
                {"id": "HOME-HO3", "name": "HomeShield HO-3", "type": "Homeowners", "coverage_types": ["dwelling", "property", "liability"]},
                {"id": "HOME-PREM", "name": "HomeShield Premium", "type": "Comprehensive", "coverage_types": ["dwelling", "property", "liability", "flood", "quake"]},
            ],
        }

        data = policy_catalog.get(category.lower(), [])
        eligible = []
        for p in data:
            if category.lower() in ("life", "auto"):
                if p.get("min_age", 0) <= age <= p.get("max_age", 200):
                    eligible.append(p)
            else:
                eligible.append(p)

        result = {
            "category": category.lower(),
            "criteria": {"age": age},
            "found": len(eligible),
            "policies": eligible,
        }
        return json.dumps(result, indent=2)

    @kernel_function(
        name="calculate_coverage_needs",
        description="Calculate recommended life insurance coverage using standard methods. Returns structured data."
    )
    def calculate_coverage_needs(self, annual_income: int, dependents: int = 0, debts: int = 0, mortgage: int = 0) -> str:
        """Return coverage analysis as JSON"""
        years = 10 if dependents > 0 else 5
        income_replacement = annual_income * years

        education = max(dependents, 0) * 100_000  # $100k per child (est.)
        dime_total = debts + (annual_income * 5) + mortgage + education

        human_life_value = int(annual_income * 20 * 0.75)

        methods = {
            "income_replacement": {"years": years, "amount": income_replacement},
            "dime": {"debts": debts, "income_multiple": annual_income * 5, "mortgage": mortgage, "education": education, "total": dime_total},
            "human_life_value": {"present_value_proxy": human_life_value},
        }

        amounts = [income_replacement, dime_total, human_life_value]
        recommended = round((sum(amounts) / len(amounts)) / 50_000) * 50_000

        result = {
            "inputs": {
                "annual_income": annual_income,
                "dependents": dependents,
                "debts": debts,
                "mortgage": mortgage,
            },
            "methods": methods,
            "recommended_coverage": int(recommended),
            "notes": "Average of standard methods rounded to nearest $50k",
        }
        return json.dumps(result, indent=2)

    @kernel_function(
        name="estimate_premiums",
        description="Estimate premiums for a given coverage and age using rate tables. Returns structured data."
    )
    def estimate_premiums(self, age: int, coverage_amount: int, policy_type: str = "term") -> str:
        """Return premium estimates as JSON"""
        rate_tables = {
            "term": {
                25: {"preferred": 0.12, "standard": 0.18, "substandard": 0.35},
                30: {"preferred": 0.15, "standard": 0.22, "substandard": 0.42},
                35: {"preferred": 0.20, "standard": 0.30, "substandard": 0.58},
                40: {"preferred": 0.32, "standard": 0.48, "substandard": 0.95},
                45: {"preferred": 0.52, "standard": 0.78, "substandard": 1.55},
                50: {"preferred": 0.88, "standard": 1.32, "substandard": 2.65},
                55: {"preferred": 1.45, "standard": 2.18, "substandard": 4.35},
                60: {"preferred": 2.35, "standard": 3.53, "substandard": 7.05},
            },
            "whole": {
                25: {"preferred": 2.15, "standard": 2.58, "substandard": 3.87},
                30: {"preferred": 2.65, "standard": 3.18, "substandard": 4.77},
                35: {"preferred": 3.35, "standard": 4.02, "substandard": 6.03},
                40: {"preferred": 4.25, "standard": 5.10, "substandard": 7.65},
                45: {"preferred": 5.45, "standard": 6.54, "substandard": 9.81},
                50: {"preferred": 7.15, "standard": 8.58, "substandard": 12.87},
                55: {"preferred": 9.55, "standard": 11.46, "substandard": 17.19},
                60: {"preferred": 12.85, "standard": 15.42, "substandard": 23.13},
            },
        }

        category_key = "term" if "term" in policy_type.lower() else "whole"
        brackets = list(rate_tables[category_key].keys())
        closest_age = min(brackets, key=lambda a: abs(a - age))
        rates = rate_tables[category_key][closest_age]

        estimates = {}
        for health_class, per_thousand_rate in rates.items():
            monthly = (coverage_amount / 1000) * per_thousand_rate
            estimates[health_class] = {
                "monthly": round(monthly, 2),
                "annual": round(monthly * 12, 2),
            }

        result = {
            "policy_type": category_key,
            "age_used_for_rate": closest_age,
            "coverage_amount": coverage_amount,
            "estimates_by_health_class": estimates,
            "notes": "Rates are illustrative estimates and may vary with underwriting.",
        }
        return json.dumps(result, indent=2)

# Simple Test to Show Agent in Action

async def demo_simple_agent():
    """Demonstrate the agent with a realistic conversation"""
    
    # Initialize agent
    agent = SimpleInsuranceAgent()
    await agent.initialize()
    
    print("Insurance Agent Demo")
    print("=" * 50)
    
    # Simulate realistic conversation
    conversation = [
        "Hi, I'm looking for life insurance advice",
        "I'm 35 years old, married with 2 kids, make $80,000 per year, no debts, $300,000 mortgage, goals: income replacement and kids' education",
        "What coverage amount would you recommend?",
        "What options would fit me?",
        "If I go with term, what would $800,000 cost me?"
    ]
    
    for i, message in enumerate(conversation, 1):
        print(f"\nCustomer: {message}")
        response = await agent.chat(message)
        print(f"Agent: {response}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(demo_simple_agent())
```

---

## Key Agent Framework Concepts

### 1. **System Instructions = Agent Personality**
The `instructions` parameter defines how your agent behaves:

```python
# Professional & Technical
instructions = "You are a technical insurance expert. Use precise calculations and industry terms."

# Friendly & Approachable  
instructions = "You are a friendly advisor. Use simple language and be patient with questions."

# Conservative & Cautious
instructions = "You prioritize protection. Always recommend comprehensive coverage."
```

### 2. **Automatic Tool Invocation**
With `FunctionCallBehavior.EnableFunctions(auto_invoke=True)`, the agent automatically calls appropriate tools when needed:

```python
# User says: "I'm 30 and need life insurance"
# Agent automatically uses a policy search tool based on the conversation context

# User says: "What would $500K cost me?"
# Agent automatically uses a premium estimation tool with the right parameters
```

### 3. **Built-in Memory with ChatHistory**
The agent remembers the entire conversation automatically:

```python
chat_history = ChatHistory()

# Turn 1
chat_history.add_user_message("I'm 30 and make $60K")
response = await agent.invoke(chat_history)

# Turn 2 - Agent remembers you're 30 and make $60K
chat_history.add_user_message("What coverage do I need?") 
response = await agent.invoke(chat_history)
```

---

## Different Agent Personalities

Here's how system instructions create different agent behaviors:

```python
# Conservative Agent
conservative_agent = ChatCompletionAgent(
    service_id="azure_openai",
    kernel=kernel,
    name="ConservativeAdvisor", 
    instructions="""You are a conservative insurance advisor focused on protection.
    
    APPROACH: Always recommend higher coverage amounts. Emphasize risks of being underinsured.
    TONE: Serious and professional. Use phrases like "protect your family" and "financial security"."""
)

# Budget-Friendly Agent
budget_agent = ChatCompletionAgent(
    service_id="azure_openai", 
    kernel=kernel,
    name="BudgetAdvisor",
    instructions="""You are practical and budget-conscious.
    
    APPROACH: Find affordable solutions. Suggest term over whole life. Provide cost-effective options.
    TONE: Understanding of budget constraints. Use phrases like "good value" and "smart choice"."""
)

# Technical Expert Agent
technical_agent = ChatCompletionAgent(
    service_id="azure_openai",
    kernel=kernel, 
    name="TechnicalExpert",
    instructions="""You are a technical insurance expert who loves data.
    
    APPROACH: Use precise calculations. Reference industry standards and actuarial data.
    TONE: Professional and analytical. Support recommendations with numbers and metrics."""
)
```

---

## Hands-On Exercise

Try modifying the agent's personality by changing the `instructions`:

```python
# Exercise: Create a "Beginner-Friendly" agent
beginner_friendly_instructions = """You are an insurance advisor who specializes in explaining things to people new to insurance.

PERSONALITY:
- Patient and encouraging  
- Never use jargon without explaining it
- Break complex topics into simple steps
- Celebrate when customers understand concepts

APPROACH:
- Ask one question at a time to avoid overwhelming
- Always explain why you're asking for information
- Use analogies to explain insurance concepts
- Confirm understanding before moving to next topic

Remember: This might be their first time buying insurance - make it a positive experience!"""
```

---

## Why This Approach Works

### ? **Conversational Flow**
- Agent maintains context throughout conversation
- No need to repeat information 
- Natural back-and-forth dialogue

### ? **Intelligent Tool Use** 
- Agent decides when to use functions based on conversation
- Automatic parameter extraction from user messages
- No manual function orchestration needed

### ? **Personality & Consistency**
- System instructions ensure consistent behavior
- Agent stays in character throughout conversation
- Customizable for different use cases

### ? **Built-in Memory**
- ChatHistory automatically manages conversation context
- Agent remembers all previous interactions
- No custom memory management required

---

## Success Criteria

After completing this module, you should understand:

? **How to create a ChatCompletionAgent** with personality and tools  
? **How agents automatically invoke functions** based on conversation context  
? **How ChatHistory provides conversation memory** without custom code  
? **How system instructions define agent behavior** and personality  
? **The difference between functions and agents** for conversational AI  

---

## Next Steps

**Ready for hands-on practice!** 

**Next**: [Lab 2: PolicyAdvisorAgent](../labs/lab2-policy-advisor.md) - Build this agent yourself!

**Key Takeaways**:
- Agents provide conversational, stateful AI interactions
- System instructions are powerful for defining agent personality  
- Automatic function calling makes agents intelligent and capable
- ChatHistory handles conversation memory automatically
- Focus on the conversation experience, not complex logic

**Coming Up**: [Module 4: Multi-Agent Systems](04-multi-agent.md) - Multiple agents working together!
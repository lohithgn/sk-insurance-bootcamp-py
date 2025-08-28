# Lab 2: Build a PolicyAdvisorAgent with Semantic Kernel

**Duration**: 45 minutes  
**Objective**: Build a conversational insurance agent using SK's ChatCompletionAgent framework that follows a clear 5-step process, uses structured JSON tools, and maintains conversation memory.

## Learning Outcomes
By completing this lab, you will be able to:
- Implement a 5-step, outcome-focused PolicyAdvisorAgent (from Module 3)
- Integrate a PolicyAdvisorPlugin that returns structured JSON
- Configure automatic tool invocation with FunctionChoiceBehavior.Auto
- Maintain multi-turn memory with ChatHistory
- Test the agent with realistic conversations

## Prerequisites
- Completed [Module 3: Single Agent Architecture](../modules/03-single-agent.md)
- Working Semantic Kernel environment
- Azure OpenAI service configured

---

## What We're Building

A conversational insurance agent that:
- Follows a consistent 5-step process with brief step announcements
- Remembers customer information throughout the conversation
- Uses tools to search policies, determine coverage, and estimate premiums
- Provides clear recommendations with minimal questions and sensible defaults

---

## Step 1: Create the PolicyAdvisorPlugin (structured JSON)

```python
# File: policy_advisor_plugin.py

import json
from semantic_kernel.functions import kernel_function

class PolicyAdvisorPlugin:
    """Plugin with realistic insurance data and calculations to support the 5-step process."""

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
```

---

## Step 2: Build the PolicyAdvisorAgent (5-step process)

```python
# File: policy_advisor_agent.py

import asyncio
import os
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

from policy_advisor_plugin import PolicyAdvisorPlugin

# Load environment variables
load_dotenv()

class PolicyAdvisorAgent:
    """Insurance advisor agent built with SK ChatCompletionAgent framework."""

    def __init__(self):
        self.kernel = None
        self.agent = None
        self.chat_history = ChatHistory()

    async def initialize(self):
        """Initialize the agent with kernel, plugins, and personality."""
        print("Initializing PolicyAdvisorAgent...")

        # 1. Create kernel and add AI service
        self.kernel = Kernel()
        service = AzureChatCompletion(
            service_id="azure_openai",
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        self.kernel.add_service(service)

        # 2. Add policy plugin (tools for the agent)
        policy_plugin = PolicyAdvisorPlugin()
        self.kernel.add_plugin(policy_plugin, plugin_name="PolicyTools")

        # 3. Create ChatCompletionAgent with a clear, outcome-focused 5-step process
        self.agent = ChatCompletionAgent(
            kernel=self.kernel,
            name="PolicyAdvisorAgent",
            instructions=self._get_agent_instructions(),
            arguments=KernelArguments(settings=OpenAIChatPromptExecutionSettings(
                max_tokens=1000,
                temperature=0.7,
                function_choice_behavior=FunctionChoiceBehavior.Auto()
            ))
        )

        print("PolicyAdvisorAgent ready with 5-step process!")

    def _get_agent_instructions(self) -> str:
        """Define the agent's personality and behavior (5-step process)."""
        return """You are a professional insurance advisor. For every policy recommendation, follow this systematic process:

STEP 1 ? UNDERSTAND THE CUSTOMER
Announce: "Step 1: Understanding your situation..."
Goal: Gather a complete picture of their profile and needs (age, income, family, goals)

STEP 2 ? IDENTIFY OPTIONS
Announce: "Step 2: Identifying suitable options..."
Goal: Find policies that match their profile and constraints

STEP 3 ? DETERMINE COVERAGE
Announce: "Step 3: Calculating appropriate coverage..."
Goal: Determine how much insurance they need based on standard methods

STEP 4 ? ESTIMATE INVESTMENT
Announce: "Step 4: Estimating your investment..."
Goal: Estimate premiums for the recommended coverage

STEP 5 ? RECOMMEND SOLUTION
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
- Keep steps concise and do not re-announce Step 1 after it has been completed."""

    async def chat(self, user_message: str) -> str:
        """Have a conversation turn with the user."""
        # Add user message to conversation
        self.chat_history.add_user_message(user_message)

        # Get agent response (automatically uses tools when needed)
        response = await self.agent.get_response(self.chat_history)

        # Return the agent's response
        return response.content if response else "I didn't understand that. Could you try again?"
```

---

## Step 3: Test the Agent

```python
# File: test_agent.py

import asyncio
from policy_advisor_agent import PolicyAdvisorAgent

async def test_basic_conversation():
    """Test basic agent conversation flow aligned with the 5-step process."""

    print("Test 1: Basic Conversation Flow")
    print("=" * 50)

    # Initialize agent
    agent = PolicyAdvisorAgent()
    await agent.initialize()

    # Test conversation flow
    test_conversation = [
        "Hi, I'm looking for life insurance advice",
        "I'm 35 years old, married with 2 kids, income is $80,000, mortgage $300,000",
        "What coverage amount would you recommend?",
        "What options would fit me?",
        "If I go with term, what would $800,000 cost me?"
    ]

    for i, message in enumerate(test_conversation, 1):
        print(f"\nCustomer (Turn {i}): {message}")
        response = await agent.chat(message)
        print(f"Agent: {response[:400]}...")
        print("-" * 60)

async def test_agent_memory():
    """Test that agent remembers context across conversation and honors step shortcuts."""

    print("\nTest 2: Agent Memory & Context")
    print("=" * 50)

    agent = PolicyAdvisorAgent()
    await agent.initialize()

    # Build up context over multiple turns
    memory_test = [
        "Hi, I need insurance help",
        "I'm 28 years old",                      # Agent should remember age
        "I make $60,000 per year",               # Agent should remember income
        "I'm single with no kids",               # Agent should remember family status
        "What would $500,000 term cost monthly?" # Should jump to Step 4 without redoing Step 1
    ]

    for i, message in enumerate(memory_test, 1):
        print(f"\nTurn {i}: {message}")
        response = await agent.chat(message)
        print(f"Agent: {response[:300]}...")
        print("-" * 40)

async def run_all():
    await test_basic_conversation()
    await test_agent_memory()

if __name__ == "__main__":
    asyncio.run(run_all())
```

---

## Success Criteria & Validation

Your agent should:
- Initialize without errors
- Announce steps concisely and follow the 5-step process (Step 1 only once)
- Use tools automatically (responses reference structured JSON outputs)
- Remember prior context across turns
- Ask at most one targeted, non-redundant question when needed
- Proceed with documented assumptions when information is missing
- Provide clear, concise recommendations with brief reasoning and an Optional section when needed

---

## Run Instructions

- Ensure your .env contains Azure OpenAI settings:
  - AZURE_OPENAI_ENDPOINT
  - AZURE_OPENAI_API_KEY
  - AZURE_OPENAI_DEPLOYMENT_NAME
  - AZURE_OPENAI_API_VERSION
- Create two files beside the lab:
  - policy_advisor_plugin.py
  - policy_advisor_agent.py
- Run the tests:
  - python test_agent.py

If you prefer, adapt the example prompts from Module 3’s demo for additional testing.

---

## Congratulations!

You built a PolicyAdvisorAgent using the SK Agent framework with a robust 5-step process, structured tools, and conversation memory. Next, try customizing the agent personality (budget-friendly, conservative, technical) by swapping instructions, or extend the plugin with additional product categories.
# Module 4: Multi‑Agent Team Orchestrating the Same 5‑Step Process

**Duration**: 60–75 minutes  
**Objective**: Implement a coordinated team of specialized ChatCompletionAgents that executes the exact same 5‑step policy recommendation process introduced in Module 3, using the same tools, data shapes, and SK APIs.

## Learning Outcomes
By completing this module, you will be able to:
- Design a multi‑agent team that mirrors a single‑agent 5‑step process
- Orchestrate multiple ChatCompletionAgents with a shared Kernel and memory
- Reuse the same PolicyAdvisorPlugin tools with automatic function calling
- Run agents in parallel where appropriate to reduce latency
- Synthesize structured outputs from multiple agents into one recommendation

## Prerequisites
- Completed [Module 3: Single Agent Architecture](03-single-agent.md)
- Successfully completed [Lab 2: PolicyAdvisorAgent](../labs/lab2-policy-advisor.md)
- Understanding of SK fundamentals and ChatCompletionAgent

---

## Single Agent vs Team of Agents: Same 5 Steps, Different Execution

In Module 3, one agent performed this outcome‑focused flow:
- Step 1: Understand the customer
- Step 2: Identify options
- Step 3: Determine coverage
- Step 4: Estimate investment (premiums)
- Step 5: Recommend solution

In this module, a coordinated team performs the same steps. Each agent owns one step and uses the same tools from Module 3. The orchestrator coordinates execution and combines results.

---

## Architecture Overview

- Shared Kernel and AI service (same env vars and setup as Module 3)
- Shared plugin: PolicyAdvisorPlugin (same functions and JSON shapes)
- Shared conversation state: CustomerProfile JSON and step outputs
- Agents:
  - IntakeAgent → Step 1 (profile building, minimal questioning)
  - OptionsAgent → Step 2 (policy search)
  - CoverageAgent → Step 3 (coverage calculation)
  - PricingAgent → Step 4 (premium estimates)
  - AdvisorAgent → Step 5 (final synthesis)
- Orchestrator:
  - Runs Step 1 once, then executes Steps 2–4 in parallel
  - Passes outputs to Step 5 for a concise recommendation

---

## Building the Team with Semantic Kernel

```python
import asyncio
import os
import json
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import kernel_function, KernelArguments
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

load_dotenv()

# Reuse the SAME plugin and JSON shapes from Module 3
class PolicyAdvisorPlugin:
    """Policy search, coverage calculation, and premium estimation"""

    @kernel_function(
        name="search_available_policies",
        description="Search policies available for a customer's age and category (life, auto, home). Returns structured data."
    )
    def search_available_policies(self, age: int, category: str = "life") -> str:
        policy_catalog = {
            "life": [
                {"id": "TL20-2025", "name": "SecureLife Term 20", "type": "Term Life (20-year level)", "min_age": 18, "max_age": 65, "min_coverage": 100_000, "max_coverage": 5_000_000, "features": ["Level premiums", "Convertible options", "Accelerated benefits rider"], "best_for": "Young families needing affordable protection"},
                {"id": "TL30-2025", "name": "SecureLife Term 30", "type": "Term Life (30-year level)", "min_age": 18, "max_age": 55, "min_coverage": 250_000, "max_coverage": 3_000_000, "features": ["Level premiums", "Renewable", "Living benefits rider"], "best_for": "Longer-term obligations like a mortgage"},
                {"id": "WL-2025", "name": "WholeLife Plus", "type": "Whole Life", "min_age": 18, "max_age": 75, "min_coverage": 50_000, "max_coverage": 2_000_000, "features": ["Cash value accumulation", "Dividend eligible"], "best_for": "Lifetime coverage and savings component"},
                {"id": "UL-2025", "name": "FlexLife Universal", "type": "Universal Life", "min_age": 20, "max_age": 70, "min_coverage": 100_000, "max_coverage": 1_500_000, "features": ["Flexible premiums", "Adjustable death benefit"], "best_for": "Flexible coverage with potential growth"},
            ],
        }
        data = policy_catalog.get(category.lower(), [])
        eligible = [p for p in data if p.get("min_age", 0) <= age <= p.get("max_age", 200)]
        return json.dumps({"category": category.lower(), "criteria": {"age": age}, "found": len(eligible), "policies": eligible}, indent=2)

    @kernel_function(
        name="calculate_coverage_needs",
        description="Calculate recommended life insurance coverage using standard methods. Returns structured data."
    )
    def calculate_coverage_needs(self, annual_income: int, dependents: int = 0, debts: int = 0, mortgage: int = 0) -> str:
        years = 10 if dependents > 0 else 5
        income_replacement = annual_income * years
        education = max(dependents, 0) * 100_000
        dime_total = debts + (annual_income * 5) + mortgage + education
        human_life_value = int(annual_income * 20 * 0.75)
        methods = {
            "income_replacement": {"years": years, "amount": income_replacement},
            "dime": {"debts": debts, "income_multiple": annual_income * 5, "mortgage": mortgage, "education": education, "total": dime_total},
            "human_life_value": {"present_value_proxy": human_life_value},
        }
        amounts = [income_replacement, dime_total, human_life_value]
        recommended = round((sum(amounts) / len(amounts)) / 50_000) * 50_000
        return json.dumps({
            "inputs": {"annual_income": annual_income, "dependents": dependents, "debts": debts, "mortgage": mortgage},
            "methods": methods,
            "recommended_coverage": int(recommended),
            "notes": "Average of standard methods rounded to nearest $50k",
        }, indent=2)

    @kernel_function(
        name="estimate_premiums",
        description="Estimate premiums for a given coverage and age using rate tables. Returns structured data."
    )
    def estimate_premiums(self, age: int, coverage_amount: int, policy_type: str = "term") -> str:
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
            estimates[health_class] = {"monthly": round(monthly, 2), "annual": round(monthly * 12, 2)}
        return json.dumps({
            "policy_type": category_key,
            "age_used_for_rate": closest_age,
            "coverage_amount": coverage_amount,
            "estimates_by_health_class": estimates,
            "notes": "Rates are illustrative estimates and may vary with underwriting.",
        }, indent=2)


class MultiAgentPolicyTeam:
    """Orchestrates a 5‑agent team to execute the same 5 steps from Module 3"""

    def __init__(self):
        self.kernel = None
        self.chat_history = ChatHistory()
        self.team_state = {
            "profile": None,   # dict
            "options": None,   # dict
            "coverage": None,  # dict
            "pricing": None,   # dict
        }
        self.intake = None
        self.options = None
        self.coverage = None
        self.pricing = None
        self.advisor = None

    async def initialize(self):
        # Shared kernel and service (same as Module 3)
        self.kernel = Kernel()
        service = AzureChatCompletion(
            service_id="azure_openai",
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        self.kernel.add_service(service)

        # Reuse the same plugin/tools from Module 3
        self.kernel.add_plugin(PolicyAdvisorPlugin(), plugin_name="PolicyTools")

        settings = OpenAIChatPromptExecutionSettings(
            max_tokens=1000,
            temperature=0.7,
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
        )

        # STEP 1: Intake Agent
        self.intake = ChatCompletionAgent(
            kernel=self.kernel,
            name="IntakeAgent",
            instructions="""
You are the Intake Agent (Step 1).
Announce: "Step 1: Understanding your situation..."
Goal: Build/Update CustomerProfile JSON: {age, income, dependents, debts, mortgage, goals, health_class, preferred_term}.
Rules:
- Ask at most one non-redundant question only if it materially changes the outcome.
- If info is missing, proceed with defaults and include an "Assumptions" note.
Respond with a brief sentence + a JSON object under the label CustomerProfile.
""",
            arguments=KernelArguments(settings=settings),
        )

        # STEP 2: Options Agent
        self.options = ChatCompletionAgent(
            kernel=self.kernel,
            name="OptionsAgent",
            instructions="""
You are the Options Agent (Step 2).
Announce: "Step 2: Identifying suitable options..."
Use PolicyTools.search_available_policies(age, category="life").
Return a concise JSON with a 'policies' array of eligible options.
""",
            arguments=KernelArguments(settings=settings),
        )

        # STEP 3: Coverage Agent
        self.coverage = ChatCompletionAgent(
            kernel=self.kernel,
            name="CoverageAgent",
            instructions="""
You are the Coverage Agent (Step 3).
Announce: "Step 3: Calculating appropriate coverage..."
Use PolicyTools.calculate_coverage_needs(annual_income, dependents, debts, mortgage).
Return the tool JSON with methods and recommended_coverage.
""",
            arguments=KernelArguments(settings=settings),
        )

        # STEP 4: Pricing Agent
        self.pricing = ChatCompletionAgent(
            kernel=self.kernel,
            name="PricingAgent",
            instructions="""
You are the Pricing Agent (Step 4).
Announce: "Step 4: Estimating your investment..."
Use PolicyTools.estimate_premiums(age, coverage_amount, policy_type="term").
Return the tool JSON with estimates_by_health_class.
""",
            arguments=KernelArguments(settings=settings),
        )

        # STEP 5: Advisor Agent
        self.advisor = ChatCompletionAgent(
            kernel=self.kernel,
            name="AdvisorAgent",
            instructions="""
You are the Advisor Agent (Step 5).
Announce: "Step 5: My recommendation..."
Synthesize Step 1–4 outputs. Provide:
- Clear recommendation (policy type, coverage amount, key option)
- Premium range using pricing JSON
- 2–3 bullet reasons
If defaults were used, add an "Assumptions" note at the end. Keep it concise and user-friendly.
""",
            arguments=KernelArguments(settings=settings),
        )

    async def run(self, user_message: str) -> str:
        """Run Step 1 once, Steps 2–4 in parallel, then Step 5 synthesis"""
        # Capture user message for context continuity
        self.chat_history.add_user_message(user_message)

        # STEP 1: Intake (build/update profile)
        intake_prompt = (
            "You are Step 1. Build or update the CustomerProfile based on this message. "
            "If missing values, use defaults and state assumptions."
        )
        self.chat_history.add_assistant_message("[Team] Dispatching Step 1 (Intake)")
        intake_resp = await self.intake.get_response(self.chat_history)
        intake_text = intake_resp.content if intake_resp else ""
        # Try to extract JSON profile from the response
        profile_json = None
        try:
            # naive extraction: find the last JSON object in text
            import re
            matches = re.findall(r"\{[\s\S]*\}", intake_text)
            if matches:
                profile_json = json.loads(matches[-1])
        except Exception:
            profile_json = None
        self.team_state["profile"] = profile_json or self.team_state.get("profile") or {
            "age": 35, "income": 65000, "dependents": 0, "debts": 0, "mortgage": 0, "goals": "income replacement", "health_class": "standard", "preferred_term": 20
        }

        p = self.team_state["profile"]

        # Prepare prompts for Steps 2–4
        options_prompt = (
            f"You are Step 2. Use tools to find life policies for age={p.get('age', 35)}. "
            "Return compact JSON with 'policies'."
        )
        coverage_prompt = (
            "You are Step 3. Use tools to calculate coverage using inputs from CustomerProfile. "
            f"annual_income={p.get('income', 65000)}, dependents={p.get('dependents', 0)}, debts={p.get('debts', 0)}, mortgage={p.get('mortgage', 0)}."
        )
        pricing_prompt = (
            "You are Step 4. Use tools to estimate premiums for policy_type='term' using CustomerProfile age and the recommended coverage from Step 3 if available; otherwise use 500000."
        )

        # Temporary per-agent histories to focus the tool calls
        options_history = ChatHistory()
        options_history.add_user_message(options_prompt)

        coverage_history = ChatHistory()
        coverage_history.add_user_message(coverage_prompt)

        pricing_history = ChatHistory()
        # coverage amount will be set after Step 3 if possible

        # Run Steps 2–3 first to get coverage, then 4
        options_task = self.options.get_response(options_history)
        coverage_task = self.coverage.get_response(coverage_history)
        options_resp, coverage_resp = await asyncio.gather(options_task, coverage_task)

        options_text = options_resp.content if options_resp else "{}"
        coverage_text = coverage_resp.content if coverage_resp else "{}"

        # Parse JSON for options and coverage
        try:
            self.team_state["options"] = json.loads(options_text)
        except Exception:
            self.team_state["options"] = {"policies": []}
        try:
            self.team_state["coverage"] = json.loads(coverage_text)
        except Exception:
            self.team_state["coverage"] = {"recommended_coverage": 500000}

        # Determine coverage amount for pricing
        cov_amt = (
            self.team_state["coverage"].get("recommended_coverage")
            or (self.team_state["coverage"].get("inputs", {}) or {}).get("recommended_coverage")
            or 500000
        )
        pricing_history.add_user_message(
            f"Customer age={p.get('age', 35)}, coverage_amount={cov_amt}, policy_type='term'."
        )

        pricing_resp = await self.pricing.get_response(pricing_history)
        pricing_text = pricing_resp.content if pricing_resp else "{}"
        try:
            self.team_state["pricing"] = json.loads(pricing_text)
        except Exception:
            self.team_state["pricing"] = {"estimates_by_health_class": {}}

        # STEP 5: Advisor synthesis
        advisor_history = ChatHistory()
        advisor_history.add_user_message(
            "You are Step 5. Synthesize the following JSON objects: "
            f"\nCustomerProfile: {json.dumps(self.team_state['profile'])}"
            f"\nOptions: {json.dumps(self.team_state['options'])}"
            f"\nCoverage: {json.dumps(self.team_state['coverage'])}"
            f"\nPricing: {json.dumps(self.team_state['pricing'])}"
        )
        advisor_resp = await self.advisor.get_response(advisor_history)
        return advisor_resp.content if advisor_resp else "I couldn't create a recommendation."


# Demo
async def demo_multi_agent_team():
    team = MultiAgentPolicyTeam()
    await team.initialize()

    print("Multi‑Agent Team Demo (5‑Step Process)")
    print("=" * 50)

    conversation = [
        "Hi, I'm 35, married with 2 kids, make $80,000 per year, no debts, $300,000 mortgage.",
        "What coverage amount would you recommend, and what might it cost for term?",
    ]

    for msg in conversation:
        print(f"\nCustomer: {msg}")
        reply = await team.run(msg)
        print(f"Team: {reply}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(demo_multi_agent_team())
```

---

## Orchestration Pattern

- Step 1 runs once to build/update the profile with minimal questioning
- Steps 2–3 execute in parallel, then Step 4 uses the coverage result
- Step 5 synthesizes all JSON outputs into a concise recommendation
- If the user asks for a later step (e.g., pricing) mid‑conversation, respond immediately using the current profile + defaults and add an "Assumptions" note

---

## Agent Personalities (Focused Instructions)

- IntakeAgent: empathetic, minimal questions, produces CustomerProfile JSON
- OptionsAgent: product‑focused, returns concise eligible options
- CoverageAgent: numeric and standards‑based, returns methods + recommendation
- PricingAgent: transparent costs, returns estimates JSON
- AdvisorAgent: crisp synthesis, action‑oriented text with optional assumptions

---

## Why This Mirrors Module 3

- Same tools (PolicyAdvisorPlugin) and JSON shapes
- Same announcement pattern (Step 1 … Step 5)
- Same SK APIs with FunctionChoiceBehavior.Auto for automatic tool use
- Shared conversation state across steps to avoid repeated questions
- Parallel execution shortens time to value without changing the process

---

## Hands‑On Exercise

- Add a "BudgetCoachAgent" that reads Pricing JSON and flags affordability issues; update the Advisor’s recommendation accordingly.
- Split CoverageAgent into NeedsAgent + CoverageAgent and reconcile outputs in AdvisorAgent.

---

## Next Steps

- Do [Lab 3: Multi‑Agent System](../labs/lab3-multi-agent.md) to build this team end‑to‑end.
- Extend the orchestrator with tracing, retries, or voting if multiple product strategies are evaluated.
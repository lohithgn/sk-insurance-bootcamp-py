# Lab 3: Multi‑Agent Team (5‑Step Process)

**Duration**: 90 minutes  
**Objective**: Build a coordinated team of specialized ChatCompletionAgents that executes the exact same 5‑step policy recommendation process from Module 3, using the same tools, JSON shapes, and SK APIs.

## Learning Outcomes
By the end of this lab, you will:
- Implement a 5‑agent team that mirrors the Module 3 single‑agent flow
- Orchestrate agents with a shared Kernel and team state
- Reuse the PolicyAdvisorPlugin and automatic function calling
- Execute steps in parallel where possible to reduce latency
- Synthesize structured outputs into a concise final recommendation

## Prerequisites
- Completed [Lab 2: PolicyAdvisorAgent](lab2-policy-advisor.md)
- Reviewed [Module 3](../modules/03-single-agent.md) and [Module 4](../modules/04-multi-agent.md)
- Azure OpenAI environment variables configured

---

## What You’ll Build

A team of five agents mapped 1:1 to the 5 steps from Module 3:
- Step 1: IntakeAgent → builds/updates CustomerProfile JSON
- Step 2: OptionsAgent → searches eligible policies
- Step 3: CoverageAgent → calculates recommended coverage
- Step 4: PricingAgent → estimates premiums
- Step 5: AdvisorAgent → synthesizes a concise recommendation

The orchestrator (MultiAgentPolicyTeam) runs Step 1 once, executes Steps 2–3 in parallel, then runs Step 4 using the coverage result, and finally Step 5 to synthesize outputs.

---

## Step 1: Setup and Reuse the Same Tools

Create a Python module (or reuse your lab workspace) and add the same plugin from Module 3. It provides three tools and returns structured JSON.

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

# Same plugin and JSON shapes as Module 3
class PolicyAdvisorPlugin:
    @kernel_function(name="search_available_policies", description="Search policies for a customer's age; returns JSON")
    def search_available_policies(self, age: int, category: str = "life") -> str:
        policy_catalog = {
            "life": [
                {"id": "TL20-2025", "name": "SecureLife Term 20", "type": "Term Life (20-year level)", "min_age": 18, "max_age": 65, "min_coverage": 100_000, "max_coverage": 5_000_000,
                 "features": ["Level premiums", "Convertible options", "Accelerated benefits rider"], "best_for": "Young families needing affordable protection"},
                {"id": "TL30-2025", "name": "SecureLife Term 30", "type": "Term Life (30-year level)", "min_age": 18, "max_age": 55, "min_coverage": 250_000, "max_coverage": 3_000_000,
                 "features": ["Level premiums", "Renewable", "Living benefits rider"], "best_for": "Longer-term obligations like a mortgage"},
                {"id": "WL-2025", "name": "WholeLife Plus", "type": "Whole Life", "min_age": 18, "max_age": 75, "min_coverage": 50_000, "max_coverage": 2_000_000,
                 "features": ["Cash value accumulation", "Dividend eligible"], "best_for": "Lifetime coverage and savings component"},
                {"id": "UL-2025", "name": "FlexLife Universal", "type": "Universal Life", "min_age": 20, "max_age": 70, "min_coverage": 100_000, "max_coverage": 1_500_000,
                 "features": ["Flexible premiums", "Adjustable death benefit"], "best_for": "Flexible coverage with potential growth"},
            ],
        }
        data = policy_catalog.get(category.lower(), [])
        eligible = [p for p in data if p.get("min_age", 0) <= age <= p.get("max_age", 200)]
        return json.dumps({"category": category.lower(), "criteria": {"age": age}, "found": len(eligible), "policies": eligible}, indent=2)

    @kernel_function(name="calculate_coverage_needs", description="Calculate recommended coverage; returns JSON")
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

    @kernel_function(name="estimate_premiums", description="Estimate premiums for a coverage/age; returns JSON")
    def estimate_premiums(self, age: int, coverage_amount: int, policy_type: str = "term") -> str:
        rate_tables = {
            "term": {25: {"preferred": 0.12, "standard": 0.18, "substandard": 0.35}, 30: {"preferred": 0.15, "standard": 0.22, "substandard": 0.42}, 35: {"preferred": 0.20, "standard": 0.30, "substandard": 0.58}, 40: {"preferred": 0.32, "standard": 0.48, "substandard": 0.95}},
            "whole": {25: {"preferred": 2.15, "standard": 2.58, "substandard": 3.87}, 30: {"preferred": 2.65, "standard": 3.18, "substandard": 4.77}, 35: {"preferred": 3.35, "standard": 4.02, "substandard": 6.03}, 40: {"preferred": 4.25, "standard": 5.10, "substandard": 7.65}},
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
```

---

## Step 2: Initialize Kernel and Agents

Create a shared Kernel, add the plugin, and configure five ChatCompletionAgents with focused instructions. Use automatic function calling like Module 3.

```python
class MultiAgentPolicyTeam:
    def __init__(self):
        self.kernel = None
        self.chat_history = ChatHistory()
        self.team_state = {"profile": None, "options": None, "coverage": None, "pricing": None}
        self.intake = None
        self.options = None
        self.coverage = None
        self.pricing = None
        self.advisor = None

    async def initialize(self):
        # Shared kernel and service
        self.kernel = Kernel()
        service = AzureChatCompletion(
            service_id="azure_openai",
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        self.kernel.add_service(service)

        # Reuse Module 3 tools
        self.kernel.add_plugin(PolicyAdvisorPlugin(), plugin_name="PolicyTools")

        settings = OpenAIChatPromptExecutionSettings(
            max_tokens=1000,
            temperature=0.7,
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
        )

        self.intake = ChatCompletionAgent(
            kernel=self.kernel,
            name="IntakeAgent",
            instructions=(
                "You are the Intake Agent (Step 1).\n"
                "Announce: 'Step 1: Understanding your situation...'\n"
                "Build/Update CustomerProfile JSON: {age, income, dependents, debts, mortgage, goals, health_class, preferred_term}.\n"
                "Ask at most one non-redundant question only if it materially changes the outcome. If missing info, use defaults and add 'Assumptions'."
            ),
            arguments=KernelArguments(settings=settings),
        )

        self.options = ChatCompletionAgent(
            kernel=self.kernel,
            name="OptionsAgent",
            instructions=(
                "You are the Options Agent (Step 2).\n"
                "Announce: 'Step 2: Identifying suitable options...'\n"
                "Use PolicyTools.search_available_policies(age, category='life'). Return concise JSON with a 'policies' array."
            ),
            arguments=KernelArguments(settings=settings),
        )

        self.coverage = ChatCompletionAgent(
            kernel=self.kernel,
            name="CoverageAgent",
            instructions=(
                "You are the Coverage Agent (Step 3).\n"
                "Announce: 'Step 3: Calculating appropriate coverage...'\n"
                "Use PolicyTools.calculate_coverage_needs(annual_income, dependents, debts, mortgage). Return JSON with methods and recommended_coverage."
            ),
            arguments=KernelArguments(settings=settings),
        )

        self.pricing = ChatCompletionAgent(
            kernel=self.kernel,
            name="PricingAgent",
            instructions=(
                "You are the Pricing Agent (Step 4).\n"
                "Announce: 'Step 4: Estimating your investment...'\n"
                "Use PolicyTools.estimate_premiums(age, coverage_amount, policy_type='term'). Return JSON with estimates_by_health_class."
            ),
            arguments=KernelArguments(settings=settings),
        )

        self.advisor = ChatCompletionAgent(
            kernel=self.kernel,
            name="AdvisorAgent",
            instructions=(
                "You are the Advisor Agent (Step 5).\n"
                "Announce: 'Step 5: My recommendation...'\n"
                "Synthesize Step 1–4 JSON into a concise recommendation with: policy type, coverage amount, premium range, 2–3 reasons."
                " If defaults were used, add an 'Assumptions' note. Keep it short."
            ),
            arguments=KernelArguments(settings=settings),
        )
```

---

## Step 3: Orchestrate the Steps (Parallel Where Possible)

Run Step 1 once to build the profile, then run Steps 2 and 3 in parallel, followed by Step 4 using the coverage result, and finally Step 5 for synthesis.

```python
    async def run(self, user_message: str) -> str:
        self.chat_history.add_user_message(user_message)

        # Step 1: Intake
        intake_history = ChatHistory()
        intake_history.add_user_message(
            "You are Step 1. Build or update the CustomerProfile based on this message. If missing values, use defaults and state assumptions."
        )
        intake_resp = await self.intake.get_response(intake_history)
        intake_text = intake_resp.content if intake_resp else ""

        # Extract or default the profile
        profile = None
        try:
            import re
            matches = re.findall(r"\{[\s\S]*\}", intake_text)
            if matches:
                profile = json.loads(matches[-1])
        except Exception:
            profile = None
        self.team_state["profile"] = profile or {
            "age": 35, "income": 65000, "dependents": 0, "debts": 0, "mortgage": 0, "goals": "income replacement", "health_class": "standard", "preferred_term": 20
        }

        p = self.team_state["profile"]

        # Steps 2 and 3 in parallel
        options_history = ChatHistory()
        options_history.add_user_message(
            f"You are Step 2. Use tools to find life policies for age={p.get('age', 35)}. Return compact JSON with 'policies'."
        )
        coverage_history = ChatHistory()
        coverage_history.add_user_message(
            "You are Step 3. Use tools to calculate coverage using inputs from CustomerProfile. "
            f"annual_income={p.get('income', 65000)}, dependents={p.get('dependents', 0)}, debts={p.get('debts', 0)}, mortgage={p.get('mortgage', 0)}."
        )

        options_task = self.options.get_response(options_history)
        coverage_task = self.coverage.get_response(coverage_history)
        options_resp, coverage_resp = await asyncio.gather(options_task, coverage_task)

        options_text = options_resp.content if options_resp else "{}"
        coverage_text = coverage_resp.content if coverage_resp else "{}"

        try:
            self.team_state["options"] = json.loads(options_text)
        except Exception:
            self.team_state["options"] = {"policies": []}
        try:
            self.team_state["coverage"] = json.loads(coverage_text)
        except Exception:
            self.team_state["coverage"] = {"recommended_coverage": 500000}

        cov_amt = self.team_state["coverage"].get("recommended_coverage") or 500000

        # Step 4: Pricing
        pricing_history = ChatHistory()
        pricing_history.add_user_message(
            f"Customer age={p.get('age', 35)}, coverage_amount={cov_amt}, policy_type='term'."
        )
        pricing_resp = await self.pricing.get_response(pricing_history)
        pricing_text = pricing_resp.content if pricing_resp else "{}"
        try:
            self.team_state["pricing"] = json.loads(pricing_text)
        except Exception:
            self.team_state["pricing"] = {"estimates_by_health_class": {}}

        # Step 5: Advisor synthesis
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
```

---

## Step 4: Demo and Validate

Create a simple demo that runs through the conversation and prints the team’s response.

```python
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

## Success Checklist

- Agents announce Step 1–5 and return concise, structured outputs
- Only IntakeAgent may ask at most one necessary question
- If data is missing, defaults are used and “Assumptions” is noted
- Steps 2–3 run in parallel; Step 4 uses the coverage output
- AdvisorAgent produces a concise recommendation with reasons and optional assumptions

---

## Extensions (Optional)

- Add a BudgetCoachAgent that reviews pricing and flags affordability issues
- Allow AdvisorAgent to produce two options (value vs premium)
- Persist CustomerProfile between runs and prefill defaults from history

---

## Troubleshooting Tips

- Ensure Azure OpenAI env vars are set: AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION
- Verify the PolicyAdvisorPlugin is added to the Kernel before creating agents
- If JSON parsing fails, print the raw content to inspect the agent output
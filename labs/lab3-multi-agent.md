# 🔹 Lab 3: Multi-Agent System

**Duration**: 90 minutes  
**Objective**: Build and orchestrate multiple specialized insurance agents working together to provide comprehensive customer recommendations.

## 🎯 Learning Outcomes
By completing this lab, you will be able to:
- Design and implement specialized agent architectures using Semantic Kernel
- Create agent orchestration systems for complex workflows
- Build collaborative decision-making systems with multiple agents
- Apply multi-agent patterns to real-world insurance scenarios
- Synthesize insights from multiple specialized agents

## 📋 Prerequisites
- Completed [Lab 2: PolicyAdvisorAgent](lab2-policy-advisor.md)
- Understanding of [Module 4: Multi-Agent Systems](../modules/04-multi-agent.md)
- Working knowledge of async programming and ChatCompletionAgent

---

## 🏗️ Multi-Agent Architecture Overview

```
Customer Input → [Orchestrator] → CustomerAnalystAgent → RiskSpecialistAgent → ProductExpertAgent → PolicyAdvisorAgent → Customer Response
                                       ↓                    ↓                   ↓                     ↓
                                  Profile Data         Risk Assessment      Product Options       Final Recommendation
```

### Key Components We'll Build:
1. **CustomerAnalystAgent**: Extracts and analyzes customer needs and situation
2. **RiskSpecialistAgent**: Evaluates customer risk factors and calculates risk scores  
3. **ProductExpertAgent**: Recommends appropriate insurance products and features
4. **PolicyAdvisorAgent**: Provides final policy recommendations with pricing
5. **InsuranceAgentOrchestrator**: Coordinates the workflow between agents

---

## 🔧 Step 1: Setup Base Framework

### Required Imports and Setup

```python
import asyncio
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

load_dotenv()

class AgentRole(Enum):
    """Specialized agent roles for policy recommendation"""
    CUSTOMER_ANALYST = "customer_analyst"
    RISK_SPECIALIST = "risk_specialist"
    PRODUCT_EXPERT = "product_expert"
    POLICY_ADVISOR = "policy_advisor"

@dataclass
class CustomerProfile:
    """Customer profile with intelligent defaults"""
    age: int = 35
    income: int = 65000
    family_status: str = "single"
    occupation: str = "professional"
    health_status: str = "good"
    insurance_needs: List[str] = field(default_factory=lambda: ["life_insurance"])
    risk_factors: List[str] = field(default_factory=list)
    
    def update_from_text(self, text: str):
        """Extract customer information from natural language"""
        text_lower = text.lower()
        
        # Age extraction
        age_match = re.search(r'(\d+) years old|i\'?m (\d+)|age (\d+)', text_lower)
        if age_match:
            self.age = int(next(g for g in age_match.groups() if g))
        
        # Income extraction  
        income_match = re.search(r'\$(\d{1,3}(?:,\d{3})*)|(\d+)k|make (\d+)|income (\d+)', text_lower)
        if income_match:
            amount = next(g for g in income_match.groups() if g)
            if amount:
                amount = amount.replace(',', '')
                self.income = int(amount) * (1000 if 'k' in text_lower and len(amount) <= 3 else 1)
        
        # Family status
        if any(word in text_lower for word in ["married", "family", "children", "kids"]):
            self.family_status = "family"
        
        # Health and risk factors
        if any(word in text_lower for word in ["smoke", "smoker"]):
            if "smoking" not in self.risk_factors:
                self.risk_factors.append("smoking")
        
        if any(word in text_lower for word in ["diabetes", "heart", "blood pressure"]):
            if "health_condition" not in self.risk_factors:
                self.risk_factors.append("health_condition")

@dataclass  
class AgentAnalysis:
    """Analysis result from a specialized agent"""
    agent_role: AgentRole
    analysis: str
    recommendations: List[str]
    confidence: float
    key_factors: List[str]

print("✅ Base framework setup complete!")
```

---

## 👥 Step 2: Implement CustomerAnalystAgent

### Customer Needs Analysis Agent

```python
class CustomerAnalystAgent:
    """👥 Agent specializing in understanding customer needs and situation"""
    
    def __init__(self):
        self.role = AgentRole.CUSTOMER_ANALYST
        self.name = "Emma - Customer Analyst"
        self.kernel = None
        self.agent = None
        
        self.instructions = """You are Emma, a Customer Analyst who specializes in understanding customer needs and situations.

EXPERTISE:
- Customer needs assessment
- Life situation analysis  
- Insurance needs identification
- Coverage gap analysis

YOUR ANALYSIS SHOULD FOCUS ON:
1. Understanding their primary insurance needs
2. Assessing their life situation and dependencies
3. Identifying coverage priorities
4. Recommending coverage amounts based on their situation

APPROACH:
- Focus on their life situation and what they need to protect
- Consider their family dependencies and financial obligations
- Identify primary vs secondary insurance needs
- Use income multipliers: 6-8x for singles, 8-12x for families

Be empathetic and focus on their protection needs."""
        
    async def initialize(self, kernel: Kernel):
        """Initialize the agent"""
        self.kernel = kernel
        
        # Add customer analysis plugin
        customer_plugin = CustomerAnalysisPlugin()
        kernel.add_plugin(customer_plugin, plugin_name="CustomerTools")
        
        self.agent = ChatCompletionAgent(
            service_id="azure_openai",
            kernel=self.kernel,
            name=self.name,
            instructions=self.instructions,
            execution_settings=OpenAIChatPromptExecutionSettings(
                service_id="azure_openai",
                max_tokens=800,
                temperature=0.7,
                function_call_behavior=FunctionCallBehavior.EnableFunctions(auto_invoke=True)
            )
        )
        
        print(f"✅ {self.name} initialized")
    
    async def analyze_customer(self, customer_profile: CustomerProfile, customer_request: str) -> AgentAnalysis:
        """Analyze customer from customer needs perspective"""
        
        # Create focused prompt for customer analysis
        analysis_prompt = f"""Analyze this customer's insurance needs and situation:

CUSTOMER REQUEST: {customer_request}

CUSTOMER PROFILE:
• Age: {customer_profile.age}
• Income: ${customer_profile.income:,}
• Family Status: {customer_profile.family_status}
• Occupation: {customer_profile.occupation}
• Health Status: {customer_profile.health_status}
• Risk Factors: {', '.join(customer_profile.risk_factors) if customer_profile.risk_factors else 'None'}

Provide your specialized analysis focusing on:
1. Their primary protection needs
2. Appropriate coverage amounts
3. Priority level of different insurance types
4. Life situation considerations

Use your customer analysis tools to provide specific recommendations."""

        # Get agent's analysis
        chat_history = ChatHistory()
        chat_history.add_user_message(analysis_prompt)
        
        response = await self.agent.invoke(chat_history)
        analysis_text = response[-1].content if response else "Unable to analyze"
        
        # Return structured analysis
        return AgentAnalysis(
            agent_role=self.role,
            analysis=analysis_text,
            recommendations=self._extract_recommendations(analysis_text),
            confidence=0.85,
            key_factors=self._extract_key_factors(analysis_text)
        )
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract key recommendations from analysis text"""
        recommendations = []
        lines = text.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['recommend', 'suggest', 'should']):
                recommendations.append(line.strip())
        return recommendations[:3]  # Top 3 recommendations
    
    def _extract_key_factors(self, text: str) -> List[str]:
        """Extract key factors mentioned in analysis"""
        factors = []
        key_terms = ['age', 'income', 'family', 'coverage', 'protection', 'needs']
        for term in key_terms:
            if term in text.lower():
                factors.append(term)
        return factors

class CustomerAnalysisPlugin:
    """Plugin for customer needs analysis"""
    
    @kernel_function(
        name="assess_coverage_needs",
        description="Assess appropriate coverage amount based on customer situation"
    )
    def assess_coverage_needs(self, income: int, family_status: str, age: int) -> str:
        """Calculate recommended coverage amount"""
        
        if "family" in family_status.lower() or "married" in family_status.lower():
            multiplier = 10
            reason = "family income replacement and protection"
        else:
            multiplier = 7
            reason = "debt coverage and final expenses"
        
        base_coverage = income * multiplier
        
        # Age adjustments
        if age < 30:
            base_coverage = int(base_coverage * 1.1)
            adjustment = "increased for future earning potential"
        elif age > 55:
            base_coverage = int(base_coverage * 0.9)
            adjustment = "adjusted for approaching retirement"
        else:
            adjustment = "standard calculation"
        
        return f"""Coverage Needs Assessment:
• Recommended Amount: ${base_coverage:,}
• Calculation: {multiplier}x annual income
• Reasoning: {reason.title()}
• Age Adjustment: {adjustment.title()}

This amount should provide adequate protection for your situation."""
    
    @kernel_function(
        name="prioritize_insurance_needs",
        description="Prioritize different types of insurance based on customer situation"
    )
    def prioritize_insurance_needs(self, age: int, family_status: str, occupation: str) -> str:
        """Prioritize insurance needs"""
        
        priorities = []
        
        if family_status != "single":
            priorities.append("1. Life Insurance - HIGH PRIORITY (family protection)")
        else:
            priorities.append("1. Life Insurance - MEDIUM PRIORITY (debt/final expenses)")
        
        if age < 65:
            priorities.append("2. Disability Insurance - HIGH PRIORITY (income protection)")
        
        priorities.append("3. Health Insurance - ESSENTIAL (medical costs)")
        
        if "business" in occupation.lower() or "self" in occupation.lower():
            priorities.append("4. Business Insurance - HIGH PRIORITY (business protection)")
        
        return f"""Insurance Priority Assessment:

{chr(10).join(priorities)}

Primary Focus: Life insurance should be your immediate priority given your situation."""

print("✅ CustomerAnalystAgent implemented!")
```

---

## 🔬 Step 3: Implement RiskSpecialistAgent

### Risk Assessment Agent

```python
class RiskSpecialistAgent:
    """🔬 Agent specializing in risk assessment and categorization"""
    
    def __init__(self):
        self.role = AgentRole.RISK_SPECIALIST
        self.name = "Dr. Martinez - Risk Specialist"
        self.kernel = None
        self.agent = None
        
        self.instructions = """You are Dr. Martinez, a Risk Assessment Specialist with expertise in evaluating insurance risks.

EXPERTISE:
- Health risk evaluation
- Occupational risk assessment
- Lifestyle risk factors
- Risk categorization and pricing impact

YOUR ANALYSIS SHOULD FOCUS ON:
1. Overall risk category (Preferred, Standard, Substandard)
2. Key risk factors that affect pricing
3. Underwriting considerations
4. Risk mitigation recommendations

APPROACH:
- Assess age, health, occupation, and lifestyle factors
- Categorize overall risk level professionally
- Explain how risks impact insurance availability and pricing
- Suggest risk reduction strategies where appropriate

Be professional and data-driven in your risk assessments."""
        
    async def initialize(self, kernel: Kernel):
        """Initialize the agent"""
        self.kernel = kernel
        
        # Add risk analysis plugin
        risk_plugin = RiskAnalysisPlugin()
        kernel.add_plugin(risk_plugin, plugin_name="RiskTools")
        
        self.agent = ChatCompletionAgent(
            service_id="azure_openai",
            kernel=self.kernel,
            name=self.name,
            instructions=self.instructions,
            execution_settings=OpenAIChatPromptExecutionSettings(
                service_id="azure_openai",
                max_tokens=800,
                temperature=0.7,
                function_call_behavior=FunctionCallBehavior.EnableFunctions(auto_invoke=True)
            )
        )
        
        print(f"✅ {self.name} initialized")
    
    async def analyze_customer(self, customer_profile: CustomerProfile, customer_request: str) -> AgentAnalysis:
        """Analyze customer from risk assessment perspective"""
        
        analysis_prompt = f"""Evaluate this customer's insurance risk profile:

CUSTOMER REQUEST: {customer_request}

CUSTOMER PROFILE:
• Age: {customer_profile.age}
• Income: ${customer_profile.income:,}
• Family Status: {customer_profile.family_status}
• Occupation: {customer_profile.occupation}
• Health Status: {customer_profile.health_status}
• Risk Factors: {', '.join(customer_profile.risk_factors) if customer_profile.risk_factors else 'None'}

Provide comprehensive risk analysis focusing on:
1. Overall risk category and score
2. Key risk factors affecting pricing
3. Underwriting considerations
4. Risk mitigation recommendations

Use your risk analysis tools to provide detailed assessment."""

        chat_history = ChatHistory()
        chat_history.add_user_message(analysis_prompt)
        
        response = await self.agent.invoke(chat_history)
        analysis_text = response[-1].content if response else "Unable to analyze"
        
        return AgentAnalysis(
            agent_role=self.role,
            analysis=analysis_text,
            recommendations=self._extract_recommendations(analysis_text),
            confidence=0.90,
            key_factors=self._extract_key_factors(analysis_text)
        )
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract risk recommendations"""
        recommendations = []
        lines = text.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['recommend', 'consider', 'should', 'suggest']):
                recommendations.append(line.strip())
        return recommendations[:3]
    
    def _extract_key_factors(self, text: str) -> List[str]:
        """Extract key risk factors"""
        factors = []
        key_terms = ['age', 'health', 'occupation', 'risk', 'smoking', 'premium']
        for term in key_terms:
            if term in text.lower():
                factors.append(term)
        return factors

class RiskAnalysisPlugin:
    """Plugin for risk assessment calculations"""
    
    @kernel_function(
        name="calculate_risk_score",
        description="Calculate comprehensive risk score for underwriting"
    )
    def calculate_risk_score(self, age: int, health_status: str, occupation: str, risk_factors: str) -> str:
        """Calculate risk assessment"""
        
        base_score = 100
        
        # Age factor
        if age <= 35:
            age_factor = 0.9
        elif age <= 50:
            age_factor = 1.0
        else:
            age_factor = 1.2
        
        # Health factor
        health_factor = 1.0
        if "smoking" in risk_factors.lower():
            health_factor = 1.4
        elif "health_condition" in risk_factors.lower():
            health_factor = 1.15
        
        # Occupation factor
        high_risk_jobs = ["pilot", "construction", "mining", "police"]
        occupation_factor = 1.3 if any(job in occupation.lower() for job in high_risk_jobs) else 1.0
        
        final_score = base_score * age_factor * health_factor * occupation_factor
        
        # Risk category
        if final_score <= 100:
            category = "Preferred Plus"
        elif final_score <= 125:
            category = "Standard Plus"
        elif final_score <= 150:
            category = "Standard"
        else:
            category = "Substandard"
        
        return f"""Risk Assessment Results:
• Overall Risk Score: {final_score:.0f}
• Risk Category: {category}
• Age Impact: {age_factor:.1f}x multiplier
• Health Impact: {health_factor:.1f}x multiplier
• Occupation Impact: {occupation_factor:.1f}x multiplier

Risk Category Impact: {category} rating will affect premium pricing."""

print("✅ RiskSpecialistAgent implemented!")
```

---

## 🎯 Step 4: Implement ProductExpertAgent

### Product Recommendation Agent

```python
class ProductExpertAgent:
    """🎯 Agent specializing in insurance products and features"""
    
    def __init__(self):
        self.role = AgentRole.PRODUCT_EXPERT
        self.name = "Alex - Product Expert"
        self.kernel = None
        self.agent = None
        
        self.instructions = """You are Alex, a Product Expert with comprehensive knowledge of all insurance products.

EXPERTISE:
- Insurance product features and benefits
- Policy type recommendations
- Coverage options and riders
- Product comparisons

YOUR ANALYSIS SHOULD FOCUS ON:
1. Best product types for their situation (Term vs Whole vs Universal)
2. Specific products that match their needs
3. Important policy features and riders to consider
4. Product comparisons and trade-offs

APPROACH:
- Match products to customer needs and budget
- Explain product benefits clearly
- Recommend specific policy features
- Consider both current needs and future flexibility

Focus on finding the right products for their specific situation."""
        
    async def initialize(self, kernel: Kernel):
        """Initialize the agent"""
        self.kernel = kernel
        
        # Add product analysis plugin
        product_plugin = ProductAnalysisPlugin()
        kernel.add_plugin(product_plugin, plugin_name="ProductTools")
        
        self.agent = ChatCompletionAgent(
            service_id="azure_openai",
            kernel=self.kernel,
            name=self.name,
            instructions=self.instructions,
            execution_settings=OpenAIChatPromptExecutionSettings(
                service_id="azure_openai",
                max_tokens=800,
                temperature=0.7,
                function_call_behavior=FunctionCallBehavior.EnableFunctions(auto_invoke=True)
            )
        )
        
        print(f"✅ {self.name} initialized")
    
    async def analyze_customer(self, customer_profile: CustomerProfile, customer_request: str) -> AgentAnalysis:
        """Analyze customer from product recommendation perspective"""
        
        analysis_prompt = f"""Recommend appropriate insurance products for this customer:

CUSTOMER REQUEST: {customer_request}

CUSTOMER PROFILE:
• Age: {customer_profile.age}
• Income: ${customer_profile.income:,}
• Family Status: {customer_profile.family_status}
• Occupation: {customer_profile.occupation}
• Health Status: {customer_profile.health_status}
• Risk Factors: {', '.join(customer_profile.risk_factors) if customer_profile.risk_factors else 'None'}

Provide product recommendations focusing on:
1. Best product types for their situation
2. Specific policy features to consider
3. Product comparisons and benefits
4. Recommendations for coverage amounts

Use your product analysis tools to provide detailed recommendations."""

        chat_history = ChatHistory()
        chat_history.add_user_message(analysis_prompt)
        
        response = await self.agent.invoke(chat_history)
        analysis_text = response[-1].content if response else "Unable to analyze"
        
        return AgentAnalysis(
            agent_role=self.role,
            analysis=analysis_text,
            recommendations=self._extract_recommendations(analysis_text),
            confidence=0.88,
            key_factors=self._extract_key_factors(analysis_text)
        )
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract product recommendations"""
        recommendations = []
        lines = text.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['recommend', 'suggest', 'best', 'consider']):
                recommendations.append(line.strip())
        return recommendations[:3]
    
    def _extract_key_factors(self, text: str) -> List[str]:
        """Extract key product factors"""
        factors = []
        key_terms = ['term', 'whole', 'universal', 'coverage', 'premium', 'features']
        for term in key_terms:
            if term in text.lower():
                factors.append(term)
        return factors

class ProductAnalysisPlugin:
    """Plugin for product recommendations"""
    
    @kernel_function(
        name="recommend_product_types",
        description="Recommend appropriate insurance product types based on needs"
    )
    def recommend_product_types(self, age: int, coverage_amount: int, budget_conscious: bool = True) -> str:
        """Recommend product types"""
        
        recommendations = []
        
        # Term Life (usually best for most people)
        if age <= 60:
            term_rec = f"""1. TERM LIFE INSURANCE (Recommended)
   • 20-Year Term for coverage up to ${min(coverage_amount, 2000000):,}
   • Pros: Affordable, high coverage amounts
   • Cons: Temporary coverage, premiums increase after term
   • Best for: Young families, mortgage protection"""
            recommendations.append(term_rec)
        
        # Whole Life
        if coverage_amount <= 1000000:
            whole_rec = f"""2. WHOLE LIFE INSURANCE
   • Permanent coverage up to ${min(coverage_amount, 1000000):,}
   • Pros: Guaranteed cash value, lifetime coverage
   • Cons: Higher premiums, lower coverage amounts
   • Best for: Estate planning, permanent needs"""
            recommendations.append(whole_rec)
        
        # Universal Life
        if age <= 55:
            universal_rec = f"""3. UNIVERSAL LIFE INSURANCE
   • Flexible coverage up to ${min(coverage_amount, 1500000):,}
   • Pros: Flexible premiums, investment component
   • Cons: Complex, variable returns
   • Best for: Sophisticated investors, flexible needs"""
            recommendations.append(universal_rec)
        
        return f"""Product Type Recommendations:

{chr(10).join(recommendations)}

PRIMARY RECOMMENDATION: Term Life Insurance is typically the best choice for most people due to affordability and high coverage amounts."""

print("✅ ProductExpertAgent implemented!")
```

---

## 💰 Step 5: Implement PolicyAdvisorAgent

### Final Policy Recommendation Agent

```python
class PolicyAdvisorAgent:
    """💰 Agent specializing in final policy recommendations and pricing"""
    
    def __init__(self):
        self.role = AgentRole.POLICY_ADVISOR
        self.name = "Jordan - Policy Advisor"
        self.kernel = None
        self.agent = None
        
        self.instructions = """You are Jordan, a Policy Advisor who provides final policy recommendations with accurate pricing.

EXPERTISE:
- Premium calculations and pricing
- Policy comparisons and value analysis
- Final recommendations synthesis
- Affordability assessments

YOUR ANALYSIS SHOULD FOCUS ON:
1. Synthesizing insights from other specialists
2. Providing specific policy recommendations with pricing
3. Ensuring affordability based on customer income
4. Explaining value propositions clearly

APPROACH:
- Consider all previous analyses from the team
- Provide realistic premium estimates
- Focus on value and affordability
- Give clear next steps for the customer

Provide comprehensive final recommendations that tie everything together."""
        
    async def initialize(self, kernel: Kernel):
        """Initialize the agent"""
        self.kernel = kernel
        
        # Add pricing analysis plugin
        pricing_plugin = PricingAnalysisPlugin()
        kernel.add_plugin(pricing_plugin, plugin_name="PricingTools")
        
        self.agent = ChatCompletionAgent(
            service_id="azure_openai",
            kernel=self.kernel,
            name=self.name,
            instructions=self.instructions,
            execution_settings=OpenAIChatPromptExecutionSettings(
                service_id="azure_openai",
                max_tokens=800,
                temperature=0.7,
                function_call_behavior=FunctionCallBehavior.EnableFunctions(auto_invoke=True)
            )
        )
        
        print(f"✅ {self.name} initialized")
    
    async def analyze_customer(self, customer_profile: CustomerProfile, customer_request: str, previous_analyses: List[AgentAnalysis] = None) -> AgentAnalysis:
        """Provide final policy recommendations synthesizing all previous analyses"""
        
        # Build context from previous analyses
        context = ""
        if previous_analyses:
            for analysis in previous_analyses:
                context += f"\n{analysis.agent_role.value.upper()} INSIGHTS:\n{analysis.analysis}\n"
        
        analysis_prompt = f"""Provide final policy recommendations for this customer, synthesizing all team insights:

CUSTOMER REQUEST: {customer_request}

CUSTOMER PROFILE:
• Age: {customer_profile.age}
• Income: ${customer_profile.income:,}
• Family Status: {customer_profile.family_status}
• Occupation: {customer_profile.occupation}
• Health Status: {customer_profile.health_status}
• Risk Factors: {', '.join(customer_profile.risk_factors) if customer_profile.risk_factors else 'None'}

TEAM SPECIALIST INSIGHTS:
{context}

Provide final recommendations including:
1. Specific policy recommendations with pricing
2. Coverage amounts and terms
3. Affordability analysis
4. Clear next steps for the customer

Use your pricing tools to provide accurate cost estimates."""

        chat_history = ChatHistory()
        chat_history.add_user_message(analysis_prompt)
        
        response = await self.agent.invoke(chat_history)
        analysis_text = response[-1].content if response else "Unable to analyze"
        
        return AgentAnalysis(
            agent_role=self.role,
            analysis=analysis_text,
            recommendations=self._extract_recommendations(analysis_text),
            confidence=0.92,
            key_factors=self._extract_key_factors(analysis_text)
        )
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract final policy recommendations"""
        recommendations = []
        lines = text.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['recommend', 'policy', 'coverage', 'premium']):
                recommendations.append(line.strip())
        return recommendations[:4]  # Top 4 recommendations
    
    def _extract_key_factors(self, text: str) -> List[str]:
        """Extract key policy factors"""
        factors = []
        key_terms = ['premium', 'coverage', 'affordability', 'value', 'policy', 'term']
        for term in key_terms:
            if term in text.lower():
                factors.append(term)
        return factors

class PricingAnalysisPlugin:
    """Plugin for premium calculations"""
    
    @kernel_function(
        name="calculate_premium_estimate",
        description="Calculate estimated premiums for different coverage amounts and product types"
    )
    def calculate_premium_estimate(self, age: int, coverage_amount: int, product_type: str, risk_multiplier: float = 1.0) -> str:
        """Calculate premium estimates"""
        
        # Base rates per $1000 of coverage (monthly)
        if "term" in product_type.lower():
            if age <= 35:
                base_rate = 0.60
            elif age <= 45:
                base_rate = 1.00
            elif age <= 55:
                base_rate = 2.00
            else:
                base_rate = 4.00
        else:  # Whole life
            if age <= 35:
                base_rate = 8.50
            elif age <= 45:
                base_rate = 12.00
            elif age <= 55:
                base_rate = 18.00
            else:
                base_rate = 28.00
        
        monthly_premium = (coverage_amount / 1000) * base_rate * risk_multiplier
        annual_premium = monthly_premium * 12
        
        return f"""Premium Estimates for ${coverage_amount:,} {product_type.title()}:

• Monthly Premium: ${monthly_premium:.2f}
• Annual Premium: ${annual_premium:.2f}
• Rate per $1,000: ${base_rate * risk_multiplier:.2f}/month
• Risk Adjustment: {risk_multiplier:.2f}x applied

Cost Analysis:
• Annual cost as % of income: {(annual_premium / 65000) * 100:.1f}% (assuming $65K income)
• Daily cost: ${annual_premium / 365:.2f} per day

*Estimates based on standard underwriting. Final rates subject to medical exam."""
    
    @kernel_function(
        name="affordability_analysis",
        description="Analyze affordability of insurance options based on income"
    )
    def affordability_analysis(self, income: int, monthly_premium: float) -> str:
        """Analyze affordability"""
        
        monthly_income = income / 12
        premium_percentage = (monthly_premium / monthly_income) * 100
        
        if premium_percentage <= 3:
            affordability = "Very Affordable"
            recommendation = "Excellent fit for your budget"
        elif premium_percentage <= 5:
            affordability = "Affordable"
            recommendation = "Good fit for your budget"
        elif premium_percentage <= 8:
            affordability = "Moderate Cost"
            recommendation = "Consider reducing coverage or term length"
        else:
            affordability = "High Cost"
            recommendation = "Recommend lower coverage amount or different product type"
        
        return f"""Affordability Analysis:

• Monthly Income: ${monthly_income:,.2f}
• Insurance Premium: ${monthly_premium:.2f}
• Premium as % of Income: {premium_percentage:.1f}%
• Affordability Rating: {affordability}

RECOMMENDATION: {recommendation}

GUIDELINE: Insurance premiums should typically be 3-5% of gross monthly income for optimal financial balance."""

print("✅ PolicyAdvisorAgent implemented!")
```

---

## 🎭 Step 6: Create the Multi-Agent Orchestrator

### Simple Orchestration System

```python
class InsuranceAgentOrchestrator:
    """Orchestrator that coordinates multiple specialized agents for comprehensive policy recommendations"""
    
    def __init__(self):
        self.kernel = None
        self.agents = {}
        
    async def initialize(self):
        """Initialize all specialized agents"""
        print("🚀 Initializing Multi-Agent Policy Recommendation System")
        print("=" * 65)
        
        # Create shared kernel
        self.kernel = Kernel()
        service = AzureChatCompletion(
            service_id="azure_openai",
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        self.kernel.add_service(service)
        
        # Initialize all specialized agents
        agents_to_create = [
            CustomerAnalystAgent(),
            RiskSpecialistAgent(), 
            ProductExpertAgent(),
            PolicyAdvisorAgent()
        ]
        
        for agent in agents_to_create:
            await agent.initialize(self.kernel)
            self.agents[agent.role] = agent
        
        print(f"\n✅ Multi-Agent System Ready with {len(self.agents)} specialists!")
    
    async def get_comprehensive_recommendation(self, customer_request: str) -> Dict[str, Any]:
        """Get comprehensive policy recommendation from all agents"""
        
        # Build customer profile from request
        customer_profile = CustomerProfile()
        customer_profile.update_from_text(customer_request)
        
        print(f"\n🎯 Analyzing Customer Request: {customer_request}")
        print("=" * 50)
        
        # Get analysis from each specialized agent
        analyses = {}
        
        print("\n📋 Getting Specialist Analyses...")
        
        # Step 1: Customer Analysis
        print(f"   🔍 {self.agents[AgentRole.CUSTOMER_ANALYST].name} analyzing...")
        customer_analysis = await self.agents[AgentRole.CUSTOMER_ANALYST].analyze_customer(customer_profile, customer_request)
        analyses[AgentRole.CUSTOMER_ANALYST] = customer_analysis
        print(f"   ✅ Customer analysis complete")
        
        # Step 2: Risk Analysis
        print(f"   🔍 {self.agents[AgentRole.RISK_SPECIALIST].name} analyzing...")
        risk_analysis = await self.agents[AgentRole.RISK_SPECIALIST].analyze_customer(customer_profile, customer_request)
        analyses[AgentRole.RISK_SPECIALIST] = risk_analysis
        print(f"   ✅ Risk analysis complete")
        
        # Step 3: Product Analysis
        print(f"   🔍 {self.agents[AgentRole.PRODUCT_EXPERT].name} analyzing...")
        product_analysis = await self.agents[AgentRole.PRODUCT_EXPERT].analyze_customer(customer_profile, customer_request)
        analyses[AgentRole.PRODUCT_EXPERT] = product_analysis
        print(f"   ✅ Product analysis complete")
        
        # Step 4: Final Policy Recommendation
        print(f"   🔍 {self.agents[AgentRole.POLICY_ADVISOR].name} providing final recommendations...")
        previous_analyses = [customer_analysis, risk_analysis, product_analysis]
        policy_analysis = await self.agents[AgentRole.POLICY_ADVISOR].analyze_customer(customer_profile, customer_request, previous_analyses)
        analyses[AgentRole.POLICY_ADVISOR] = policy_analysis
        print(f"   ✅ Final recommendations complete")
        
        # Synthesize recommendations
        final_recommendation = self._synthesize_recommendations(analyses, customer_profile)
        
        return {
            "customer_profile": customer_profile,
            "specialist_analyses": analyses,
            "final_recommendation": final_recommendation,
            "timestamp": datetime.now().isoformat()
        }
    
    def _synthesize_recommendations(self, analyses: Dict[AgentRole, AgentAnalysis], profile: CustomerProfile) -> Dict[str, Any]:
        """Synthesize all agent analyses into final recommendation"""
        
        # Extract key insights from each agent
        customer_analysis = analyses[AgentRole.CUSTOMER_ANALYST].analysis
        risk_analysis = analyses[AgentRole.RISK_SPECIALIST].analysis
        product_analysis = analyses[AgentRole.PRODUCT_EXPERT].analysis
        policy_analysis = analyses[AgentRole.POLICY_ADVISOR].analysis
        
        # Create synthesis
        synthesis = {
            "recommended_coverage_amount": self._extract_coverage_amount(customer_analysis),
            "recommended_product_type": self._extract_product_type(product_analysis),
            "risk_category": self._extract_risk_category(risk_analysis),
            "estimated_monthly_premium": self._extract_premium(policy_analysis),
            "key_benefits": self._extract_benefits(analyses),
            "next_steps": [
                "Review the recommended policy options",
                "Complete a formal application",
                "Schedule medical exam if required",
                "Finalize coverage and begin protection"
            ]
        }
        
        return synthesis
    
    def _extract_coverage_amount(self, analysis: str) -> str:
        """Extract recommended coverage amount from analysis"""
        import re
        amounts = re.findall(r'\$(\d{1,3}(?:,\d{3})*)', analysis)
        if amounts:
            return f"${amounts[0]}"
        return "$500,000"  # Default
    
    def _extract_product_type(self, analysis: str) -> str:
        """Extract recommended product type"""
        if "term" in analysis.lower():
            return "Term Life Insurance"
        elif "whole" in analysis.lower():
            return "Whole Life Insurance"
        else:
            return "Term Life Insurance"  # Default
    
    def _extract_risk_category(self, analysis: str) -> str:
        """Extract risk category"""
        categories = ["Preferred Plus", "Standard Plus", "Standard", "Substandard"]
        for category in categories:
            if category.lower() in analysis.lower():
                return category
        return "Standard"  # Default
    
    def _extract_premium(self, analysis: str) -> str:
        """Extract monthly premium estimate"""
        import re
        premiums = re.findall(r'\$(\d+\.?\d*)', analysis)
        if premiums:
            return f"${premiums[0]}"
        return "$45.00"  # Default
    
    def _extract_benefits(self, analyses: Dict[AgentRole, AgentAnalysis]) -> List[str]:
        """Extract key benefits from all analyses"""
        benefits = [
            "Comprehensive protection for your family",
            "Affordable premium based on your risk profile", 
            "Product type matched to your specific needs",
            "Coverage amount appropriate for your situation"
        ]
        return benefits

print("✅ InsuranceAgentOrchestrator implemented!")
```

---

## 🧪 Step 7: Test the Multi-Agent System

### Comprehensive Testing

```python
async def test_multi_agent_system():
    """Test the complete multi-agent insurance system"""
    
    print("🧪 Testing Multi-Agent Insurance System")
    print("=" * 65)
    
    # Initialize orchestrator
    orchestrator = InsuranceAgentOrchestrator()
    await orchestrator.initialize()
    
    # Test different customer scenarios
    test_scenarios = [
        {
            "name": "Young Family",
            "request": "Hi, I'm 32 years old, married with 2 young children. I make $75,000 per year as a teacher and want life insurance to protect my family if something happens to me."
        },
        {
            "name": "Single Professional", 
            "request": "I'm 28, single, work as a software engineer making $95,000. I want life insurance mainly to cover my student loans and provide some protection."
        },
        {
            "name": "Older Professional",
            "request": "I'm 45, married, no kids, household income around $120,000. Looking for life insurance for estate planning and to cover our mortgage."
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🎬 Scenario {i}: {scenario['name']}")
        print("=" * 40)
        
        result = await orchestrator.get_comprehensive_recommendation(scenario['request'])
        
        # Display results
        print(f"\n📊 COMPREHENSIVE RECOMMENDATION:")
        print(f"   Recommended Coverage: {result['final_recommendation']['recommended_coverage_amount']}")
        print(f"   Product Type: {result['final_recommendation']['recommended_product_type']}")
        print(f"   Risk Category: {result['final_recommendation']['risk_category']}")
        print(f"   Est. Monthly Premium: {result['final_recommendation']['estimated_monthly_premium']}")
        
        print(f"\n🎯 Key Benefits:")
        for benefit in result['final_recommendation']['key_benefits']:
            print(f"   • {benefit}")
        
        print(f"\n📋 Specialist Insights:")
        for role, analysis in result['specialist_analyses'].items():
            agent_name = role.value.replace('_', ' ').title()
            print(f"   {agent_name}: {len(analysis.recommendations)} recommendations provided")
        
        print("-" * 50)

async def run_lab3():
    """Run complete Lab 3 multi-agent system"""
    
    try:
        print("🚀 Starting Lab 3: Multi-Agent Insurance System")
        print("=" * 60)
        
        # Run the comprehensive test
        await test_multi_agent_system()
        
        print(f"\n🎉 Lab 3 completed successfully!")
        print(f"✅ Multi-agent system working correctly")
        
    except Exception as e:
        print(f"❌ Lab 3 failed: {str(e)}")
        import traceback
        traceback.print_exc()

# Run the lab
if __name__ == "__main__":
    asyncio.run(run_lab3())
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Agent Initialization Failed** | Agents not starting properly | Check Azure OpenAI credentials and deployment names |
| **Plugin Not Found** | Function calls failing | Ensure plugins are added to kernel before agent creation |
| **Analysis Incomplete** | Missing recommendations | Verify agent instructions and prompt structure |
| **Timeout Errors** | Long response times | Reduce analysis complexity or increase timeout |

### Debug Helper

```python
async def debug_multi_agent_system():
    """Debug multi-agent system issues"""
    
    print("🔧 Debugging Multi-Agent System...")
    
    try:
        # Test kernel setup
        kernel = Kernel()
        service = AzureChatCompletion(
            service_id="azure_openai",
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        kernel.add_service(service)
        print("✅ Kernel setup: OK")
        
        # Test individual agents
        customer_agent = CustomerAnalystAgent()
        await customer_agent.initialize(kernel)
        print("✅ Customer agent: OK")
        
        # Test orchestrator
        orchestrator = InsuranceAgentOrchestrator()
        await orchestrator.initialize()
        print("✅ Orchestrator: OK")
        
        # Test simple recommendation
        result = await orchestrator.get_comprehensive_recommendation("I'm 30, need insurance")
        print(f"✅ Simple test: {type(result)}")
        
    except Exception as e:
        print(f"❌ Debug failed: {str(e)}")
        import traceback
        traceback.print_exc()

# Uncomment to run debug: asyncio.run(debug_multi_agent_system())
```

---

## ✅ Success Criteria

By completing this lab, you should have:

- ✅ Successfully implemented 4 specialized insurance agents using Semantic Kernel
- ✅ Created a working multi-agent orchestrator with simple coordination
- ✅ Built collaborative decision-making with multiple perspectives
- ✅ Processed test scenarios end-to-end
- ✅ Demonstrated specialized expertise in each agent
- ✅ Achieved coordinated multi-agent workflows
- ✅ Built a scalable and maintainable multi-agent architecture

---

## 📚 Next Steps

**🎉 Congratulations!** You've successfully built a complete multi-agent insurance system using Semantic Kernel.

**Key Achievements**:
- ✅ Multi-agent architecture with specialized roles
- ✅ Simple but effective orchestration pattern
- ✅ Integration with Semantic Kernel ChatCompletionAgent
- ✅ Real-world insurance application with comprehensive analysis

**Advanced Extensions** (Optional):
- Add more specialized agents (Underwriter, Compliance, etc.)
- Implement parallel agent processing for better performance
- Add persistent storage for customer profiles and recommendations
- Create a web interface for the multi-agent system
- Add conversation memory between agents

**Resources**:
- [Module 4: Multi-Agent Systems](../modules/04-multi-agent.md) - Deep dive theory
- [Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/) - Official documentation

---

*You've now mastered both single-agent and multi-agent systems with Semantic Kernel! 🚀*
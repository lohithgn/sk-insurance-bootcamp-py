# Module 4: Multi-Agent Systems with Semantic Kernel

**Duration**: 75 minutes  
**Objective**: Design and implement sophisticated multi-agent systems using Semantic Kernel's ChatCompletionAgent framework for complex insurance workflows with orchestration, communication, and collaborative decision-making.

## Learning Outcomes
By completing this module, you will be able to:
- Design multi-agent architectures using Semantic Kernel's agent framework
- Implement agent orchestration with multiple ChatCompletionAgents
- Create specialized agents with distinct personalities and responsibilities
- Build collaborative decision-making systems with agent communication
- Handle multi-agent coordination for policy recommendations
- Apply orchestration patterns for complex business processes

## Prerequisites
- Completed [Module 3: Single Agent Architecture](03-single-agent.md)
- Successfully completed [Lab 2: PolicyAdvisorAgent](../labs/lab2-policy-advisor.md)
- Understanding of ChatCompletionAgent framework
- Familiarity with async programming patterns

---

## Multi-Agent Architecture with Semantic Kernel

### When to Use Multiple Agents vs Single Agent

**Single Agent** ✅ for:
- Simple, focused conversations
- Single domain expertise
- Straightforward question-answer workflows

**Multi-Agent** ✅ for:
- Complex decision-making requiring different perspectives
- Specialized expertise areas working together
- Scenarios where multiple viewpoints add value
- Collaborative analysis and recommendations

### Policy Recommendation Multi-Agent System Architecture

```
                    📋 CUSTOMER REQUEST
                         ⬇️
                ┌─────────────────────────┐
                │   Policy Orchestrator   │
                │    (Coordinator)        │
                └─────────────────────────┘
                         ⬇️
           ┌─────────────────────────────────────┐
           │     Customer Profile Building       │
           │   • Extract age, income, family     │
           │   • Identify insurance needs        │
           │   • Apply intelligent defaults      │
           └─────────────────────────────────────┘
                         ⬇️
    ┌────────────────────────────────────────────────────────────┐
    │                PARALLEL AGENT ANALYSIS                     │
    └────────────────────────────────────────────────────────────┘
                         ⬇️
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │    👥 EMMA   │  │ 🔬 DR. MARTINEZ │  │   🎯 ALEX   │  │ 💰 JORDAN  │
    │   Customer   │  │     Risk        │  │   Product   │  │   Pricing   │
    │   Analyst    │  │   Specialist    │  │   Expert    │  │   Analyst   │
    └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
           │                │                │                │
           ▼                ▼                ▼                ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │   ANALYSIS   │  │   ANALYSIS   │  │   ANALYSIS   │  │   ANALYSIS   │
    │             │  │             │  │             │  │             │
    │• Life needs │  │• Risk score │  │• Term life  │  │• Monthly:   │
    │• Coverage   │  │• Health     │  │• Whole life │  │  $67.50     │
    │  amount:    │  │  factors    │  │• Universal  │  │• Annual:    │
    │  $750,000   │  │• Category:  │  │• Features   │  │  $810       │
    │• Family     │  │  Standard   │  │• Riders     │  │• Affordab-  │
    │  protection │  │• Occupation │  │• Comparison │  │ ility: ✅  │
    │• Priority:  │  │  impact     │  │• Best match │  │• Value      │
    │  HIGH       │  │• Premium    │  │• Recommend: │  │  analysis   │
    │             │  │  impact     │  │  20-Yr Term │  │             │
    └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
           │                │                │                │
           └────────────────┼────────────────┼────────────────┘
                           │                │
                           ▼                ▼
                    ┌─────────────────────────────┐
                    │     SYNTHESIS ENGINE        │
                    │   (Orchestrator Logic)      │
                    │                            │
                    │ • Combine all analyses     │
                    │ • Extract key insights     │
                    │ • Resolve conflicts        │
                    │ • Generate final rec       │
                    └─────────────────────────────┘
                              ⬇️
                    ┌─────────────────────────────┐
                    │    COMPREHENSIVE            │
                    │    RECOMMENDATION           │
                    │                            │
                    │ ✅ Coverage: $750,000       │
                    │ ✅ Product: 20-Year Term    │
                    │ ✅ Risk: Standard           │
                    │ ✅ Premium: $67.50/month    │
                    │ ✅ Next Steps Defined       │
                    └─────────────────────────────┘
```

### Agent Specialization Strategy

For policy recommendations, we leverage multiple specialized agents:
- **👥 Emma (Customer Analyst)**: Understands customer needs and situation
- **🔬 Dr. Martinez (Risk Specialist)**: Evaluates risk factors and categories  
- **🎯 Alex (Product Expert)**: Knows all available insurance products
- **💰 Jordan (Pricing Analyst)**: Calculates accurate premium estimates

```python
import asyncio
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import re

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
    PRICING_ANALYST = "pricing_analyst"

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

class BaseInsuranceAgent:
    """Base class for specialized insurance agents"""
    
    def __init__(self, role: AgentRole, name: str, instructions: str):
        self.role = role
        self.name = name
        self.instructions = instructions
        self.kernel = None
        self.agent = None
        
    async def initialize(self, kernel: Kernel):
        """Initialize the agent"""
        self.kernel = kernel
        
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
        """Analyze customer from this agent's perspective"""
        
        # Create focused prompt for this agent's analysis
        analysis_prompt = f"""Analyze this customer from your specialized perspective:

CUSTOMER REQUEST: {customer_request}

CUSTOMER PROFILE:
• Age: {customer_profile.age}
• Income: ${customer_profile.income:,}
• Family Status: {customer_profile.family_status}
• Occupation: {customer_profile.occupation}
• Health Status: {customer_profile.health_status}
• Risk Factors: {', '.join(customer_profile.risk_factors) if customer_profile.risk_factors else 'None'}

Provide your specialized analysis and recommendations from your area of expertise."""

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
            confidence=0.85,  # Default confidence
            key_factors=self._extract_key_factors(analysis_text)
        )
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract key recommendations from analysis text"""
        # Simple extraction - in production could use more sophisticated NLP
        recommendations = []
        lines = text.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['recommend', 'suggest', 'should']):
                recommendations.append(line.strip())
        return recommendations[:3]  # Top 3 recommendations
    
    def _extract_key_factors(self, text: str) -> List[str]:
        """Extract key factors mentioned in analysis"""
        factors = []
        # Look for key terms relevant to insurance
        key_terms = ['age', 'income', 'family', 'risk', 'health', 'occupation', 'coverage', 'premium']
        for term in key_terms:
            if term in text.lower():
                factors.append(term)
        return factors

class CustomerAnalystAgent(BaseInsuranceAgent):
    """👥 Agent specializing in understanding customer needs and situation"""
    
    def __init__(self):
        instructions = """You are Emma, a Customer Analyst who specializes in understanding customer needs and situations.

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
        
        super().__init__(AgentRole.CUSTOMER_ANALYST, "Emma - Customer Analyst", instructions)
    
    async def initialize(self, kernel: Kernel):
        customer_plugin = CustomerAnalysisPlugin()
        kernel.add_plugin(customer_plugin, plugin_name="CustomerTools")
        await super().initialize(kernel)

class RiskSpecialistAgent(BaseInsuranceAgent):
    """🔬 Agent specializing in risk assessment and categorization"""
    
    def __init__(self):
        instructions = """You are Dr. Martinez, a Risk Assessment Specialist with expertise in evaluating insurance risks.

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
        
        super().__init__(AgentRole.RISK_SPECIALIST, "Dr. Martinez - Risk Specialist", instructions)
    
    async def initialize(self, kernel: Kernel):
        risk_plugin = RiskAnalysisPlugin()
        kernel.add_plugin(risk_plugin, plugin_name="RiskTools")
        await super().initialize(kernel)

class ProductExpertAgent(BaseInsuranceAgent):
    """🎯 Agent specializing in insurance products and features"""
    
    def __init__(self):
        instructions = """You are Alex, a Product Expert with comprehensive knowledge of all insurance products.

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
        
        super().__init__(AgentRole.PRODUCT_EXPERT, "Alex - Product Expert", instructions)
    
    async def initialize(self, kernel: Kernel):
        product_plugin = ProductAnalysisPlugin()
        kernel.add_plugin(product_plugin, plugin_name="ProductTools")
        await super().initialize(kernel)

class PricingAnalystAgent(BaseInsuranceAgent):
    """💰 Agent specializing in premium calculations and cost analysis"""
    
    def __init__(self):
        instructions = """You are Jordan, a Pricing Analyst who specializes in premium calculations and cost analysis.

EXPERTISE:
- Premium calculations
- Cost projections
- Affordability analysis
- Value comparisons

YOUR ANALYSIS SHOULD FOCUS ON:
1. Accurate premium estimates for recommended coverage
2. Cost comparisons between different options
3. Affordability assessment based on income
4. Value analysis (cost per $1000 of coverage)

APPROACH:
- Calculate realistic premium estimates
- Consider their budget and affordability
- Compare costs across different product options
- Show value propositions clearly

Provide accurate, realistic cost estimates and affordability guidance."""
        
        super().__init__(AgentRole.PRICING_ANALYST, "Jordan - Pricing Analyst", instructions)
    
    async def initialize(self, kernel: Kernel):
        pricing_plugin = PricingAnalysisPlugin()
        kernel.add_plugin(pricing_plugin, plugin_name="PricingTools")
        await super().initialize(kernel)

# Specialized Plugins for Each Agent

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
    
    @kernel_function(
        name="compare_policy_features",
        description="Compare key features across different policy types"
    )
    def compare_policy_features(self, primary_need: str) -> str:
        """Compare policy features"""
        
        return f"""Policy Feature Comparison:

TERM LIFE:
• Coverage Period: 10-30 years
• Premium: Low, increases after term
• Cash Value: None
• Best For: Temporary high coverage needs

WHOLE LIFE:
• Coverage Period: Lifetime
• Premium: High, level for life
• Cash Value: Guaranteed growth
• Best For: Permanent protection, estate planning

UNIVERSAL LIFE:
• Coverage Period: Lifetime (if funded properly)
• Premium: Flexible, can vary
• Cash Value: Market-based growth
• Best For: Advanced planning, investment growth

RECOMMENDATION: Based on "{primary_need}", Term Life typically provides the best value for protection needs."""

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

# Multi-Agent Policy Recommendation System

class PolicyRecommendationOrchestrator:
    """Orchestrator that coordinates multiple agents for comprehensive policy recommendations"""
    
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
            PricingAnalystAgent()
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
        
        for role, agent in self.agents.items():
            print(f"   🔍 {agent.name} analyzing...")
            analysis = await agent.analyze_customer(customer_profile, customer_request)
            analyses[role] = analysis
            print(f"   ✅ {agent.name} analysis complete")
        
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
        pricing_analysis = analyses[AgentRole.PRICING_ANALYST].analysis
        
        # Create synthesis
        synthesis = {
            "recommended_coverage_amount": self._extract_coverage_amount(customer_analysis),
            "recommended_product_type": self._extract_product_type(product_analysis),
            "risk_category": self._extract_risk_category(risk_analysis),
            "estimated_monthly_premium": self._extract_premium(pricing_analysis),
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
        # Simple extraction - look for dollar amounts
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

# Demo and Testing

async def demo_multi_agent_system():
    """Demonstrate the multi-agent policy recommendation system"""
    
    print("🏢 Multi-Agent Policy Recommendation System Demo")
    print("=" * 65)
    
    # Initialize system
    orchestrator = PolicyRecommendationOrchestrator()
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
            agent_name = analysis.agent_role.value.replace('_', ' ').title()
            print(f"   {agent_name}: {len(analysis.recommendations)} recommendations provided")
        
        print("-" * 50)

async def demo_individual_agents():
    """Demonstrate individual agent capabilities"""
    
    print("\n🔬 Individual Agent Capabilities Demo")
    print("=" * 45)
    
    # Initialize system
    orchestrator = PolicyRecommendationOrchestrator()
    await orchestrator.initialize()
    
    # Sample customer profile
    sample_profile = CustomerProfile(age=35, income=80000, family_status="family")
    sample_request = "I need $500,000 life insurance for my family"
    
    # Test each agent individually
    for role, agent in orchestrator.agents.items():
        print(f"\n🎯 Testing {agent.name}")
        print(f"   Specialization: {role.value.replace('_', ' ').title()}")
        
        analysis = await agent.analyze_customer(sample_profile, sample_request)
        
        print(f"   Analysis Preview: {analysis.analysis[:100]}...")
        print(f"   Recommendations: {len(analysis.recommendations)} provided")
        print(f"   Key Factors: {', '.join(analysis.key_factors[:3])}")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(demo_multi_agent_system())
    asyncio.run(demo_individual_agents())
```

---

## Multi-Agent System Architecture Breakdown

### 🏗️ **System Components**

```
┌─────────────────────────────────────────────────────────┐
│                    ARCHITECTURE LAYERS                  │
└─────────────────────────────────────────────────────────┘

📱 INPUT LAYER
    └── Customer Request (Natural Language)

🧠 PROFILE LAYER  
    └── CustomerProfile (Intelligent Defaults + Extraction)

🤖 AGENT LAYER (4 Specialized Agents)
    ├── 👥 Emma: Customer Analyst
    ├── 🔬 Dr. Martinez: Risk Specialist  
    ├── 🎯 Alex: Product Expert
    └── 💰 Jordan: Pricing Analyst

🔧 PLUGIN LAYER (Domain-Specific Tools)
    ├── CustomerAnalysisPlugin
    ├── RiskAnalysisPlugin
    ├── ProductAnalysisPlugin  
    └── PricingAnalysisPlugin

⚙️ ORCHESTRATION LAYER
    └── PolicyRecommendationOrchestrator

📊 OUTPUT LAYER
    └── Comprehensive Recommendation
```

### 🎯 **Agent Specializations**

| Agent | Symbol | Role | Key Functions | Output |
|-------|--------|------|---------------|---------|
| **Emma** | 👥 | Customer Analyst | Needs assessment, coverage calculation | Coverage amount, priorities |
| **Dr. Martinez** | 🔬 | Risk Specialist | Health/lifestyle evaluation, risk scoring | Risk category, premium impact |
| **Alex** | 🎯 | Product Expert | Product matching, feature comparison | Product recommendations |
| **Jordan** | 💰 | Pricing Analyst | Premium calculation, affordability | Cost estimates, value analysis |

## Key Multi-Agent Concepts Demonstrated

### 1. **Agent Specialization**

Each agent has a distinct area of expertise:

```python
# Customer Analyst - Understands needs and situation
CustomerAnalystAgent() → Coverage needs, life situation analysis

# Risk Specialist - Evaluates risk factors  
RiskSpecialistAgent() → Health, occupation, lifestyle risks

# Product Expert - Knows all insurance products
ProductExpertAgent() → Policy types, features, comparisons

# Pricing Analyst - Calculates accurate costs
PricingAnalystAgent() → Premium estimates, affordability analysis
```

### 2. **Collaborative Analysis**

Multiple agents analyze the same customer from different perspectives:

```python
# Each agent provides specialized insight
customer_insight = customer_analyst.analyze_customer(profile, request)
risk_insight = risk_specialist.analyze_customer(profile, request)  
product_insight = product_expert.analyze_customer(profile, request)
pricing_insight = pricing_analyst.analyze_customer(profile, request)

# Orchestrator synthesizes all insights
final_recommendation = synthesize_recommendations(all_insights)
```

### 3. **Simplified Coordination**

No complex handoffs - agents work independently then combine results:

```python
# Simple coordination pattern
analyses = {}
for role, agent in self.agents.items():
    analyses[role] = await agent.analyze_customer(profile, request)

# Synthesize results from all agents
return self._synthesize_recommendations(analyses, profile)
```

### 4. **Specialized Tools per Agent**

Each agent has domain-specific plugins:

```python
CustomerAnalysisPlugin() → assess_coverage_needs(), prioritize_insurance_needs()
RiskAnalysisPlugin() → calculate_risk_score(), assess_health_factors()
ProductAnalysisPlugin() → recommend_product_types(), compare_policy_features()
PricingAnalysisPlugin() → calculate_premium_estimate(), affordability_analysis()
```

---

## Benefits of This Multi-Agent Approach

### ✅ **Specialized Expertise**
- Each agent focuses on their area of strength
- Deeper analysis in each domain
- More accurate recommendations

### ✅ **Comprehensive Coverage**
- Customer needs from multiple angles
- Risk factors thoroughly evaluated  
- Product options fully explored
- Costs accurately estimated

### ✅ **Simple Architecture**
- No complex agent handoffs
- Independent agent analysis
- Clean synthesis of results
- Easy to understand and maintain

### ✅ **Scalable Design**
- Easy to add new specialist agents
- Agents can be enhanced independently
- Clear separation of concerns

---

## Multi-Agent vs Single Agent for Policy Recommendations

| Aspect | Single Agent | Multi-Agent System |
|--------|-------------|-------------------|
| **Expertise Depth** | General knowledge | Deep specialized knowledge |
| **Analysis Quality** | Good overall | Excellent in each domain |
| **Recommendation Accuracy** | Solid | Highly accurate |
| **System Complexity** | Simple | Moderate |
| **Maintenance** | Easy | Modular and manageable |
| **Best For** | Simple recommendations | Complex policy decisions |

---

## Success Criteria

By completing this module, you should be able to:

✅ **Design multi-agent systems** with specialized roles for policy recommendations  
✅ **Implement agent coordination** without complex handoff mechanisms  
✅ **Create domain-specific agents** with specialized tools and expertise  
✅ **Synthesize insights** from multiple agents into comprehensive recommendations  
✅ **Apply orchestration patterns** for collaborative decision-making  
✅ **Understand when multi-agent approaches add value** over single agents  

---

## Next Steps

**Ready to build your own multi-agent system!**

**Next**: [Lab 3: Multi-Agent System](../labs/lab3-multi-agent.md) - Implement this multi-agent architecture hands-on!

**Key Takeaways**:
- Multi-agent systems excel when different specialized expertise is needed
- Agents can work independently and have their results synthesized
- Specialization leads to higher quality analysis and recommendations  
- Simple coordination often works better than complex handoff mechanisms
- Each agent should have clear, focused responsibilities
- Orchestrators synthesize multiple perspectives into final recommendations

**Advanced Applications**:
- Add more specialist agents (compliance, underwriting, etc.)
- Implement agent voting on recommendations
- Create confidence scoring across agents
- Add real-time policy database integration
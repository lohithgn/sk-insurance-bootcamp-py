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

## Building a Simple Policy Recommendation Agent

Let's build a focused agent that demonstrates core concepts:

```python
import asyncio
import os
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
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
        
        # 2. Add simple plugins (tools) for the agent
        policy_plugin = PolicyPlugin()
        self.kernel.add_plugin(policy_plugin, plugin_name="PolicyTools")
        
        # 3. Create the agent with personality and tool access
        self.agent = ChatCompletionAgent(
            service_id="azure_openai",
            kernel=self.kernel,
            name="InsuranceAgent",
            instructions="""You are a friendly insurance advisor helping customers find the right coverage.

PERSONALITY:
- Warm and approachable, but professional
- Ask questions to understand their needs
- Explain recommendations clearly
- Use your tools to get accurate policy information

PROCESS:
1. Learn about the customer (age, family, income, needs)  
2. Use your PolicyTools to find suitable options
3. Explain why you're recommending specific policies
4. Answer any questions they have

Remember: You have access to policy search and premium calculation tools - use them!""",
            execution_settings=OpenAIChatPromptExecutionSettings(
                service_id="azure_openai",
                max_tokens=800,
                temperature=0.7,
                function_call_behavior=FunctionCallBehavior.EnableFunctions(auto_invoke=True)
            )
        )
        
        print("? Simple Insurance Agent ready!")
    
    async def chat(self, user_message: str) -> str:
        """Have a conversation with the agent"""
        
        # Add user message and get agent response
        self.chat_history.add_user_message(user_message)
        response = await self.agent.invoke(self.chat_history)
        
        # Return the agent's response
        return response[-1].content if response else "I didn't understand that. Could you try again?"

# Simple Plugin with Static Data - Focus on Agent Concepts, Not Complex Logic

class PolicyPlugin:
    """Simple plugin with static policy data to keep focus on agent concepts"""
    
    @kernel_function(
        name="find_life_policies",
        description="Find life insurance policies for a customer based on their age"
    )
    def find_life_policies(self, age: int) -> str:
        """Find suitable life insurance policies"""
        
        # Static policy data - keep it simple
        policies = [
            {"name": "BasicTerm20", "type": "20-Year Term", "max_age": 65, "rate": "$0.60 per $1K"},
            {"name": "WholeCare", "type": "Whole Life", "max_age": 75, "rate": "$2.50 per $1K"},  
            {"name": "SimpleTerm10", "type": "10-Year Term", "max_age": 60, "rate": "$0.45 per $1K"}
        ]
        
        # Filter by age
        suitable = [p for p in policies if age <= p["max_age"]]
        
        if not suitable:
            return f"No policies available for age {age}"
        
        result = f"Found {len(suitable)} life insurance options for age {age}:\n\n"
        for i, policy in enumerate(suitable, 1):
            result += f"{i}. {policy['name']} ({policy['type']})\n"
            result += f"   Rate: {policy['rate']} of coverage\n\n"
        
        return result
    
    @kernel_function(
        name="calculate_premium",
        description="Calculate monthly premium for life insurance based on age and coverage amount"
    )
    def calculate_premium(self, age: int, coverage_amount: int, policy_type: str = "term") -> str:
        """Calculate insurance premium"""
        
        # Simple premium calculation with static rates
        if policy_type.lower() == "term":
            base_rate = 0.6  # $0.60 per $1000
            age_factor = 1.0 + ((age - 25) * 0.02)  # 2% increase per year after 25
        else:  # whole life
            base_rate = 2.5  # $2.50 per $1000
            age_factor = 1.0 + ((age - 25) * 0.015)  # 1.5% increase per year after 25
        
        monthly_premium = (coverage_amount / 1000) * base_rate * age_factor
        annual_premium = monthly_premium * 12
        
        return f"""Premium Estimate:
• Coverage: ${coverage_amount:,} {policy_type} life insurance
• Age: {age}
• Monthly Premium: ${monthly_premium:.2f}
• Annual Premium: ${annual_premium:.2f}

*Rates shown are estimates for illustration purposes"""
    
    @kernel_function(
        name="get_coverage_recommendation", 
        description="Recommend coverage amount based on income and family situation"
    )
    def get_coverage_recommendation(self, annual_income: int, has_family: bool = False) -> str:
        """Recommend appropriate coverage amount"""
        
        # Simple rule-based recommendation
        if has_family:
            multiplier = 10  # 10x income for families
            reason = "to replace income and cover family expenses"
        else:
            multiplier = 6   # 6x income for singles  
            reason = "to cover debts and final expenses"
        
        recommended_coverage = annual_income * multiplier
        
        return f"""Coverage Recommendation:
• Recommended Amount: ${recommended_coverage:,}
• Calculation: {multiplier}x your annual income of ${annual_income:,}
• Reasoning: This amount should be sufficient {reason}

This follows standard industry guidelines for life insurance coverage."""

# Simple Test to Show Agent in Action

async def demo_simple_agent():
    """Demonstrate the agent with a realistic conversation"""
    
    # Initialize agent
    agent = SimpleInsuranceAgent()
    await agent.initialize()
    
    print("?? Insurance Agent Demo")
    print("=" * 50)
    
    # Simulate realistic conversation
    conversation = [
        "Hi, I'm looking for life insurance advice",
        "I'm 35 years old, married with kids, and make $80,000 per year", 
        "What coverage amount would you recommend?",
        "Can you show me what policies are available and what they'd cost?",
        "The 20-year term looks good. What would $800,000 coverage cost me?"
    ]
    
    for i, message in enumerate(conversation, 1):
        print(f"\n?? Customer: {message}")
        response = await agent.chat(message)
        print(f"?? Agent: {response}")
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
With `FunctionCallBehavior.EnableFunctions(auto_invoke=True)`, the agent automatically calls functions when needed:

```python
# User says: "I'm 30 and need life insurance"
# Agent automatically calls find_life_policies(age=30) 

# User says: "What would $500K cost me?"
# Agent automatically calls calculate_premium(age=30, coverage_amount=500000)
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
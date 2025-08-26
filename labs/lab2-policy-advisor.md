# Lab 2: Build a PolicyAdvisorAgent with Semantic Kernel

**Duration**: 45 minutes  
**Objective**: Build a conversational insurance agent using SK's ChatCompletionAgent framework that can recommend policies using simple plugins.

## Learning Outcomes
By completing this lab, you will be able to:
- Create a ChatCompletionAgent with personality and tools
- Build simple plugins for the agent to use automatically  
- Handle multi-turn conversations with built-in memory
- Configure agent behavior through system instructions
- Test agent functionality with realistic scenarios

## Prerequisites
- Completed [Module 3: Single Agent Architecture](../modules/03-single-agent.md)
- Working Semantic Kernel environment
- Azure OpenAI service configured

---

## What We're Building

A conversational insurance agent that:
- **Remembers** customer information throughout the conversation
- **Uses tools** to find policies and calculate premiums automatically
- **Provides recommendations** based on customer needs
- **Maintains personality** through system instructions

---

## Step 1: Create the Simple Policy Plugin

```python
# File: policy_data_plugin.py

from semantic_kernel.functions import kernel_function

class PolicyDataPlugin:
    """Simple insurance policy plugin with static data for agent to use"""
    
    @kernel_function(
        name="find_life_policies",
        description="Find life insurance policies suitable for customer's age"
    )
    def find_life_policies(self, age: int) -> str:
        """Find life insurance policies based on age"""
        
        # Simple static policy data
        policies = [
            {"name": "TermLife20", "type": "20-Year Term", "max_age": 65, "rate": "$0.60 per $1K"},
            {"name": "WholeCare", "type": "Whole Life", "max_age": 75, "rate": "$2.50 per $1K"},
            {"name": "SimpleTerm", "type": "10-Year Term", "max_age": 60, "rate": "$0.45 per $1K"}
        ]
        
        # Filter policies by age eligibility
        suitable_policies = [p for p in policies if age <= p["max_age"]]
        
        if not suitable_policies:
            return f"No policies available for age {age}"
        
        # Format response
        result = f"Found {len(suitable_policies)} life insurance options for age {age}:\n\n"
        for i, policy in enumerate(suitable_policies, 1):
            result += f"{i}. {policy['name']} ({policy['type']})\n"
            result += f"   Rate: {policy['rate']} of coverage\n"
            result += f"   Available up to age {policy['max_age']}\n\n"
        
        return result
    
    @kernel_function(
        name="calculate_premium", 
        description="Calculate monthly premium for life insurance based on age and coverage amount"
    )
    def calculate_premium(self, age: int, coverage_amount: int, policy_type: str = "term") -> str:
        """Calculate insurance premium estimate"""
        
        # Simple premium calculation with static rates
        if policy_type.lower() == "term":
            base_rate = 0.6  # $0.60 per $1000 coverage
            age_factor = 1.0 + ((age - 25) * 0.02)  # 2% increase per year after 25
        else:  # whole life
            base_rate = 2.5  # $2.50 per $1000 coverage
            age_factor = 1.0 + ((age - 25) * 0.015)  # 1.5% increase per year after 25
        
        # Calculate premiums
        monthly_premium = (coverage_amount / 1000) * base_rate * age_factor
        annual_premium = monthly_premium * 12
        
        return f"""Premium Calculation:
• Coverage: ${coverage_amount:,} {policy_type} life insurance
• Age: {age}
• Monthly Premium: ${monthly_premium:.2f}
• Annual Premium: ${annual_premium:.2f}

*Estimates for illustration purposes only"""
    
    @kernel_function(
        name="recommend_coverage",
        description="Recommend appropriate coverage amount based on income and family situation"
    )
    def recommend_coverage(self, annual_income: int, has_family: bool = False) -> str:
        """Recommend coverage amount using industry guidelines"""
        
        # Standard industry recommendation rules
        if has_family:
            multiplier = 10  # 10x income for families with dependents
            reason = "to replace income and support family expenses"
        else:
            multiplier = 6   # 6x income for single individuals
            reason = "to cover debts and final expenses"
        
        recommended_amount = annual_income * multiplier
        
        return f"""Coverage Recommendation:
• Recommended Amount: ${recommended_amount:,}
• Calculation: {multiplier}x your ${annual_income:,} annual income
• Reasoning: This should be sufficient {reason}

This follows standard insurance industry guidelines."""
```

---

## Step 2: Build the PolicyAdvisorAgent

```python
# File: policy_advisor_agent.py

import asyncio
import os
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

from policy_data_plugin import PolicyDataPlugin

# Load environment variables
load_dotenv()

class PolicyAdvisorAgent:
    """Insurance advisor agent built with SK ChatCompletionAgent framework"""
    
    def __init__(self):
        self.kernel = None
        self.agent = None
        self.chat_history = ChatHistory()
    
    async def initialize(self):
        """Initialize the agent with kernel, plugins, and personality"""
        
        print("?? Initializing PolicyAdvisorAgent...")
        
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
        policy_plugin = PolicyDataPlugin()
        self.kernel.add_plugin(policy_plugin, plugin_name="PolicyTools")
        
        # 3. Create ChatCompletionAgent with personality
        execution_settings = OpenAIChatPromptExecutionSettings(
            service_id="azure_openai",
            max_tokens=800,
            temperature=0.7,
            function_call_behavior=FunctionCallBehavior.EnableFunctions(auto_invoke=True)
        )
        
        self.agent = ChatCompletionAgent(
            service_id="azure_openai",
            kernel=self.kernel,
            name="PolicyAdvisorAgent",
            instructions=self._get_agent_instructions(),
            execution_settings=execution_settings
        )
        
        print("? PolicyAdvisorAgent ready!")
    
    def _get_agent_instructions(self) -> str:
        """Define the agent's personality and behavior"""
        return """You are PolicyAdvisorAgent, a friendly and knowledgeable insurance advisor helping customers find the right life insurance coverage.

PERSONALITY:
- Warm, professional, and patient
- Ask thoughtful questions to understand needs
- Explain recommendations clearly without jargon
- Show genuine care for customer's financial protection

YOUR PROCESS:
1. DISCOVER: Learn about the customer
   - Age, income, family situation, current coverage, goals
   
2. ANALYZE: Use your PolicyTools to find options
   - Search for suitable policies based on their profile
   - Calculate premiums for different scenarios
   - Recommend appropriate coverage amounts

3. ADVISE: Provide personalized recommendations
   - Explain why specific policies fit their needs
   - Show premium calculations
   - Give clear next steps

IMPORTANT GUIDELINES:
- Always gather key information (age, income, family) before recommendations
- Use your tools to provide accurate, data-driven advice
- Explain your reasoning clearly
- Reference previous conversation points naturally
- If you need more information, ask specific questions

Remember: You're helping them make important financial decisions for their family's security."""
    
    async def chat(self, user_message: str) -> str:
        """Have a conversation turn with the user"""
        
        try:
            # Add user message to conversation
            self.chat_history.add_user_message(user_message)
            
            # Get agent response (automatically uses tools when needed)
            response = await self.agent.invoke(self.chat_history)
            
            # Return agent's response
            if response and len(response) > 0:
                return response[-1].content
            else:
                return "I apologize, I had trouble generating a response. Could you please try again?"
                
        except Exception as e:
            error_msg = f"I encountered an issue: {str(e)[:100]}... Let me try to help you anyway. What specific insurance questions do you have?"
            self.chat_history.add_assistant_message(error_msg)
            return error_msg
    
    def get_conversation_summary(self) -> dict:
        """Get conversation statistics"""
        user_messages = [msg for msg in self.chat_history.messages if msg.role.value == "user"]
        return {
            "total_messages": len(self.chat_history.messages),
            "user_turns": len(user_messages),
            "conversation_active": len(self.chat_history.messages) > 0
        }
    
    def reset_conversation(self):
        """Start a new conversation"""
        self.chat_history = ChatHistory()
        print("?? Conversation reset")
```

---

## Step 3: Test the Agent

```python
# File: test_agent.py

import asyncio
from policy_advisor_agent import PolicyAdvisorAgent

async def test_basic_conversation():
    """Test basic agent conversation flow"""
    
    print("?? Test 1: Basic Conversation Flow")
    print("=" * 50)
    
    # Initialize agent
    agent = PolicyAdvisorAgent()
    await agent.initialize()
    
    # Test conversation flow
    test_conversation = [
        "Hi, I'm looking for life insurance advice",
        "I'm 35 years old and married with 2 children",  
        "My annual income is about $75,000",
        "What coverage amount would you recommend for someone in my situation?",
        "Can you show me what policies are available for my age?",
        "What would $750,000 in term coverage cost me monthly?"
    ]
    
    for i, message in enumerate(test_conversation, 1):
        print(f"\n??? Customer (Turn {i}): {message}")
        response = await agent.chat(message)
        print(f"?? Agent: {response[:200]}...")
        
        # Show conversation stats
        stats = agent.get_conversation_summary()
        print(f"?? Stats: {stats['user_turns']} turns, {stats['total_messages']} total messages")
        print("-" * 60)

async def test_agent_memory():
    """Test that agent remembers context across conversation"""
    
    print("\n?? Test 2: Agent Memory & Context")
    print("=" * 50)
    
    agent = PolicyAdvisorAgent()
    await agent.initialize()
    
    # Build up context over multiple turns
    memory_test = [
        "Hi, I need insurance help",
        "I'm 28 years old",  # Agent should remember age
        "I make $60,000 per year",  # Agent should remember income
        "I'm single with no kids",  # Agent should remember family status
        "Now what do you recommend?"  # Agent should use all previous context
    ]
    
    for i, message in enumerate(memory_test, 1):
        print(f"\n?? Turn {i}: {message}")
        response = await agent.chat(message)
        
        # Check if agent references previous information
        if i == 5:  # Final turn
            context_words = ["28", "60,000", "60000", "single", "no kids", "remember", "mentioned"]
            context_found = any(word in response.lower() for word in context_words)
            if context_found:
                print("? Agent remembered previous context!")
            else:
                print("? Agent may not be using previous context")
        
        print(f"?? Agent: {response[:150]}...")
        print("-" * 40)

async def run_comprehensive_test():
    """Run all tests for the PolicyAdvisorAgent"""
    
    print("?? Starting Comprehensive PolicyAdvisorAgent Testing")
    print("=" * 60)
    
    try:
        # Run test suites
        await test_basic_conversation()
        await test_agent_memory()
        
        print("\n?? All tests completed!")
        print("? PolicyAdvisorAgent is working correctly!")
        
    except Exception as e:
        print(f"? Test suite failed: {str(e)}")

# Interactive testing function
async def interactive_test():
    """Interactive test mode for manual testing"""
    
    print("?? Interactive Test Mode")
    print("=" * 30)
    print("Type 'exit' to quit, 'reset' to start new conversation")
    
    agent = PolicyAdvisorAgent()
    await agent.initialize()
    
    while True:
        try:
            user_input = input("\n??? You: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("?? Goodbye!")
                break
            elif user_input.lower() == 'reset':
                agent.reset_conversation()
                continue
            elif user_input.strip() == '':
                print("Please enter a message or 'exit' to quit.")
                continue
            
            response = await agent.chat(user_input)
            print(f"?? Agent: {response}")
            
            # Show conversation stats
            stats = agent.get_conversation_summary()
            print(f"?? [{stats['user_turns']} turns]")
            
        except KeyboardInterrupt:
            print("\n?? Goodbye!")
            break
        except Exception as e:
            print(f"? Error: {str(e)}")

# Main execution
if __name__ == "__main__":
    print("PolicyAdvisorAgent Lab 2 Testing")
    print("Choose test mode:")
    print("1. Comprehensive automated tests")
    print("2. Interactive manual testing")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(run_comprehensive_test())
    elif choice == "2":
        asyncio.run(interactive_test())
    else:
        print("Invalid choice. Running comprehensive tests by default.")
        asyncio.run(run_comprehensive_test())
```

---

## Success Criteria & Validation

### ? Your Agent Should:

1. **Create Successfully**: Initialize without errors
2. **Remember Context**: Reference information from previous messages  
3. **Use Tools Automatically**: Call plugin functions when relevant
4. **Maintain Personality**: Respond as a professional insurance advisor
5. **Handle Edge Cases**: Gracefully manage invalid or unclear inputs

### ?? Validation Tests:

Run the test and verify:
- Agent remembers customer is 35, married, with children, income $75K
- Agent automatically searches policies when discussing options  
- Agent calculates premiums when asked about costs
- Agent provides coverage recommendations based on family situation

---

## ?? Congratulations!

You've successfully built a PolicyAdvisorAgent using the SK Agent Framework! 

**What you've accomplished:**
- ? Created a conversational AI agent with personality
- ? Integrated simple plugins that the agent uses automatically
- ? Implemented conversation memory with ChatHistory  
- ? Configured agent behavior through system instructions
- ? Built a production-ready insurance advisory system

**Next Steps:**
- **[Lab 3: Multi-Agent System](lab3-multi-agent.md)** - Build multiple agents working together
- **[Module 4: Multi-Agent Architecture](../modules/04-multi-agent.md)** - Learn orchestration patterns
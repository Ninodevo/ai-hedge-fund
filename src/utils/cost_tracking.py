"""Cost tracking utilities for LLM and API calls"""

from typing import Dict, Optional, Any
from enum import Enum


class ModelProvider(str, Enum):
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    OLLAMA = "OLLAMA"


def calculate_llm_cost(
    token_usage: Dict[str, Any],
    model_name: str,
    model_provider: str = "OPENAI"
) -> float:
    """
    Calculate cost for LLM API calls based on token usage and model pricing.
    
    Args:
        token_usage: Dictionary with token usage information
        model_name: Name of the model used
        model_provider: Provider of the model (default: OPENAI)
    
    Returns:
        Cost in USD
    """
    if not token_usage:
        return 0.0
    
    # Extract token counts
    input_tokens = token_usage.get('prompt_tokens') or token_usage.get('input_tokens', 0)
    output_tokens = token_usage.get('completion_tokens') or token_usage.get('output_tokens', 0)
    
    if model_provider.upper() != "OPENAI":
        # For non-OpenAI providers, return 0 for now (can be extended later)
        return 0.0
    
    # OpenAI pricing per million tokens (as of 2025)
    pricing = {
        "gpt-5": {"input": 1.25, "output": 10.00},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-5-nano": {"input": 0.10, "output": 0.80},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4.1": {"input": 2.50, "output": 10.00},  # Assuming similar to gpt-4o
    }
    
    # Find matching pricing (check if model_name starts with any key)
    model_pricing = None
    for model_key, prices in pricing.items():
        if model_name.lower().startswith(model_key.lower()):
            model_pricing = prices
            break
    
    # Default to GPT-5-mini pricing if not found
    if not model_pricing:
        model_pricing = pricing.get("gpt-5-mini", {"input": 0.25, "output": 2.00})
    
    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
    cost_usd = input_cost + output_cost
    
    return round(cost_usd, 6)


def extract_token_usage_from_llm_response(result: Any) -> Optional[Dict[str, Any]]:
    """
    Extract token usage information from an LLM response object.
    
    Args:
        result: The LLM response object (could be from LangChain, OpenAI, etc.)
    
    Returns:
        Dictionary with token usage information, or None if not available
    """
    token_usage = {}
    
    # Try to get token usage from response metadata
    if hasattr(result, 'response_metadata'):
        metadata = result.response_metadata
        if 'token_usage' in metadata:
            token_usage = metadata['token_usage']
        elif 'usage' in metadata:
            token_usage = metadata['usage']
    
    # Also check usage_metadata attribute (OpenAI SDK)
    if not token_usage and hasattr(result, 'usage_metadata'):
        usage_meta = result.usage_metadata
        token_usage = {
            'input_tokens': getattr(usage_meta, 'input_tokens', 0),
            'output_tokens': getattr(usage_meta, 'output_tokens', 0),
            'total_tokens': getattr(usage_meta, 'total_tokens', 0)
        }
    
    # Check for direct attributes (some LangChain wrappers)
    if not token_usage:
        if hasattr(result, 'prompt_tokens') or hasattr(result, 'completion_tokens'):
            token_usage = {
                'prompt_tokens': getattr(result, 'prompt_tokens', 0),
                'completion_tokens': getattr(result, 'completion_tokens', 0),
                'total_tokens': getattr(result, 'total_tokens', 0)
            }
    
    # Check response_metadata for OpenAI-style responses
    if not token_usage and hasattr(result, 'response_metadata'):
        metadata = result.response_metadata
        if isinstance(metadata, dict):
            if 'token_usage' in metadata:
                token_usage = metadata['token_usage']
            elif 'usage' in metadata:
                token_usage = metadata['usage']
    
    return token_usage if token_usage else None


class CostTracker:
    """Track costs across multiple API calls and LLM invocations"""
    
    def __init__(self):
        self.llm_costs: Dict[str, float] = {}  # agent_name -> cost
        self.api_costs: Dict[str, float] = {}  # api_name -> cost
        self.llm_token_usage: Dict[str, Dict[str, Any]] = {}  # agent_name -> token_usage
        self.api_call_counts: Dict[str, int] = {}  # api_name -> count
    
    def add_llm_cost(self, agent_name: str, cost: float, token_usage: Optional[Dict[str, Any]] = None):
        """Add LLM cost for an agent"""
        if agent_name not in self.llm_costs:
            self.llm_costs[agent_name] = 0.0
            self.llm_token_usage[agent_name] = {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            }
        
        self.llm_costs[agent_name] += cost
        
        if token_usage:
            self.llm_token_usage[agent_name]['prompt_tokens'] += token_usage.get('prompt_tokens', 0) or token_usage.get('input_tokens', 0)
            self.llm_token_usage[agent_name]['completion_tokens'] += token_usage.get('completion_tokens', 0) or token_usage.get('output_tokens', 0)
            self.llm_token_usage[agent_name]['total_tokens'] += token_usage.get('total_tokens', 0) or (
                self.llm_token_usage[agent_name]['prompt_tokens'] + self.llm_token_usage[agent_name]['completion_tokens']
            )
    
    def add_api_cost(self, api_name: str, cost: float, call_count: int = 1):
        """Add API cost"""
        if api_name not in self.api_costs:
            self.api_costs[api_name] = 0.0
            self.api_call_counts[api_name] = 0
        
        self.api_costs[api_name] += cost
        self.api_call_counts[api_name] += call_count
    
    def get_total_cost(self) -> float:
        """Get total cost across all LLM and API calls"""
        total_llm = sum(self.llm_costs.values())
        total_api = sum(self.api_costs.values())
        return round(total_llm + total_api, 6)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all costs"""
        return {
            "total_cost_usd": self.get_total_cost(),
            "llm_costs": {
                "total_usd": round(sum(self.llm_costs.values()), 6),
                "by_agent": {k: round(v, 6) for k, v in self.llm_costs.items()},
                "token_usage": self.llm_token_usage
            },
            "api_costs": {
                "total_usd": round(sum(self.api_costs.values()), 6),
                "by_api": {k: round(v, 6) for k, v in self.api_costs.items()},
                "call_counts": self.api_call_counts
            }
        }


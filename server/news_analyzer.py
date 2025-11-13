"""
News analyzer module for analyzing company news and extracting key insights.
"""
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import ChatPromptTemplate
import json
import os

from src.llm.models import get_model, get_model_info, ModelProvider


class FinancialMove(BaseModel):
    """Represents a financial move or action."""
    type: str = Field(description="Type of financial move (e.g., 'acquisition', 'earnings', 'dividend', 'buyback', 'guidance', 'partnership', 'product_launch', 'regulatory', 'management_change')")
    description: str = Field(description="Description of the financial move")
    impact: str = Field(default="neutral", description="Expected impact: 'positive', 'negative', or 'neutral'")
    magnitude: str = Field(default="medium", description="Magnitude: 'high', 'medium', or 'low'")


class NewsAnalysis(BaseModel):
    """Structured analysis of company news."""
    summary: str = Field(description="Brief summary of the most important news items")
    key_points: List[str] = Field(description="List of key points from the news")
    financial_moves: List[FinancialMove] = Field(default_factory=list, description="List of financial moves or actions identified")
    additional_info: Union[List[str], str] = Field(default_factory=list, description="Additional important information that could impact the stock")
    confidence: int = Field(description="Confidence score 0-100", ge=0, le=100)
    
    @field_validator('additional_info', mode='before')
    @classmethod
    def normalize_additional_info(cls, v):
        """Convert string to list if needed."""
        if isinstance(v, str):
            # Split by newlines or return as single-item list
            if '\n' in v:
                return [line.strip() for line in v.split('\n') if line.strip()]
            return [v] if v.strip() else []
        return v if isinstance(v, list) else []


def analyze_news(
    news_items: List[Dict[str, Any]],
    symbol: str,
    api_keys: Optional[Dict[str, str]] = None,
    model_name: str = "gpt-5-mini",
    model_provider: str = "OPENAI"
) -> Optional[NewsAnalysis]:
    """
    Analyze company news and extract key insights, financial moves, and important information.
    
    Args:
        news_items: List of news items with title, source, date, url, sentiment
        symbol: Stock ticker symbol
        api_keys: Optional dictionary of API keys
        model_name: LLM model name to use
        model_provider: LLM provider name
        
    Returns:
        NewsAnalysis object with structured analysis, or None if analysis fails
    """
    if not news_items:
        return None
    
    # Limit to most recent 10 articles to avoid token limits
    recent_news = news_items[:10]
    
    # Format news items for the prompt
    news_text = []
    for idx, news in enumerate(recent_news, 1):
        title = news.get("title", "N/A")
        source = news.get("source", "Unknown")
        date = news.get("date", "Unknown")
        news_text.append(
            f"{idx}. Title: {title}\n   Source: {source}\n   Date: {date}"
        )
    
    news_content = "\n\n".join(news_text)
    
    # Create prompt template
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a financial news analyst specializing in extracting actionable insights from company news.

Your task is to analyze news articles about a company and provide:
1. A concise summary of the most important developments
2. Key points that investors should know
3. Financial moves (acquisitions, earnings, dividends, buybacks, guidance changes, partnerships, product launches, regulatory changes, management changes, etc.)
4. Additional important information that could impact the stock

Focus on:
- Financial implications and materiality
- Strategic moves and business developments
- Regulatory or legal issues
- Management changes
- Market positioning and competitive dynamics
- Any information that could significantly impact valuation or stock price

Be specific and factual. Avoid generic statements."""
        ),
        (
            "human",
            """Analyze the following news articles for {symbol}:

{news_content}

Provide a comprehensive analysis including:
- Summary: Brief overview of the most important developments
- Key Points: List of 5-10 most important points investors should know (must be an array of strings)
- Financial Moves: Identify any financial moves, strategic actions, or material events (acquisitions, earnings announcements, dividends, buybacks, guidance updates, partnerships, product launches, regulatory actions, management changes, etc.). For each move, you MUST include ALL four fields:
  * type: The type of financial move (string)
  * description: Description of the move (string)
  * impact: Expected impact - MUST be one of: "positive", "negative", or "neutral" (string, required)
  * magnitude: Magnitude - MUST be one of: "high", "medium", or "low" (string, required)
- Additional Info: Any other important information that could impact the stock (must be an array of strings, not a single string)
- Confidence: Your confidence in this analysis (0-100, integer)

IMPORTANT: 
- All financial_moves items MUST include "impact" and "magnitude" fields
- additional_info MUST be an array/list of strings, not a single string
- Return your response in valid JSON format matching the NewsAnalysis schema exactly."""
        )
    ])
    
    prompt = template.invoke({
        "symbol": symbol,
        "news_content": news_content
    })
    
    # Get model and API keys - use environment variables as fallback
    if api_keys is None:
        api_keys = {}
    
    # Add environment variables as fallback for common API keys
    if "OPENAI_API_KEY" not in api_keys:
        api_keys["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
    
    try:
        # Convert model_provider string to enum
        # Handle both "OPENAI" (enum key) and "OpenAI" (enum value) formats
        if isinstance(model_provider, str):
            try:
                # Try as enum key first (e.g., "OPENAI")
                model_provider_enum = ModelProvider[model_provider]
            except KeyError:
                # Try as enum value (e.g., "OpenAI")
                model_provider_enum = ModelProvider(model_provider)
        else:
            model_provider_enum = model_provider
        
        # get_model_info expects a string that matches the enum value (e.g., "OpenAI")
        # Since ModelProvider is a str Enum, we can use the value directly
        provider_value = model_provider_enum.value if hasattr(model_provider_enum, 'value') else str(model_provider_enum)
        model_info = get_model_info(model_name, provider_value)
        llm = get_model(model_name, model_provider_enum, api_keys)
        
        # Configure structured output for models that support JSON mode
        # If model_info is None or model has json_mode, use structured output
        use_structured_output = not (model_info and not model_info.has_json_mode())
        if use_structured_output:
            llm_structured = llm.with_structured_output(
                NewsAnalysis,
                method="json_mode",
            )
        else:
            llm_structured = None
        
        # Call LLM - try structured output first, fall back to manual parsing if it fails
        result = None
        raw_content = None
        
        if use_structured_output:
            try:
                result = llm_structured.invoke(prompt)
            except Exception as structured_error:
                # Structured output failed - likely due to field name mismatches
                # Fall back to getting raw JSON and parsing manually
                print(f"Structured output failed for {symbol}, falling back to manual parsing: {structured_error}")
                try:
                    # Get raw response without structured output
                    raw_response = llm.invoke(prompt)
                    if hasattr(raw_response, 'content'):
                        raw_content = raw_response.content
                    else:
                        raw_content = str(raw_response)
                except Exception as e:
                    print(f"Failed to get raw response for {symbol}: {e}")
                    raw_content = None
        else:
            # For non-json-mode models, get raw response
            raw_response = llm.invoke(prompt)
            if hasattr(raw_response, 'content'):
                raw_content = raw_response.content
            else:
                raw_content = str(raw_response)
        
        # If we have raw content (from fallback or non-json-mode), parse it manually
        if raw_content:
            parsed_result = _extract_json_from_response(raw_content)
            if parsed_result:
                # Always try to fix validation issues before first validation attempt
                fixed_result = _fix_validation_issues(parsed_result)
                if fixed_result:
                    try:
                        return NewsAnalysis(**fixed_result)
                    except Exception as validation_error:
                        print(f"Validation error for {symbol} after fixing: {validation_error}")
                        return None
                else:
                    print(f"Warning: Failed to fix validation issues for {symbol}")
                    return None
            else:
                print(f"Warning: Failed to extract JSON from response for {symbol}")
                return None
        
        # Handle structured output result
        if result is not None:
            # For models with json_mode or when model_info is None, result should be NewsAnalysis
            # But it might also be a dict if structured output failed
            if isinstance(result, dict):
                # Structured output returned a dict (might have wrong field names)
                fixed_result = _fix_validation_issues(result)
                if fixed_result:
                    try:
                        return NewsAnalysis(**fixed_result)
                    except Exception as validation_error:
                        print(f"Validation error for dict result for {symbol} after fixing: {validation_error}")
                        return None
                else:
                    print(f"Warning: Failed to fix validation issues for dict result for {symbol}")
                    return None
            elif isinstance(result, NewsAnalysis):
                # Try to get the raw data and validate/fix it
                try:
                    # Try to get dict representation - use model_dump with validation disabled
                    # to get raw data even if validation would fail
                    try:
                        result_dict = result.model_dump(mode='python')
                    except Exception:
                        # If model_dump fails, try to get dict via __dict__ or model_fields
                        result_dict = {}
                        for field_name in result.model_fields.keys():
                            try:
                                value = getattr(result, field_name, None)
                                result_dict[field_name] = value
                            except Exception:
                                pass
                    
                    # Always fix validation issues before re-validating
                    fixed_result = _fix_validation_issues(result_dict)
                    if fixed_result:
                        try:
                            return NewsAnalysis(**fixed_result)
                        except Exception as validation_error:
                            print(f"Validation error for structured output result for {symbol} after fixing: {validation_error}")
                            return None
                    else:
                        # If fixing failed, try to return the original if it's valid
                        try:
                            # Try to validate the original
                            result.model_validate(result.model_dump())
                            return result
                        except Exception:
                            print(f"Failed to fix validation issues for {symbol}")
                            return None
                except Exception as validation_error:
                    print(f"Error processing structured output result for {symbol}: {validation_error}")
                    return None
            else:
                print(f"Warning: Unexpected result type from structured output: {type(result)}")
                return None
        
        # If we got here, we have neither raw_content nor a valid result
        print(f"Warning: No valid result or raw content for {symbol}")
        return None
            
    except Exception as e:
        import traceback
        print(f"Error analyzing news for {symbol}: {e}")
        traceback.print_exc()
        return None


def _extract_json_from_response(content: str) -> Optional[dict]:
    """Extract JSON from markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7:]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
        else:
            # Try to parse as plain JSON
            return json.loads(content)
    except Exception as e:
        print(f"Error extracting JSON from response: {e}")
    return None


def _normalize_field_names(data: dict) -> dict:
    """Normalize field names to match the schema (handle case variations)."""
    # Create a comprehensive mapping for all possible variations
    field_mapping = {
        # NewsAnalysis fields - handle all case variations
        'summary': 'summary', 'Summary': 'summary', 'SUMMARY': 'summary',
        'key_points': 'key_points', 'KeyPoints': 'key_points', 'Key_Points': 'key_points',
        'keyPoints': 'key_points', 'KEY_POINTS': 'key_points',
        'financial_moves': 'financial_moves', 'FinancialMoves': 'financial_moves',
        'Financial_Moves': 'financial_moves', 'financialMoves': 'financial_moves',
        'FINANCIAL_MOVES': 'financial_moves',
        'additional_info': 'additional_info', 'AdditionalInfo': 'additional_info',
        'Additional_Info': 'additional_info', 'additionalInfo': 'additional_info',
        'ADDITIONAL_INFO': 'additional_info',
        'confidence': 'confidence', 'Confidence': 'confidence', 'CONFIDENCE': 'confidence',
        # FinancialMove fields
        'type': 'type', 'Type': 'type', 'TYPE': 'type',
        'description': 'description', 'Description': 'description', 'DESCRIPTION': 'description',
        'impact': 'impact', 'Impact': 'impact', 'IMPACT': 'impact',
        'magnitude': 'magnitude', 'Magnitude': 'magnitude', 'MAGNITUDE': 'magnitude',
    }
    
    normalized = {}
    for key, value in data.items():
        if isinstance(key, str):
            # Try exact match first
            normalized_key = field_mapping.get(key)
            if normalized_key is None:
                # Try case-insensitive match
                key_lower = key.lower()
                # Find matching key in mapping (case-insensitive)
                for map_key, map_value in field_mapping.items():
                    if map_key.lower() == key_lower:
                        normalized_key = map_value
                        break
                # If still not found, try converting common patterns
                if normalized_key is None:
                    # Convert CamelCase to snake_case
                    import re
                    snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', key).lower()
                    normalized_key = field_mapping.get(snake_case, snake_case)
        else:
            normalized_key = key
        
        normalized[normalized_key] = value
    
    return normalized


def _fix_validation_issues(data: dict) -> Optional[dict]:
    """Fix common validation issues in the parsed data."""
    try:
        # First normalize field names
        fixed_data = _normalize_field_names(data)
        
        # Fix financial_moves: ensure all have impact and magnitude
        if 'financial_moves' in fixed_data and isinstance(fixed_data['financial_moves'], list):
            normalized_moves = []
            for move in fixed_data['financial_moves']:
                if isinstance(move, dict):
                    normalized_move = _normalize_field_names(move)
                    if 'impact' not in normalized_move or not normalized_move['impact']:
                        normalized_move['impact'] = 'neutral'
                    if 'magnitude' not in normalized_move or not normalized_move['magnitude']:
                        normalized_move['magnitude'] = 'medium'
                    normalized_moves.append(normalized_move)
            fixed_data['financial_moves'] = normalized_moves
        
        # Fix additional_info: convert string to list
        if 'additional_info' in fixed_data:
            if isinstance(fixed_data['additional_info'], str):
                if '\n' in fixed_data['additional_info']:
                    fixed_data['additional_info'] = [line.strip() for line in fixed_data['additional_info'].split('\n') if line.strip()]
                else:
                    fixed_data['additional_info'] = [fixed_data['additional_info']] if fixed_data['additional_info'].strip() else []
            elif not isinstance(fixed_data['additional_info'], list):
                fixed_data['additional_info'] = []
        
        return fixed_data
    except Exception as e:
        print(f"Error fixing validation issues: {e}")
        return None

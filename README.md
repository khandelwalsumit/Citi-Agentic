```python

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain.tools import BaseTool
from typing import Any, Optional, List, Dict, Sequence, Union
from pydantic import Field, BaseModel
import json
import logging

from vertexai.generative_models import (
    GenerativeModel, 
    GenerationConfig, 
    Tool as VertexTool, 
    FunctionDeclaration,
    Content,
    Part
)

from core.auth import *

logger = logging.getLogger(__name__)


def _clean_schema(schema: dict) -> dict:
    """Remove keys that Vertex AI's FunctionDeclaration doesn't accept."""
    blocked_keys = {"additionalProperties", "title", "$defs", "definitions"}
    cleaned = {k: v for k, v in schema.items() if k not in blocked_keys}
    # Recursively clean nested properties
    if "properties" in cleaned:
        cleaned["properties"] = {
            k: _clean_schema(v) if isinstance(v, dict) else v
            for k, v in cleaned["properties"].items()
        }
    return cleaned


class VertexAILLM(BaseChatModel):
    """Custom Chat Model wrapper for Vertex AI Gemini models with tool calling and response schema support.
    
    This implementation maintains the exact Vertex AI tool calling logic using:
    - Part.from_dict() for function calls
    - Part.from_function_response() for function responses
    - response_schema for structured outputs
    """
    
    model_name: str = "gemini-2.0-flash-exp"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.95
    top_k: int = 40
    
    # Response schema support
    response_schema: Optional[Dict[str, Any]] = None
    response_mime_type: Optional[str] = None
    auto_execute_tools: bool = True
    
    # Use Field for mutable defaults
    tools: List[BaseTool] = Field(default_factory=list)
    vertex_tools: List[VertexTool] = Field(default_factory=list)
    
    # Non-pydantic fields
    gen_model: Optional[Any] = None
    generation_config: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_vertex_ai()

    def _initialize_vertex_ai(self):
        """Initialize Vertex AI with proper authentication."""
        try:
            token = get_coin_token()
            init_vertexai(token=token)
            
            # Build generation config with optional response schema
            gen_config_params = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "top_p": self.top_p,
                "top_k": self.top_k,
            }
            
            # Add response schema if provided
            if self.response_schema:
                gen_config_params["response_schema"] = self.response_schema
                # Default to JSON if schema is provided but mime type isn't
                gen_config_params["response_mime_type"] = self.response_mime_type or "application/json"
            
            self.generation_config = GenerationConfig(**gen_config_params)
            self.gen_model = GenerativeModel(self.model_name)
            
        except Exception as e:
            raise Exception(f"Error initializing Vertex AI: {str(e)}")

    def _langchain_tool_to_vertex_fd(self, tool: BaseTool) -> FunctionDeclaration:
        """Convert a LangChain tool to a Vertex FunctionDeclaration."""
        # Extract schema from tool
        if hasattr(tool, 'args_schema') and tool.args_schema:
            schema = tool.args_schema.model_json_schema()
            params = _clean_schema(schema)
        else:
            # Fallback if no schema
            params = {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Input to the tool"}
                },
                "required": ["input"]
            }
        
        return FunctionDeclaration(
            name=tool.name,
            description=tool.description or "Tool function",
            parameters=params if params.get("properties") else None,
        )

    def _convert_tools_to_vertex_format(self, tools: List[BaseTool]) -> List[VertexTool]:
        """Convert LangChain tools to Vertex AI tool format."""
        function_declarations = []
        
        for tool in tools:
            function_declarations.append(self._langchain_tool_to_vertex_fd(tool))
        
        # Wrap all function declarations in a single VertexTool
        return [VertexTool(function_declarations=function_declarations)]

    def _execute_function_call(self, function_name: str, function_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function call using the bound tools.
        
        Returns a dict suitable for Part.from_function_response().
        """
        if not self.tools:
            return {"result": f"No tools available to execute function: {function_name}"}
        
        # Find and execute the matching tool
        for tool in self.tools:
            if tool.name == function_name:
                try:
                    # Execute the tool with the provided arguments
                    result = tool.run(function_args)
                    
                    # Try to parse as JSON if it's a string
                    try:
                        if isinstance(result, str):
                            parsed_result = json.loads(result)
                            if isinstance(parsed_result, dict):
                                return parsed_result
                    except (json.JSONDecodeError, TypeError):
                        pass
                    
                    # Return as dict with 'result' key
                    return {"result": str(result)}
                    
                except Exception as e:
                    logger.error(f"Error executing function {function_name}: {str(e)}")
                    return {"result": f"Error executing function {function_name}: {str(e)}"}
        
        return {"result": f"Function {function_name} not found in available tools."}

    def _convert_messages_to_vertex_format(self, messages: List[BaseMessage]) -> List[Content]:
        """Convert LangChain messages to Vertex AI Content format."""
        contents = []
        tool_call_id_to_name: Dict[str, str] = {}
        
        for message in messages:
            if isinstance(message, HumanMessage):
                contents.append(Content(role="user", parts=[Part.from_text(message.content)]))
            elif isinstance(message, AIMessage):
                parts = []
                if message.content:
                    parts.append(Part.from_text(message.content))
                # Preserve prior assistant tool calls so Vertex can continue correctly.
                normalized_tool_calls: List[Dict[str, Any]] = []
                for tc in getattr(message, "tool_calls", []) or []:
                    normalized_tool_calls.append(
                        {
                            "id": tc.get("id"),
                            "name": tc.get("name"),
                            "args": tc.get("args", {}),
                        }
                    )
                # Backward-compat: some stacks store OpenAI-style tool_calls in additional_kwargs.
                if not normalized_tool_calls:
                    for tc in (message.additional_kwargs.get("tool_calls") or []):
                        fn = tc.get("function", {})
                        raw_args = fn.get("arguments", "{}")
                        try:
                            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                        except (json.JSONDecodeError, TypeError):
                            parsed_args = {}
                        normalized_tool_calls.append(
                            {
                                "id": tc.get("id"),
                                "name": fn.get("name"),
                                "args": parsed_args if isinstance(parsed_args, dict) else {},
                            }
                        )

                for tc in normalized_tool_calls:
                    tc_id = tc.get("id")
                    tc_name = tc.get("name")
                    if tc_id and tc_name:
                        tool_call_id_to_name[str(tc_id)] = str(tc_name)
                    parts.append(
                        Part.from_dict({
                            "function_call": {
                                "name": tc_name,
                                "args": tc.get("args", {}),
                            }
                        })
                    )
                if parts:
                    contents.append(Content(role="model", parts=parts))
            elif isinstance(message, SystemMessage):
                # System messages can be prepended as user messages in Vertex AI
                contents.append(Content(role="user", parts=[Part.from_text(f"System: {message.content}")]))
            elif isinstance(message, ToolMessage):
                # Map LangChain tool output back to Vertex function response format.
                tool_name = getattr(message, "name", None) or message.additional_kwargs.get("name")
                if not tool_name:
                    tool_call_id = getattr(message, "tool_call_id", None) or message.additional_kwargs.get("tool_call_id")
                    if tool_call_id:
                        tool_name = tool_call_id_to_name.get(str(tool_call_id))
                if tool_name:
                    response_payload: Dict[str, Any]
                    try:
                        parsed = json.loads(message.content) if isinstance(message.content, str) else message.content
                        response_payload = parsed if isinstance(parsed, dict) else {"result": parsed}
                    except (json.JSONDecodeError, TypeError):
                        response_payload = {"result": str(message.content)}
                    contents.append(
                        Content(
                            role="user",
                            parts=[
                                Part.from_function_response(
                                    name=tool_name,
                                    response=response_payload,
                                )
                            ],
                        )
                    )
                else:
                    logger.warning(
                        "Skipping ToolMessage because tool name could not be resolved. "
                        "tool_call_id=%s",
                        getattr(message, "tool_call_id", None),
                    )
        
        return contents

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from messages.
        
        This implementation uses the exact Vertex AI tool calling logic:
        - Part.from_dict() for function calls
        - Part.from_function_response() for function responses
        - response_schema for structured outputs
        """
        try:
            # Convert LangChain messages to Vertex AI format
            contents = self._convert_messages_to_vertex_format(messages)
            
            # Build generation config with optional response schema
            gen_config_params = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "top_p": self.top_p,
                "top_k": self.top_k,
            }
            
            if stop:
                gen_config_params["stop_sequences"] = stop
            
            # Add response schema if provided (can be overridden in kwargs)
            if "response_schema" in kwargs:
                gen_config_params["response_schema"] = kwargs["response_schema"]
                gen_config_params["response_mime_type"] = kwargs.get("response_mime_type", "application/json")
            elif self.response_schema:
                gen_config_params["response_schema"] = self.response_schema
                gen_config_params["response_mime_type"] = self.response_mime_type or "application/json"
            
            gen_config = GenerationConfig(**gen_config_params)
            
            # Build generate_content kwargs
            call_kwargs = {
                "generation_config": gen_config,
            }
            
            # Only pass tools if we have them
            if self.vertex_tools:
                call_kwargs["tools"] = self.vertex_tools
            
            # Handle multi-turn conversation for tool calls
            max_iterations = 5  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                # Call the model
                response = self.gen_model.generate_content(contents, **call_kwargs)
                
                # Handle empty response
                if not response.candidates:
                    logger.warning("Vertex AI returned no candidates.")
                    message = AIMessage(content="[Model returned no response. The request may have been blocked.]")
                    return ChatResult(generations=[ChatGeneration(message=message)])
                
                candidate = response.candidates[0]
                
                # Check if response was blocked
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = str(candidate.finish_reason)
                    if finish_reason not in ("1", "STOP", "FinishReason.STOP"):
                        logger.info(f"Vertex AI finish_reason: {finish_reason}")
                
                # Handle candidate with no content
                if not hasattr(candidate, "content") or not candidate.content:
                    message = AIMessage(content="[Model returned empty content]")
                    return ChatResult(generations=[ChatGeneration(message=message)])
                
                if not candidate.content.parts:
                    message = AIMessage(content="[Model returned no content parts]")
                    return ChatResult(generations=[ChatGeneration(message=message)])
                
                # Parse the response parts
                text_parts = []
                function_calls = []
                
                for part in candidate.content.parts:
                    # Check for text
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
                    
                    # Check for function call
                    fn_call = getattr(part, "function_call", None)
                    if fn_call and getattr(fn_call, "name", None):
                        args = {}
                        if fn_call.args:
                            try:
                                args = dict(fn_call.args)
                            except Exception:
                                args = json.loads(type(fn_call.args).to_json(fn_call.args))
                        
                        function_calls.append({
                            "name": fn_call.name,
                            "args": args
                        })
                
                # If we have text and no function calls, we're done
                if text_parts and not function_calls:
                    message = AIMessage(content="\n".join(text_parts))
                    return ChatResult(generations=[ChatGeneration(message=message)])
                
                # If we have function calls, either expose them (LangGraph mode)
                # or execute internally (legacy mode).
                if function_calls:
                    if not self.auto_execute_tools:
                        tool_calls = []
                        openai_style_tool_calls = []
                        for idx, fc in enumerate(function_calls):
                            call_id = f"call_{iteration}_{idx}"
                            tool_calls.append(
                                {
                                    "id": call_id,
                                    "type": "tool_call",
                                    "name": fc["name"],
                                    "args": fc["args"],
                                }
                            )
                            openai_style_tool_calls.append(
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": fc["name"],
                                        "arguments": json.dumps(fc["args"]),
                                    },
                                }
                            )
                        message = AIMessage(
                            content="\n".join(text_parts) if text_parts else "",
                            tool_calls=tool_calls,
                            additional_kwargs={"tool_calls": openai_style_tool_calls},
                        )
                        return ChatResult(generations=[ChatGeneration(message=message)])

                    logger.info(
                        "auto_execute_tools=True; executing %d tool call(s) inside wrapper.",
                        len(function_calls),
                    )
                    # Add the model's response to contents using Part.from_dict()
                    model_parts = []
                    for fc in function_calls:
                        model_parts.append(
                            Part.from_dict({
                                "function_call": {
                                    "name": fc["name"],
                                    "args": fc["args"]
                                }
                            })
                        )
                    contents.append(Content(role="model", parts=model_parts))
                    
                    # Execute functions and add responses using Part.from_function_response()
                    function_response_parts = []
                    for fc in function_calls:
                        result = self._execute_function_call(fc["name"], fc["args"])
                        function_response_parts.append(
                            Part.from_function_response(
                                name=fc["name"],
                                response=result
                            )
                        )
                    
                    # Add function responses as user message
                    contents.append(Content(role="user", parts=function_response_parts))
                    
                    iteration += 1
                else:
                    # No text and no function calls
                    break
            
            # If we exhausted iterations
            if iteration >= max_iterations:
                message = AIMessage(content="Maximum tool call iterations reached. Please try rephrasing your request.")
                return ChatResult(generations=[ChatGeneration(message=message)])
            
            # Return empty message if we broke out of loop
            message = AIMessage(content="")
            return ChatResult(generations=[ChatGeneration(message=message)])
            
        except Exception as e:
            logger.error(f"Error in Vertex AI call: {str(e)}")
            raise Exception(f"Error in Vertex AI call: {str(e)}")

    def bind_tools(
        self,
        tools: Union[Sequence[BaseTool], List[Any]],
        **kwargs: Any,
    ) -> "VertexAIChatModel":
        """Bind tools to the model for tool calling.
        
        Creates a new instance with tools bound.
        """
        # Convert tools to Vertex format
        vertex_tools = self._convert_tools_to_vertex_format(list(tools))
        auto_execute_tools = kwargs.get("auto_execute_tools", self.auto_execute_tools)
        
        # Create a new instance with the same config but with tools
        return self.__class__(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            tools=list(tools),
            vertex_tools=vertex_tools,
            response_schema=self.response_schema,
            response_mime_type=self.response_mime_type,
            auto_execute_tools=auto_execute_tools,
        )

    def with_structured_output(
        self,
        schema: Union[Dict[str, Any], type[BaseModel]],
        **kwargs: Any
    ) -> "VertexAIChatModel":
        """Bind a response schema for structured output.
        
        Args:
            schema: Either a Pydantic model or a JSON schema dict
            **kwargs: Additional arguments (e.g., response_mime_type)
        
        Returns:
            A new instance with response schema bound
        """
        # Convert Pydantic model to JSON schema if needed
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = schema.model_json_schema()
            # Clean the schema for Vertex AI
            json_schema = _clean_schema(json_schema)
        else:
            json_schema = _clean_schema(schema)
        
        # Create a new instance with response schema
        return self.__class__(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            tools=self.tools,
            vertex_tools=self.vertex_tools,
            response_schema=json_schema,
            response_mime_type=kwargs.get("response_mime_type", "application/json"),
            auto_execute_tools=kwargs.get("auto_execute_tools", self.auto_execute_tools),
        )

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "vertexai_gemini_chat"


# ============================================================================
# EXAMPLES
# ============================================================================

def example_1_basic_tools():
    """Example 1: Basic tool calling (your original use case)."""
    from langchain.tools import tool
    
    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return f"The weather in {location} is sunny with a temperature of 72°F."
    
    @tool
    def calculate_sum(a: int, b: int) -> int:
        """Calculate the sum of two numbers."""
        return a + b
    
    tools = [get_weather, calculate_sum]
    
    llm = VertexAILLM(temperature=0.7, max_tokens=2000)
    llm_with_tools = llm.bind_tools(tools)
    
    response = llm_with_tools.invoke([
        HumanMessage(content="What's the weather in New York and what is 25 + 17?")
    ])
    
    print("=== Example 1: Basic Tools ===")
    print(f"Response Type: {type(response).__name__}")
    print(f"Content: {response.content}")
    print()


def example_2_structured_output_with_pydantic():
    """Example 2: Structured output using Pydantic model."""
    from pydantic import BaseModel, Field
    
    class WeatherReport(BaseModel):
        """Weather report structure."""
        location: str = Field(description="The location name")
        temperature: float = Field(description="Temperature in Fahrenheit")
        conditions: str = Field(description="Weather conditions")
        forecast: List[str] = Field(description="3-day forecast")
    
    llm = VertexAILLM(temperature=0.7, max_tokens=2000)
    llm_with_schema = llm.with_structured_output(WeatherReport)
    
    response = llm_with_schema.invoke([
        HumanMessage(content="Give me a weather report for San Francisco")
    ])
    
    print("=== Example 2: Structured Output (Pydantic) ===")
    print(f"Response Type: {type(response).__name__}")
    print(f"Content:\n{response.content}")
    
    # Parse the JSON response
    try:
        weather_data = json.loads(response.content)
        print(f"\nParsed Data:")
        print(f"  Location: {weather_data.get('location')}")
        print(f"  Temperature: {weather_data.get('temperature')}°F")
        print(f"  Conditions: {weather_data.get('conditions')}")
    except json.JSONDecodeError:
        print("Response is not valid JSON")
    print()


def example_3_structured_output_with_dict():
    """Example 3: Structured output using dictionary schema."""
    
    schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Person's name"
            },
            "age": {
                "type": "integer",
                "description": "Person's age"
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of skills"
            },
            "experience": {
                "type": "object",
                "properties": {
                    "years": {"type": "integer"},
                    "companies": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        },
        "required": ["name", "age", "skills"]
    }
    
    llm = VertexAILLM(temperature=0.7, max_tokens=2000)
    llm_with_schema = llm.with_structured_output(schema)
    
    response = llm_with_schema.invoke([
        HumanMessage(content="Create a profile for a senior software engineer named John with 10 years experience")
    ])
    
    print("=== Example 3: Structured Output (Dict Schema) ===")
    print(f"Response Type: {type(response).__name__}")
    print(f"Content:\n{response.content}")
    print()


def example_4_tools_and_structured_output():
    """Example 4: Combining tools with structured output."""
    from langchain.tools import tool
    from pydantic import BaseModel, Field
    
    # Define tools
    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return json.dumps({
            "location": location,
            "temperature": 72,
            "conditions": "sunny"
        })
    
    # Define response schema
    class WeatherSummary(BaseModel):
        """Summary of weather information."""
        locations: List[str] = Field(description="List of locations queried")
        average_temp: float = Field(description="Average temperature across locations")
        summary: str = Field(description="Brief summary of weather conditions")
    
    llm = VertexAILLM(temperature=0.7, max_tokens=2000)
    
    # First bind tools
    llm_with_tools = llm.bind_tools([get_weather])
    
    # Then add structured output
    llm_complete = llm_with_tools.with_structured_output(WeatherSummary)
    
    response = llm_complete.invoke([
        HumanMessage(content="Get weather for New York and Los Angeles, then summarize")
    ])
    
    print("=== Example 4: Tools + Structured Output ===")
    print(f"Response Type: {type(response).__name__}")
    print(f"Content:\n{response.content}")
    print()


def example_5_runtime_schema_override():
    """Example 5: Override schema at runtime in kwargs."""
    from pydantic import BaseModel, Field
    
    class ProductInfo(BaseModel):
        """Product information."""
        name: str = Field(description="Product name")
        price: float = Field(description="Product price")
        category: str = Field(description="Product category")
    
    llm = VertexAILLM(temperature=0.7, max_tokens=2000)
    
    # Pass schema at runtime through kwargs
    response = llm.invoke(
        [HumanMessage(content="Tell me about the iPhone 15")],
        response_schema=_clean_schema(ProductInfo.model_json_schema())
    )
    
    print("=== Example 5: Runtime Schema Override ===")
    print(f"Response Type: {type(response).__name__}")
    print(f"Content:\n{response.content}")
    print()


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*80)
    print("VERTEX AI CHAT MODEL - COMPREHENSIVE EXAMPLES")
    print("="*80 + "\n")
    
    example_1_basic_tools()
    example_2_structured_output_with_pydantic()
    example_3_structured_output_with_dict()
    # example_4_tools_and_structured_output()  # Note: combining both might need special handling
    example_5_runtime_schema_override()
    
    print("="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    run_all_examples()

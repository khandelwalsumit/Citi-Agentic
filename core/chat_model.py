"""
LangChain BaseChatModel wrapper for Citi's R2D2 Vertex AI endpoint.

This bridges the gap between vertexai.generative_models.GenerativeModel
and LangChain/LangGraph, which expect a ChatModel with .bind_tools(),
.invoke(), etc.

Usage:
    from core.chat_model import CitiVertexChat
    from core.auth import get_api_key, init_vertex

    # 1. Initialize Vertex AI with your R2D2 credentials (once)
    token = get_api_key(client_id, client_secret, client_scopes)
    init_vertex(token)

    # 2. Create the LangChain-compatible model
    llm = CitiVertexChat(model_name="gemini-2.5-flash")

    # 3. Use with LangGraph agents, chains, etc.
    from langgraph.prebuilt import create_react_agent
    agent = create_react_agent(llm, tools=[...])
"""

from __future__ import annotations

import uuid
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool as VertexTool,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _langchain_tool_to_vertex_fd(tool: dict) -> FunctionDeclaration:
    """Convert an OpenAI-format tool dict to a Vertex FunctionDeclaration."""
    func = tool["function"]
    params = func.get("parameters", {})
    # Vertex AI doesn't accept 'additionalProperties' or 'title' at top level
    params.pop("additionalProperties", None)
    params.pop("title", None)
    return FunctionDeclaration(
        name=func["name"],
        description=func.get("description", ""),
        parameters=params if params.get("properties") else None,
    )


def _convert_tools_to_vertex(tools: Sequence) -> list[VertexTool]:
    """Convert mixed tool formats to Vertex AI Tool objects.

    Accepts:
      - LangChain BaseTool instances
      - @tool-decorated functions
      - Pydantic BaseModel subclasses
      - Raw dicts in OpenAI function-calling format
    """
    from langchain_core.utils.function_calling import convert_to_openai_tool

    declarations = []
    for t in tools:
        if isinstance(t, dict) and "function" in t:
            openai_tool = t
        else:
            openai_tool = convert_to_openai_tool(t)
        declarations.append(_langchain_tool_to_vertex_fd(openai_tool))

    return [VertexTool(function_declarations=declarations)]


def _messages_to_vertex(
    messages: Sequence[BaseMessage],
) -> tuple[list[Content], str | None]:
    """Convert LangChain messages to Vertex AI Content list + system instruction."""
    contents: list[Content] = []
    system_parts: list[str] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append(
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )

        elif isinstance(msg, HumanMessage):
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            contents.append(Content(role="user", parts=[Part.from_text(text)]))

        elif isinstance(msg, AIMessage):
            parts: list[Part] = []
            if msg.content:
                text = msg.content if isinstance(msg.content, str) else str(msg.content)
                if text:
                    parts.append(Part.from_text(text))
            # Tool calls from the model
            for tc in msg.tool_calls or []:
                parts.append(
                    Part.from_function_call(name=tc["name"], args=tc["args"])
                )
            if parts:
                contents.append(Content(role="model", parts=parts))

        elif isinstance(msg, ToolMessage):
            # Tool results go back as function responses
            try:
                response_data = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                response_data = {"result": msg.content}
            contents.append(
                Content(
                    role="user",
                    parts=[
                        Part.from_function_response(
                            name=msg.name or msg.tool_call_id,
                            response=response_data,
                        )
                    ],
                )
            )

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return contents, system_instruction


def _parse_vertex_response(response) -> AIMessage:
    """Convert Vertex AI response to LangChain AIMessage."""
    candidate = response.candidates[0]
    parts = candidate.content.parts

    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for part in parts:
        # Check for function call
        fn_call = getattr(part, "function_call", None)
        if fn_call and fn_call.name:
            tool_calls.append(
                {
                    "name": fn_call.name,
                    "args": dict(fn_call.args) if fn_call.args else {},
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                }
            )
        # Check for text
        elif hasattr(part, "text") and part.text:
            text_parts.append(part.text)

    content = "\n".join(text_parts) if text_parts else ""

    return AIMessage(
        content=content,
        tool_calls=tool_calls,
    )


# ---------------------------------------------------------------------------
# The ChatModel wrapper
# ---------------------------------------------------------------------------

class CitiVertexChat(BaseChatModel):
    """LangChain ChatModel backed by Citi's R2D2 Vertex AI endpoint.

    Requires vertexai.init() to have been called first (see core.auth).

    Example::

        from core.auth import get_api_key, init_vertex
        from core.chat_model import CitiVertexChat

        init_vertex(token=get_api_key(cid, csec, cscopes))
        llm = CitiVertexChat(model_name="gemini-2.5-flash")

        # Basic usage
        llm.invoke("What is 2+2?")

        # With tools (for agents)
        llm_with_tools = llm.bind_tools([my_tool])
        llm_with_tools.invoke("Search for quarterly earnings")
    """

    model_name: str = Field(default="gemini-2.5-flash")
    temperature: float = Field(default=0.0)
    max_output_tokens: int = Field(default=8192)
    top_p: float = Field(default=0.95)
    top_k: int = Field(default=40)

    # Private — not serialized
    _model: GenerativeModel = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # GenerativeModel is lightweight — no network call here.
        # vertexai.init() must already have been called.
        self._model = GenerativeModel(self.model_name)

    # -- LangChain interface -------------------------------------------------

    @property
    def _llm_type(self) -> str:
        return "citi-vertex-ai"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }

    def bind_tools(
        self,
        tools: Sequence,
        *,
        tool_choice: Optional[str] = None,
        **kwargs,
    ):
        """Bind tools so the model can call them.

        Accepts LangChain tools, @tool functions, Pydantic models, or dicts.
        Returns a new Runnable with tools attached.
        """
        vertex_tools = _convert_tools_to_vertex(tools)
        return self.bind(tools=vertex_tools, **kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        """Core generation — called by .invoke(), .stream(), agents, etc."""
        contents, system_instruction = _messages_to_vertex(messages)

        # Build generation config
        gen_config = GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            stop_sequences=stop or [],
        )

        # If there's a system instruction, create a model instance with it.
        # GenerativeModel is lightweight so this is fine per-call.
        if system_instruction:
            model = GenerativeModel(
                self.model_name,
                system_instruction=system_instruction,
            )
        else:
            model = self._model

        # Tools passed via bind_tools()
        vertex_tools = kwargs.get("tools", None)

        try:
            response = model.generate_content(
                contents,
                generation_config=gen_config,
                tools=vertex_tools,
            )
        except Exception as e:
            logger.error("Vertex AI generate_content failed: %s", e)
            raise

        ai_message = _parse_vertex_response(response)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

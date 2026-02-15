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
from typing import Any, Dict, Iterator, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
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


def _langchain_tool_to_vertex_fd(tool: dict) -> FunctionDeclaration:
    """Convert an OpenAI-format tool dict to a Vertex FunctionDeclaration."""
    func = tool["function"]
    params = func.get("parameters", {})
    params = _clean_schema(params)
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
    """Convert LangChain messages to Vertex AI Content list + system instruction.

    System messages are extracted separately. They will be injected as the
    first user message prefixed with "[System Instructions]" for maximum
    compatibility with R2D2 proxy endpoints that may not support the
    system_instruction model parameter.
    """
    contents: list[Content] = []
    system_parts: list[str] = []

    for msg in messages:
        # Handle tuples like ("user", "hello") that LangGraph may pass
        if isinstance(msg, (tuple, list)) and len(msg) == 2:
            role, text = msg
            text = str(text)
            if role in ("system",):
                system_parts.append(text)
            elif role in ("assistant", "ai", "model"):
                contents.append(Content(role="model", parts=[Part.from_text(text)]))
            else:
                contents.append(Content(role="user", parts=[Part.from_text(text)]))
            continue

        if isinstance(msg, str):
            contents.append(Content(role="user", parts=[Part.from_text(msg)]))
            continue

        if isinstance(msg, SystemMessage):
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            system_parts.append(text)

        elif isinstance(msg, HumanMessage):
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            contents.append(Content(role="user", parts=[Part.from_text(text)]))

        elif isinstance(msg, AIMessage):
            parts: list[Part] = []
            if msg.content:
                text = msg.content if isinstance(msg.content, str) else str(msg.content)
                if text.strip():
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
                if not isinstance(response_data, dict):
                    response_data = {"result": msg.content}
            except (json.JSONDecodeError, TypeError):
                response_data = {"result": msg.content}

            name = msg.name or getattr(msg, "tool_call_id", None) or "unknown_tool"
            contents.append(
                Content(
                    role="user",
                    parts=[
                        Part.from_function_response(
                            name=name,
                            response=response_data,
                        )
                    ],
                )
            )

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return contents, system_instruction


def _parse_vertex_response(response) -> AIMessage:
    """Convert Vertex AI response to LangChain AIMessage.

    Handles empty candidates, safety blocks, and malformed responses.
    """
    # ---- Handle missing or empty candidates ----
    if not response.candidates:
        # Check if blocked by safety filter
        block_reason = ""
        if hasattr(response, "prompt_feedback"):
            pf = response.prompt_feedback
            if hasattr(pf, "block_reason") and pf.block_reason:
                block_reason = f" (block_reason: {pf.block_reason})"
            if hasattr(pf, "block_reason_message") and pf.block_reason_message:
                block_reason += f" {pf.block_reason_message}"

        logger.warning("Vertex AI returned no candidates.%s", block_reason)

        # Return a fallback AIMessage so LangGraph doesn't crash
        return AIMessage(
            content=f"[Model returned no response{block_reason}. "
                    "The request may have been blocked by safety filters "
                    "or the prompt may need adjustment.]",
        )

    candidate = response.candidates[0]

    # Check finish reason for issues
    finish_reason = getattr(candidate, "finish_reason", None)
    if finish_reason and str(finish_reason) not in ("1", "STOP", "FinishReason.STOP"):
        logger.info("Vertex AI finish_reason: %s", finish_reason)

    # Handle candidate with no content
    if not hasattr(candidate, "content") or not candidate.content:
        return AIMessage(content="[Model returned empty content]")

    if not candidate.content.parts:
        return AIMessage(content="[Model returned no content parts]")

    # ---- Parse parts ----
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for part in candidate.content.parts:
        # Check for function call
        fn_call = getattr(part, "function_call", None)
        if fn_call and getattr(fn_call, "name", None):
            args = {}
            if fn_call.args:
                # fn_call.args can be a proto MapComposite or dict
                try:
                    args = dict(fn_call.args)
                except Exception:
                    args = json.loads(type(fn_call.args).to_json(fn_call.args))
            tool_calls.append({
                "name": fn_call.name,
                "args": args,
                "id": f"call_{uuid.uuid4().hex[:12]}",
            })
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

        # ---- Inject system prompt as first user message ----
        # R2D2 proxy may not support system_instruction on GenerativeModel,
        # so we prepend it as a clearly-labelled user message instead.
        if system_instruction:
            sys_content = Content(
                role="user",
                parts=[Part.from_text(
                    f"[SYSTEM INSTRUCTIONS — follow these at all times]\n\n"
                    f"{system_instruction}"
                )],
            )
            # Need a model ack to keep user/model turn alternation valid
            ack_content = Content(
                role="model",
                parts=[Part.from_text("Understood. I will follow these instructions.")],
            )
            contents = [sys_content, ack_content] + contents

        # ---- Guard: must have at least one content ----
        if not contents:
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="[No input messages provided]")
            )])

        # ---- Build generation config ----
        gen_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        if stop:
            gen_kwargs["stop_sequences"] = stop

        gen_config = GenerationConfig(**gen_kwargs)

        # ---- Build generate_content kwargs ----
        call_kwargs: dict[str, Any] = {
            "generation_config": gen_config,
        }

        # Only pass tools if we actually have them
        vertex_tools = kwargs.get("tools")
        if vertex_tools:
            call_kwargs["tools"] = vertex_tools

        # ---- Call Vertex AI ----
        try:
            response = self._model.generate_content(contents, **call_kwargs)
        except Exception as e:
            logger.error("Vertex AI generate_content failed: %s", e)
            raise

        ai_message = _parse_vertex_response(response)
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

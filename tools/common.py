"""
Common tools available to agents.

Add your own tools here using the @tool decorator.
Register them in TOOL_REGISTRY so agents can reference them by name
in their skill.md files.

Example:
    @tool
    def my_tool(query: str) -> str:
        '''Description of what this tool does.'''
        return "result"

    TOOL_REGISTRY["my_tool"] = my_tool
"""

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Example tools — replace with your actual Citi tools
# ---------------------------------------------------------------------------

@tool
def search_documents(query: str) -> str:
    """Search internal documents and knowledge bases for relevant information.

    Args:
        query: The search query describing what information to find.
    """
    # TODO: Replace with actual Citi document search (e.g., Elasticsearch, SharePoint API)
    return f"[Document search results for: {query}] — Replace this stub with your actual search implementation."


@tool
def query_database(sql_description: str) -> str:
    """Query internal databases based on a natural language description.

    Args:
        sql_description: Natural language description of the data needed.
    """
    # TODO: Replace with actual BigQuery or internal DB query
    return f"[Database query results for: {sql_description}] — Replace this stub with your actual DB query."


@tool
def send_notification(recipient: str, message: str) -> str:
    """Send a notification or message to a team member.

    Args:
        recipient: The recipient's name or ID.
        message: The notification content.
    """
    # TODO: Replace with actual notification system (Slack, email, Symphony)
    return f"Notification sent to {recipient}: {message}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A mathematical expression like '2 + 2' or '(100 * 0.05) / 12'.
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: Expression contains invalid characters."
    try:
        result = eval(expression)  # Safe — only digits and math operators
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def read_file(file_path: str) -> str:
    """Read contents of a local file.

    Args:
        file_path: Path to the file to read.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 10000:
            return content[:10000] + "\n... [truncated]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


# ---------------------------------------------------------------------------
# Tool registry — agents reference tools by these keys in their skill.md
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict = {
    "search_documents": search_documents,
    "query_database": query_database,
    "send_notification": send_notification,
    "calculate": calculate,
    "read_file": read_file,
}

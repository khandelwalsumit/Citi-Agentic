chek below code 

```python

from google.protobuf.struct_pb2 import Struct

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Part construction helpers (SDK-version-safe)
# ---------------------------------------------------------------------------

def _make_func_call_part(name: str, args: dict) -> Part:
    """Create a Part containing a function_call.

    Works across all SDK versions by falling back to raw protobuf
    if Part.from_function_call doesn't exist.
    """
    if hasattr(Part, "from_function_call"):
        return Part.from_function_call(name=name, args=args)

    # Build from raw gapic protobuf types
    from google.cloud.aiplatform_v1.types import content as gapic

    s = Struct()
    s.update(args or {})
    raw_part = gapic.Part(function_call=gapic.FunctionCall(name=name, args=s))
    return Part._from_gapic(raw_part)


def _make_func_response_part(name: str, response: dict) -> Part:
    """Create a Part containing a function_response.

    Works across all SDK versions by falling back to raw protobuf.
    """
    if hasattr(Part, "from_function_response"):
        return Part.from_function_response(name=name, response=response)

    from google.cloud.aiplatform_v1.types import content as gapic

    s = Struct()
    s.update(response or {})
    raw_part = gapic.Part(
        function_response=gapic.FunctionResponse(name=name, response=s)
    )
    return Part._from_gapic(raw_part)

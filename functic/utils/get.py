from typing import Any, Dict, Text


def get_safe_value(obj: Dict, *keys: Text, default: Any = None) -> Any:
    """Safely get nested dictionary values"""

    try:
        result = obj
        for key in keys:
            result = result[key]
        return result
    except (KeyError, TypeError, IndexError):
        return default

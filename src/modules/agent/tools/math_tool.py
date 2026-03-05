"""Math tool — deterministic mathematical calculations."""

import math

from langchain_core.tools import tool


@tool
def calculate_cagr(start_value: float, end_value: float, years: float) -> str:
    """Calculate the Compound Annual Growth Rate (CAGR).

    Formula: CAGR = (end_value / start_value)^(1/years) - 1

    Use this tool when calculating growth rates between two time periods.

    Args:
        start_value: The beginning value.
        end_value: The ending value.
        years: The number of years between the two values.

    Returns:
        The CAGR as a percentage string with explanation.
    """
    if start_value <= 0:
        return "Error: start_value must be positive."
    if end_value <= 0:
        return "Error: end_value must be positive."
    if years <= 0:
        return "Error: years must be positive."

    cagr = (end_value / start_value) ** (1 / years) - 1
    percentage = cagr * 100

    return (
        f"CAGR = ({end_value} / {start_value})^(1/{years}) - 1 = {cagr:.6f} = {percentage:.2f}%\n"
        f"This means an average annual growth rate of {percentage:.2f}% over {years} years."
    )


@tool
def calculate_percentage(value: float, total: float) -> str:
    """Calculate what percentage a value is of a total.

    Formula: percentage = (value / total) * 100

    Args:
        value: The part value.
        total: The total/whole value.

    Returns:
        The percentage with explanation.
    """
    if total == 0:
        return "Error: total cannot be zero."

    percentage = (value / total) * 100
    return f"{value} is {percentage:.2f}% of {total}"


@tool
def calculate_percentage_change(old_value: float, new_value: float) -> str:
    """Calculate the percentage change between two values.

    Formula: change = ((new_value - old_value) / old_value) * 100

    Args:
        old_value: The original/baseline value.
        new_value: The new/current value.

    Returns:
        The percentage change with explanation.
    """
    if old_value == 0:
        return "Error: old_value cannot be zero."

    change = ((new_value - old_value) / old_value) * 100
    direction = "increase" if change > 0 else "decrease"
    return (
        f"Change from {old_value} to {new_value}: {change:+.2f}% ({direction})\n"
        f"Absolute change: {new_value - old_value:+.2f}"
    )


@tool
def calculate_arithmetic(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports: +, -, *, /, **, (), and common math functions.
    Use this for basic arithmetic operations.

    Args:
        expression: A mathematical expression to evaluate (e.g., '(44 + 12 + 16 + 28) / 4').

    Returns:
        The result of the calculation.
    """
    # Only allow safe characters
    allowed_chars = set("0123456789.+-*/() ,")
    safe_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "ceil": math.ceil,
        "floor": math.floor,
    }

    try:
        # Basic safety check
        cleaned = expression.strip()
        for char in cleaned:
            if char not in allowed_chars and not char.isalpha():
                return f"Error: Unsupported character '{char}' in expression."

        result = eval(cleaned, {"__builtins__": {}}, safe_names)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"

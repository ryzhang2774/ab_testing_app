"""Business-friendly ordering for selected customer segments."""
import re


def ordered_groups(values, grouping):
    """Sort FICO ranges by their numeric boundaries; preserve other data order."""
    labels = list(dict.fromkeys(values))
    if 'fico' not in str(grouping).casefold():
        return labels
    def key(label):
        name = str(label).strip()
        numbers = re.findall(r'\d+(?:\.\d+)?', name)
        if not numbers:
            return (1, 0, name.casefold())
        return (0, float(numbers[0]), name.casefold())
    return sorted(labels, key=key)


def credit_line_amount(label):
    match = re.fullmatch(r'(?:credit[ _-]*line\s*)?\$?\s*(\d[\d,]*(?:\.\d+)?)', str(label).strip(), re.I)
    return float(match[1].replace(',', '')) if match else None


def historical_strategy_order(values):
    """Sort monetary strategies numerically. Preserve input order for other strategies."""
    options = list(dict.fromkeys(values))
    amounts = {option: credit_line_amount(option) for option in options}
    standalone_controls = {'control', 'bau', 'current'}
    if all(value is not None or str(option).strip().casefold() in standalone_controls
           for option, value in amounts.items()):
        return sorted(options, key=lambda option: -1 if amounts[option] is None else amounts[option])
    return options

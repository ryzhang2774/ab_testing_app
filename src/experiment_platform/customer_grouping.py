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

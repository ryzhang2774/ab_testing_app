"""Stable, distinguishable option colors shared by planning and analysis charts."""
ROLE_COLORS = {'Control': '#006ba2', 'Test': '#008679', 'Historical only': '#a0b4c0'}
STRATEGY_COLORS = ['#006ba2', '#008679', '#cc7c0e', '#795bb1', '#3299c0', '#cb5863',
                   '#687ca3', '#a76e27', '#6e9e36', '#ba5899', '#407c80', '#c07b67',
                   '#3f679a', '#c0992c', '#688eaa', '#987357', '#9e6d98', '#69996c',
                   '#907b3f', '#b26952', '#6d719b', '#5195a3', '#a25b71', '#8a9560']


def strategy_styles(options):
    patterns = ['', '/', '\\', 'x', '.', '+']
    return ({option: STRATEGY_COLORS[i % len(STRATEGY_COLORS)] for i, option in enumerate(options)},
            {option: patterns[(i // len(STRATEGY_COLORS)) % len(patterns)] for i, option in enumerate(options)})

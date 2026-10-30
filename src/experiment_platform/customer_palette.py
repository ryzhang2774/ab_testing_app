"""Stable, distinguishable option colors shared by planning and analysis charts."""
STRATEGY_COLORS = ['#005b88', '#008fc5', '#f18f01', '#8064a2', '#00a58e', '#d35f70',
                   '#687ca3', '#a76e27', '#6e9e36', '#ba5899', '#407c80', '#c07b67',
                   '#3f679a', '#c0992c', '#688eaa', '#987357', '#9e6d98', '#69996c',
                   '#907b3f', '#b26952', '#6d719b', '#5195a3', '#a25b71', '#8a9560']


def strategy_styles(options):
    patterns = ['', '/', '\\', 'x', '.', '+']
    return ({option: STRATEGY_COLORS[i % len(STRATEGY_COLORS)] for i, option in enumerate(options)},
            {option: patterns[(i // len(STRATEGY_COLORS)) % len(patterns)] for i, option in enumerate(options)})

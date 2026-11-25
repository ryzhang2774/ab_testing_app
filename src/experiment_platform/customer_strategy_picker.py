"""Offline Plotly interaction bridge for customer strategy selection."""
from __future__ import annotations

from functools import lru_cache, partial
import atexit
from pathlib import Path
import shutil
import tempfile

import streamlit as st
from streamlit.components.v1 import declare_component
from plotly.offline import get_plotlyjs

from .customer_grouping import historical_strategy_order


@lru_cache(maxsize=1)
def _component_assets():
    # Serve the already-installed Plotly bundle locally. Copy once per process;
    # browsers cache the script instead of receiving megabytes on every edit.
    target = Path(tempfile.mkdtemp(prefix='experiment-strategy-picker-'))
    atexit.register(shutil.rmtree, target, ignore_errors=True)
    source = Path(__file__).with_name('strategy_picker')
    for name in ('index.html', 'picker.js', 'picker.css'):
        shutil.copyfile(source / name, target / name)
    (target / 'plotly.min.js').write_text(get_plotlyjs(), encoding='utf-8')
    return str(target)


def apply_strategy_event(current, event, control, allowed, revision):
    """Validate browser intent; never accept arbitrary strategy lists or x data."""
    original = list(current)
    if not isinstance(event, dict) or event.get('revision') != revision:
        return original, 'Chart updated. Try the selection again.'
    allowed = set(allowed) - {control}
    action, target = event.get('action'), event.get('strategy')
    if not isinstance(target, str) or target not in allowed:
        return original, 'Choose an available Test credit line.'
    if action == 'toggle':
        result = [value for value in original if value != target] if target in original else original + [target]
        message = ('Removed Test: ' if target in original else 'Added Test: ') + target
    elif action == 'move':
        previous = event.get('previous')
        if previous == control or previous not in original:
            return original, 'That Test is no longer selected.'
        result = [value for value in original if value != previous] + [target]
        message = 'Moved Test: ' + previous + ' → ' + target
    else:
        return original, 'Choose a Test point or drag a selected Test line.'
    return historical_strategy_order(dict.fromkeys(result)), message


def _receive_event(component_key, tests_key, control, allowed, revision):
    event = st.session_state.get(component_key)
    if not isinstance(event, dict) or not isinstance(event.get('id'), str) or len(event['id']) > 120:
        return
    ack_key = component_key + '_ack'
    if st.session_state.get(ack_key) == event['id']:
        return
    tests, message = apply_strategy_event(st.session_state.get(tests_key, []), event, control, allowed, revision)
    # Component callbacks run before the next script pass, so the multiselect
    # receives this value before it is instantiated again.
    st.session_state[tests_key] = tests
    st.session_state[ack_key] = event['id']
    st.session_state[component_key + '_message'] = message


def render_strategy_picker(fig, trend, control, tests, allowed, *, key, tests_key, revision):
    try:
        component = declare_component('customer_strategy_picker', path=_component_assets())
    except OSError:
        st.plotly_chart(fig, use_container_width=True, key=key + '_fallback')
        st.caption('Chart editing is unavailable in this environment. Use the Test options on the right.')
        return None
    amounts = dict(zip(trend['labels'], trend['x']))
    choices = [dict(label=label, value=amounts[label]) for label in allowed if label in amounts and label != control]
    draggable = {}
    for i, shape in enumerate(fig.layout.shapes or ()):
        if shape.type == 'line' and shape.line.dash == 'dash':
            label = next((label for label in tests if amounts.get(label) == shape.x0), None)
            if label is not None:
                shape.editable = True
                shape.layer = 'above'
                draggable[str(i)] = label
        else:
            shape.editable = False
    # Never allow dragging axes, region rectangles, the legend or Control.
    fig.update_layout(dragmode=False, clickmode='event', hovermode='closest', showlegend=False,
                      margin=dict(t=64, b=40, l=48, r=16))
    callback = partial(_receive_event, key, tests_key, control, list(allowed), revision)
    return component(figure=fig.to_json(), choices=choices, selected=list(tests),
                     control=control, control_value=amounts.get(control), draggable=draggable,
                     revision=revision, ack=st.session_state.get(key + '_ack'),
                     message=st.session_state.get(key + '_message', ''),
                     key=key, default=None, on_change=callback)

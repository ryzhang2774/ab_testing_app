import pytest

from src.experiment_platform.customer_strategy_picker import apply_strategy_event, _receive_event


def event(action='toggle', strategy='2000', previous=None, revision='current'):
    return dict(id='event-1', revision=revision, action=action, strategy=strategy, previous=previous)


def test_click_toggles_only_available_noncontrol_strategies():
    selected, _ = apply_strategy_event(['1000'], event(), '500', ['1000','2000','3000'], 'current')
    assert selected == ['1000','2000']
    selected, _ = apply_strategy_event(selected, event(), '500', ['1000','2000','3000'], 'current')
    assert selected == ['1000']
    assert apply_strategy_event(['2000'], event(), '500', ['2000'], 'current')[0] == []
    for invalid in ['500','9999',None,{},['2000']]:
        assert apply_strategy_event(['1000'], event(strategy=invalid), '500', ['500','1000','2000'], 'current')[0] == ['1000']


def test_drag_moves_one_test_and_merges_duplicates_without_changing_control():
    selected, _ = apply_strategy_event(['1000','2000'], event('move','3000','1000'), '500', ['1000','2000','3000'], 'current')
    assert selected == ['2000','3000']
    selected, _ = apply_strategy_event(selected, event('move','2000','3000'), '500', ['1000','2000','3000'], 'current')
    assert selected == ['2000']
    assert apply_strategy_event(selected, event('move','1000','500'), '500', ['1000','2000'], 'current')[0] == selected
    assert apply_strategy_event(selected, event('move','1000','9999'), '500', ['1000','2000'], 'current')[0] == selected


@pytest.mark.parametrize('bad', [None, [], '2000', {'action':'toggle'}, event(revision='old-data'), event(action='delete')])
def test_stale_or_malformed_events_do_not_mutate_assignments(bad):
    assert apply_strategy_event(['1000'], bad, '500', ['1000','2000'], 'current')[0] == ['1000']


def test_component_callback_updates_manual_widget_once_per_event(monkeypatch):
    from src.experiment_platform import customer_strategy_picker
    state = {'picker':event(), 'tests':['1000']}
    monkeypatch.setattr(customer_strategy_picker.st, 'session_state', state)
    _receive_event('picker', 'tests', '500', ['1000','2000'], 'current')
    assert state['tests'] == ['1000','2000']
    assert state['picker_ack'] == 'event-1'
    _receive_event('picker', 'tests', '500', ['1000','2000'], 'current')
    assert state['tests'] == ['1000','2000']

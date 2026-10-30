from streamlit.testing.v1 import AppTest
from src.experiment_platform import portfolio


def test_chooser_monitor_analyzer_navigation(monkeypatch):
    monkeypatch.setattr(portfolio, 'load_experiments', lambda: [portfolio.normalize_experiment(row) for row in portfolio.seed_experiments()])
    app = AppTest.from_file('app.py', default_timeout=60).run()
    assert not app.exception
    assert app.button(key='home_choose_customer')
    assert not any('Your experiments' in item.value for item in app.markdown)
    app.button(key='nav_monitor').click().run()
    assert not app.exception
    content=' '.join(item.value for item in app.markdown)
    assert 'Collecting samples' in content and 'Collecting outcomes' in content
    assert 'Projected portfolio value' not in content
    app.button(key='nav_analyzer').click().run()
    assert not app.exception
    app.button(key='open_selected_analysis').click().run()
    assert not app.exception
    assert app.session_state.customer_intent == 'analyze'
    assert any(item.label == 'Data source' for item in app.radio)


def test_customer_entry_explains_randomized_flow_and_keeps_workflows():
    app = AppTest.from_file('app.py', default_timeout=90).run()
    app.button(key='home_choose_customer').click().run()
    assert not app.exception
    intro = next(item.value for item in app.markdown if '<section class="customer-explainer"' in item.value)
    for phrase in ['Choose your customers', 'Assign at random', 'Control', 'Test(s)', 'Compare outcomes']:
        assert phrase in intro
    assert app.button(key='customer_demo_plan') and app.button(key='customer_demo_analyze')
    app.button(key='customer_demo_plan').click().run(timeout=90)
    assert not app.exception and app.session_state.customer_intent == 'plan'
    app.button(key='customer_switch_workflow').click().run()
    app.button(key='customer_demo_analyze').click().run(timeout=90)
    assert not app.exception and app.session_state.customer_intent == 'analyze'
    assert app.radio(key='simple_source')


def test_geo_and_time_landing_explain_workflows_and_keep_navigation():
    for chooser, plan, analyze, switch, intent, title, phrases in [
        ('home_choose_geo', 'geo_demo_plan', 'geo_demo_analyze', 'geo_switch_workflow', 'geo_intent',
         'How does a geographic experiment work?', ['Choose your markets', 'Identify Test and Control', 'Track before and after', 'Compare the changes']),
        ('home_choose_timeseries', 'ts_demo_plan', 'ts_demo_analyze', 'ts_switch_workflow', 'ts_intent',
         'How does a time series experiment work?', ['Choose an outcome', 'Study the history', 'Launch to everyone', 'Assess the change']),
    ]:
        app = AppTest.from_file('app.py', default_timeout=90).run()
        app.button(key=chooser).click().run(timeout=90)
        assert not app.exception
        explainer = next(item.value for item in app.markdown if title in item.value and '<section' in item.value)
        assert all(phrase in explainer for phrase in phrases)
        assert app.button(key=plan) and app.button(key=analyze)
        app.button(key=plan).click().run(timeout=90)
        assert not app.exception and app.session_state[intent] == 'plan'
        app.button(key=switch).click().run(timeout=90)
        app.button(key=analyze).click().run(timeout=90)
        assert not app.exception and app.session_state[intent] == 'analyze'

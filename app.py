"""Streamlit dashboard for the Aquatics Operations Assistant MVP."""
from datetime import date
import streamlit as st
import pandas as pd
from engine import load_rows, analyze, expiring_certifications, report, answer_query

try:
    from predictor import train_and_evaluate, predict_shift_risk, features_from_shift_and_staff
    _PREDICTOR_AVAILABLE = True
except ImportError:
    _PREDICTOR_AVAILABLE = False

st.set_page_config(page_title='Aquatics Operations Assistant', page_icon='🏊', layout='wide')
st.title('🏊 Aquatics Operations Assistant')
st.caption('IBM Agentic AI project MVP · Fictional demonstration data · Human-reviewed recommendations')
st.info('Prototype uses explainable, rule-based decision logic. It does not connect to IBM watsonx Orchestrate or live workplace systems.')

staff = load_rows('staff.csv')
shifts = load_rows('uncovered_shifts.csv')
assignments = load_rows('assignments.csv')

with st.sidebar:
    st.header('Scenario settings')
    selection = st.selectbox('Uncovered shift', list(range(len(shifts))), format_func=lambda i: f"{shifts[i]['shift_id']} · {shifts[i]['date']} · {shifts[i]['start']}–{shifts[i]['end']}")
    analysis_date = st.date_input('Review as of', value=date.today())
    st.caption('To try the no-candidate scenario, select SHIFT-002.')

shift = shifts[selection]
result = analyze(shift, staff, assignments, analysis_date)
expiring = expiring_certifications(staff, analysis_date)

# ── Section 0 · Natural-language query panel ────────────────────────────────
st.subheader('0 · Ask a staffing question')
st.caption(
    'Type a question in plain English. The assistant uses keyword-based intent classification '
    'to route your question to the relevant staffing data. No external AI service is used.'
)
query_text = st.text_input(
    'Your question',
    placeholder='e.g. "Who can cover the lifeguard shift?" or "Are any certifications expiring?"',
)
if query_text.strip():
    qr = answer_query(query_text, staff, shifts, assignments, analysis_date)
    if qr['supported']:
        st.info(f"**Intent recognised:** {qr['intent'].replace('_', ' ').title()}")
        st.success(qr['answer'])
    else:
        st.warning(qr['answer'])
    if qr['detail'] and qr['intent'] in ('coverage', 'staff_info'):
        with st.expander('Detailed candidate data for this query'):
            detail_rows = []
            for c in qr['detail']:
                if 'eligible' in c:
                    detail_rows.append({
                        'Staff member': c['name'],
                        'Eligible': 'Yes' if c['eligible'] else 'No',
                        'Explanation': 'Passed all checks' if c['eligible'] else ', '.join(c['reasons']),
                    })
                else:
                    detail_rows.append({k: v for k, v in c.items()})
            if detail_rows:
                st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
st.divider()

# ── Summary metrics ──────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric('Uncovered shifts (sample)', len(shifts))
c2.metric('Eligible for selected shift', len(result['qualified']))
c3.metric('Certification alerts', len(expiring))

st.subheader('1 · Coverage recommendation')
st.write(f"**Shift:** {shift['shift_id']} · {shift['date']} · {shift['start']}–{shift['end']} · {shift['role']}")
if result['recommended']:
    recommended = result['recommended']
    candidate_count = result['candidate_count']
    hours_remaining = result['hours_remaining']
    runner_up = result['runner_up']
    st.success(
        f"**Suggested: {recommended['name']}** · "
        f"{recommended['hours_this_week']} hrs already scheduled this week · "
        f"{hours_remaining:.1f} hrs remaining under {int(recommended['hours_this_week'] + hours_remaining)}-hr weekly limit"
    )
    reasoning_parts = [
        f"Selected from {candidate_count} eligible candidate(s).",
        "Chosen by fewest current weekly hours (lowest workload heuristic).",
    ]
    if runner_up:
        reasoning_parts.append(
            f"Next eligible candidate: {runner_up['name']} ({runner_up['hours_this_week']} hrs this week)."
        )
    st.write('**Reasoning:** ' + ' '.join(reasoning_parts))
    approve = st.checkbox('I am the supervisor and have reviewed this suggested coverage (demo only)')
    if approve:
        st.warning('Demo acknowledgment recorded only in this screen. No actual scheduling update or message was sent.')
else:
    st.error('No qualified candidate found. Escalate to Aquatics Director; do not fill with an ineligible employee.')
st.caption('Real facilities must validate current certification, authorization, coverage ratios, availability and labor rules before assignments.')

st.subheader('2 · Explainable staff checks')
rows = []
for c in result['checks']:
    rows.append({'Staff member': c['name'], 'Eligible': 'Yes' if c['eligible'] else 'No', 'Hours this week': c['hours_this_week'], 'Certification expiration': c['cert_expires'], 'Explanation': 'Passed all example checks' if c['eligible'] else ', '.join(c['reasons'])})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.subheader('3 · Certification alerts')
if expiring:
    st.dataframe(pd.DataFrame(expiring), hide_index=True, use_container_width=True)
else:
    st.write('No certifications expiring within 30 days of the selected review date.')

with st.expander('4 · Decision workflow / audit trail', expanded=True):
    for step in result['audit']:
        st.write(step)

text_report = report(result, expiring)
st.download_button('Download operations report (.md)', text_report, file_name=f"aquatics_report_{shift['shift_id']}.md", mime='text/markdown')

# ── Section 5 · Experimental ML prediction ───────────────────────────────────
with st.expander('5 · Experimental: Staffing shortage prediction (ML)', expanded=False):
    st.warning(
        '⚠️ **Experimental feature — do not use for real staffing decisions.** '
        'This model was trained on 200 rows of synthetic data generated for demonstration purposes. '
        'Test-set performance does not establish real-world predictive accuracy. '
        'The rule-based eligibility engine in Section 1 remains the authoritative source for all recommendations.'
    )
    if not _PREDICTOR_AVAILABLE:
        st.error('scikit-learn is not installed. Run `python -m pip install scikit-learn` to enable this section.')
    else:
        # Train once per session and cache the result
        if 'ml_trained' not in st.session_state:
            st.session_state['ml_trained'] = train_and_evaluate()
        trained = st.session_state['ml_trained']

        # Derive pre-shift features from the currently selected shift + live staff data
        feats = features_from_shift_and_staff(shift, staff)
        pred  = predict_shift_risk(trained, **feats)

        # Risk badge
        risk_colors = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
        badge = risk_colors.get(pred['risk_label'], '⚪')
        st.subheader(f"{badge} Predicted shortage risk: **{pred['risk_label']}**")
        st.write(
            f"Risk score: **{pred['probability']:.3f}** "
            f"(0 = very low risk, 1 = very high risk)  \n"
            f"Shift: {shift['shift_id']} · {shift['role']} · "
            f"{shift['date']} {shift['start']}–{shift['end']}"
        )

        # Pre-shift features used
        with st.expander('Pre-shift features used by the model'):
            feat_rows = [
                {'Feature': k.replace('_', ' ').title(), 'Value': v}
                for k, v in pred['features_used'].items()
            ]
            st.dataframe(pd.DataFrame(feat_rows), hide_index=True, use_container_width=True)
            st.caption(
                'These counts are computed from current staff records before the shift starts. '
                'The model never sees the actual eligibility outcome (which would be data leakage).'
            )

        st.divider()

        # Model evaluation on held-out test set
        st.subheader('Model evaluation (held-out test set — last 40 of 200 synthetic rows)')
        m = trained['metrics']
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric('Precision', f"{m['precision']:.3f}")
        mc2.metric('Recall',    f"{m['recall']:.3f}")
        mc3.metric('F1',        f"{m['f1']:.3f}")
        mc4.metric('Accuracy',  f"{m['accuracy']:.3f}")

        st.caption(
            f"Train: {trained['train_size']} rows (chronologically first) · "
            f"Test: {trained['test_size']} rows (chronologically last). "
            'Both risk classes present in test set: '
            + ('Yes ✓' if trained['both_classes_in_test'] else 'No ✗')
        )

        # Confusion matrix
        st.write('**Confusion matrix (test set)**')
        cm = trained['confusion_matrix']
        cm_df = pd.DataFrame(
            cm,
            index=['Actual: Low risk (0)', 'Actual: At risk (1)'],
            columns=['Predicted: Low risk (0)', 'Predicted: At risk (1)'],
        )
        st.dataframe(cm_df, use_container_width=True)
        st.caption(
            'Logistic Regression trained on 200 rows of synthetic data (seed=42). '
            'Chronological split preserves temporal integrity. '
            'This performance is on fictitious data only and does not generalise to real facilities.'
        )

with st.expander('Sample data (fictional)'):
    st.dataframe(pd.DataFrame(staff), hide_index=True, use_container_width=True)
    st.dataframe(pd.DataFrame(assignments), hide_index=True, use_container_width=True)

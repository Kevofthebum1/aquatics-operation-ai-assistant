# Aquatics Operations Assistant — IBM Agentic AI Project MVP (with Experimental ML)

A **rule-based, agent-style prototype** built for a student demonstration of IBM Agentic AI concepts.
It is not an IBM watsonx Orchestrate deployment or an LLM-powered autonomous agent.
All personnel data are fictional.

## Running on Windows / VS Code terminal

1. Install Python 3.10 or later from https://www.python.org/downloads/
   (check **"Add Python to PATH"** during installation).
2. Open a terminal inside this folder.
3. Run:

   ```bash
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

   `requirements.txt` now includes `scikit-learn` for the experimental prediction section.
   If you want to run without it, the dashboard still works — Section 5 will show a graceful error.

4. A browser window opens automatically at http://localhost:8501. Press Ctrl+C to stop.
   If the browser does not open, navigate to http://localhost:8501 manually.

For macOS/Linux, replace `python` with `python3`.

## Running tests

```bash
python -m unittest -v test_engine.py
```

Expected result: **43 tests, 0 failures** across five test classes.

## Suggested demo (about 3 minutes)

1. **Natural-language query panel (Section 0):** Type a question such as:
   - *"Who can cover the lifeguard shift?"*
   - *"Are any certifications expiring soon?"*
   - *"What shifts are uncovered?"*
   - *"List all employees on the team"*
   - *"What is the weather today?"* — to see the graceful unsupported-query response.
2. **Coverage recommendation (Section 1):** Select SHIFT-001 in the sidebar. Observe the enriched
   recommendation showing eligible candidate count, remaining weekly hours, and the runner-up.
3. **Explainable staff checks (Section 2):** Review each candidate's pass/fail and reasoning.
4. **Certification alerts (Section 3):** See Taylor Park's expired certification flagged.
5. **Dynamic audit trail (Section 4):** Expand the trail to see the actual decisions recorded
   during this run — which candidates were excluded and why, how the ranking was done, and the
   human oversight gate.
6. Select SHIFT-002 to see the safe escalation path when no candidate qualifies.
7. Download the operations report (.md).
8. **Experimental ML section (Section 5):** Expand "Staffing shortage prediction (ML)" to see the
   predicted risk score, pre-shift features, test-set metrics (precision, recall, F1), and
   confusion matrix. Switch between SHIFT-001 and SHIFT-002 to see risk scores change.

## Architecture

```
Streamlit UI (app.py)
  └── Section 0: answer_query()      ← intent classifier + dispatcher
  └── Sections 1–4: analyze()        ← multi-step workflow orchestrator
  │       ├── eligible()             ← 5-rule eligibility checker per candidate
  │       ├── expiring_certifications() ← proactive cert scan
  │       └── report()              ← downloadable Markdown report
  └── Section 5: predictor.py        ← EXPERIMENTAL — Logistic Regression shortage signal
          ├── train_and_evaluate()   ← trains on shift_history.csv, returns metrics
          ├── predict_shift_risk()   ← returns risk label + probability for one shift
          └── features_from_shift_and_staff() ← derives pre-shift features (no leakage)
```

## Agentic AI capabilities demonstrated

| Capability | Implementation |
|---|---|
| Understand staffing problems from user input | Keyword-based intent classifier in `answer_query()` routes plain-English questions to the correct data function |
| Analyse availability, qualifications, certifications | `eligible()` checks 5 constraints; `expiring_certifications()` proactively scans ahead 30 days |
| Multi-step recommendations based on constraints | `analyze()` pipelines: load → check each rule → filter → rank → recommend or escalate |
| Explain why employees are eligible or ineligible | Per-candidate `reasons` list shown in the explainable checks table and downloadable report |
| Escalate unresolved staffing problems | When no candidate qualifies, the system sets status to `Escalate` and directs the supervisor to the Aquatics Director |

## What is new since the initial MVP

### Natural-language query panel (Priority 1)
- New `answer_query()` function in `engine.py` classifies questions into four intents:
  `coverage`, `certification`, `staff_info`, `shift_list`.
- Ambiguous or unrecognised questions receive a polite explanation of supported topics.
- The system never invents staffing information; all answers are derived from the CSV data.
- Covered by 10 new automated tests in `QueryDispatcherTests`.

### Dynamic audit trail (Priority 2)
- The `analyze()` function now builds a live, step-by-step log as it evaluates candidates.
- Step 1 records the shift and inputs loaded.
- Step 2 lists every excluded candidate with their specific exclusion reason(s).
- Step 3 shows the ranking order with actual hours.
- Step 4 names the recommended candidate (or states the escalation decision) with workload detail.
- Step 5 always records the human-oversight gate.
- Covered by 6 new automated tests in `AuditTrailTests`.

### Experimental ML prediction (Section 5)
- New `predictor.py` module, completely isolated from `engine.py`.
- Trains a **Logistic Regression** (scikit-learn) on `data/shift_history.csv` —
  200 synthetic historical shifts (seed=42, generated once and committed).
- **Chronological split:** first 160 rows train, last 40 test — no future leakage.
- **Features (all pre-shift):** shift duration, role, day of week, start hour,
  available staff count, staff with valid certs, staff with hours budget room.
- **Label:** `understaffed_risk = 1` when fewer than 2 fully eligible candidates
  are projected; derived from a simulated eligibility funnel, not from the
  actual `eligible()` outcome.
- **No data leakage:** `staff_fully_eligible` (the direct label source) is not
  included as a feature; it is computed internally and discarded.
- **Evaluation on test set:**
  - Precision: 0.909 · Recall: 0.769 · F1: 0.833 · Accuracy: 0.900
  - Confusion matrix: [[26, 1], [3, 10]] (rows = actual, cols = predicted)
- Section 5 is **always collapsed by default** and carries a prominent warning
  that predictions are experimental, trained on synthetic data, and must never
  be used to approve unqualified employees.
- Covered by 14 new automated tests in `PredictorTests`.

### Enriched recommendation output (Priority 3)
- `analyze()` now returns `candidate_count`, `runner_up`, and `hours_remaining`.
- The recommendation card shows: selected candidate, hours already scheduled, hours remaining under
  the weekly limit, number of eligible candidates, the selection heuristic used, and the runner-up
  for comparison.
- No arbitrary confidence percentages are displayed.
- Covered by 5 new automated tests in `EnrichedRecommendationTests`.

## Supported now

- Reads sample CSV inputs.
- Keyword-based natural-language query panel for staffing and certification questions.
- Checks role match, yes/no availability, certification expiry at shift end, weekly hours limit,
  and overlap with existing assignments.
- Ranks candidates by fewest current weekly hours.
- Explains every exclusion; proactively alerts on expiring certifications.
- Dynamic audit trail recording actual decisions made during each analysis run.
- Enriched recommendation card with candidate count, runner-up comparison, and hours context.
- Downloadable Markdown operations report.
- Human supervisor retains responsibility for all actual coverage decisions.

## ML methodology and limitations

### How the model works
1. `train_and_evaluate()` loads `data/shift_history.csv`, splits it chronologically
   (160 train / 40 test), fits `LogisticRegression(max_iter=1000, random_state=42)`,
   and returns metrics + a fitted model object cached in `st.session_state`.
2. When a shift is selected, `features_from_shift_and_staff()` computes 7 numerical
   features from the shift dict and live staff records — identical inputs a real
   operations system would have before the shift starts.
3. `predict_shift_risk()` calls `model.predict_proba()` and maps the probability
   to Low / Medium / High using fixed thresholds (0.35 / 0.65).
4. The prediction is displayed as a colour-coded badge with the raw score and
   a disclaimer. It has **zero influence** on the eligibility decision in Section 1.

### ML limitations
- Dataset is 200 rows of synthetic data — not representative of any real facility.
- Class balance (~36% at-risk) reflects the generation seed, not operational reality.
- Logistic Regression is linear; it cannot capture non-linear interactions.
- Features are approximations (e.g., no schedule-conflict check in the history data).
- Label noise (~8%) was added deliberately; some training examples are intentionally mislabelled.
- Test-set performance is on the same synthetic distribution as training; it does not
  establish generalisation to real-world shift data.
- The model is retrained from scratch on every new Streamlit session (< 50 ms).

## Not implemented / limitations

- ML prediction is experimental and trained on synthetic data only; not validated for real use.
- No live Salesforce, UKG, scheduling, IBM watsonx, LLM, or external API integrations.
- Intent classification is keyword-based, not semantic; unusual phrasing may not be recognised.
- No verification of real policies, labor regulations, facility coverage ratios or certifications.
- Availability is a global yes/no field, not a date-specific availability calendar.
- Weekly-hours check does not compute exact calendar weeks or overtime rules.
- No persistence or authenticated approval; the supervisor checkbox is a UI-only demonstration.
- Certification check uses one field per staff member; real facilities require multiple credential types.
- Sorting by least hours is an illustrative heuristic, not a validated fairness algorithm.

## Roadmap to IBM watsonx Orchestrate

1. Make `eligible()`, `analyze()`, and `answer_query()` callable as secured tools/APIs.
2. Configure an Orchestrate agent to identify user intent and call these tools.
3. Keep hard safety checks deterministic and require human approval before writes.
4. Evaluate agent accuracy and failure handling; audit all decisions and protect sensitive data.

## Testing evidence

Run the bundled **43 unittest tests** across five test classes:

| Class | Tests | Coverage |
|---|---|---|
| `WorkflowTests` | 8 | Recommendation, certification, availability, conflict, hours, escalation, cert alerts, past date |
| `QueryDispatcherTests` | 10 | All 4 supported intents, unsupported query, empty query, escalation path, no-invention check |
| `AuditTrailTests` | 6 | 5-step structure, shift ID, exclusion names, recommended candidate, escalation, supervisor gate |
| `EnrichedRecommendationTests` | 5 | Candidate count, runner-up identity, hours comparison, hours remaining, no-candidate fields |
| `PredictorTests` | 14 | Training keys, train/test sizes, both classes present, metric ranges, F1 threshold, confusion matrix shape and totals, no leakage, predict keys, risk label validity, probability range, SHIFT-001 < SHIFT-002 risk, feature keys, isolation from engine.py |

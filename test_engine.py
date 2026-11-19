import unittest
from datetime import date
from engine import load_rows, analyze, expiring_certifications, answer_query

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.staff = load_rows('staff.csv')
        self.shifts = load_rows('uncovered_shifts.csv')
        self.assignments = load_rows('assignments.csv')
        self.today = date(2026, 11, 18)

    def test_recommends_lowest_hours_eligible(self):
        self.assertEqual(analyze(self.shifts[0], self.staff, self.assignments, self.today)['recommended']['name'], 'Alex Rivera')

    def test_expired_cert_excluded(self):
        result = analyze(self.shifts[0], self.staff, self.assignments, self.today)
        self.assertIn('Certification expired before shift ends', next(c for c in result['checks'] if c['name'] == 'Taylor Park')['reasons'])

    def test_unavailable_excluded(self):
        result = analyze(self.shifts[0], self.staff, self.assignments, self.today)
        self.assertIn('Not available', next(c for c in result['checks'] if c['name'] == 'Morgan Lee')['reasons'])

    def test_conflict_excluded(self):
        result = analyze(self.shifts[0], self.staff, self.assignments, self.today)
        self.assertIn('Already assigned to overlapping shift', next(c for c in result['checks'] if c['name'] == 'Jordan Kim')['reasons'])

    def test_weekly_hours_excluded(self):
        result = analyze(self.shifts[0], self.staff, self.assignments, self.today)
        self.assertIn('Would exceed weekly hours limit', next(c for c in result['checks'] if c['name'] == 'Sam Patel')['reasons'])

    def test_no_candidate_escalation(self):
        result = analyze(self.shifts[1], self.staff, self.assignments, self.today)
        self.assertIsNone(result['recommended'])
        self.assertTrue(result['status'].startswith('Escalate'))

    def test_alert_on_expired_certification(self):
        alert_names = [x['name'] for x in expiring_certifications(self.staff, self.today)]
        self.assertIn('Taylor Park', alert_names)

    def test_past_shift_not_recommended(self):
        shift = dict(self.shifts[0], date='2026-11-01')
        self.assertIsNone(analyze(shift, self.staff, [], self.today)['recommended'])

# ---------------------------------------------------------------------------
# Priority 1 tests — natural-language query dispatcher
# ---------------------------------------------------------------------------

class QueryDispatcherTests(unittest.TestCase):
    def setUp(self):
        self.staff = load_rows('staff.csv')
        self.shifts = load_rows('uncovered_shifts.csv')
        self.assignments = load_rows('assignments.csv')
        self.today = date(2026, 11, 18)

    def test_cert_query_returns_supported(self):
        """A certification question is recognised and returns supported=True."""
        qr = answer_query('Are any certifications expiring soon?', self.staff, self.shifts, self.assignments, self.today)
        self.assertTrue(qr['supported'])
        self.assertEqual(qr['intent'], 'certification')

    def test_cert_query_includes_expired_staff(self):
        """Taylor Park's expired cert appears in the certification query answer."""
        qr = answer_query('Who has an expired certification?', self.staff, self.shifts, self.assignments, self.today)
        self.assertIn('Taylor Park', qr['answer'])

    def test_coverage_query_with_shift_id(self):
        """A coverage question mentioning SHIFT-001 is routed and returns a recommendation."""
        qr = answer_query('Who can cover SHIFT-001?', self.staff, self.shifts, self.assignments, self.today)
        self.assertTrue(qr['supported'])
        self.assertEqual(qr['intent'], 'coverage')
        self.assertIn('Alex Rivera', qr['answer'])

    def test_coverage_query_no_shift_id_lists_all_shifts(self):
        """A coverage question with no specific shift lists all uncovered shifts."""
        qr = answer_query('Who can cover the shift?', self.staff, self.shifts, self.assignments, self.today)
        self.assertTrue(qr['supported'])
        self.assertIn('SHIFT-001', qr['answer'])
        self.assertIn('SHIFT-002', qr['answer'])

    def test_shift_list_query(self):
        """An open-shifts question returns the list of uncovered shifts."""
        qr = answer_query('What shifts are uncovered?', self.staff, self.shifts, self.assignments, self.today)
        self.assertTrue(qr['supported'])
        self.assertEqual(qr['intent'], 'shift_list')
        self.assertIn('SHIFT-001', qr['answer'])

    def test_staff_roster_query(self):
        """A roster question returns staff info and is supported."""
        qr = answer_query('List all employees on the team', self.staff, self.shifts, self.assignments, self.today)
        self.assertTrue(qr['supported'])
        self.assertEqual(qr['intent'], 'staff_info')
        self.assertIn('Alex Rivera', qr['answer'])

    def test_unsupported_query_returns_not_supported(self):
        """A completely unrelated question returns supported=False."""
        qr = answer_query('What is the weather today?', self.staff, self.shifts, self.assignments, self.today)
        self.assertFalse(qr['supported'])
        self.assertEqual(qr['intent'], 'unknown')

    def test_empty_query_returns_not_supported(self):
        """An empty string returns supported=False without raising an error."""
        qr = answer_query('', self.staff, self.shifts, self.assignments, self.today)
        self.assertFalse(qr['supported'])
        self.assertEqual(qr['intent'], 'unknown')

    def test_coverage_no_candidate_escalation_in_answer(self):
        """A coverage query for SHIFT-002 (no eligible candidate) mentions escalation."""
        qr = answer_query('Who can cover SHIFT-002?', self.staff, self.shifts, self.assignments, self.today)
        self.assertTrue(qr['supported'])
        self.assertIn('Escalate', qr['answer'])

    def test_query_does_not_invent_staff(self):
        """The answer to any supported query only references actual staff names."""
        import re
        real_names = {m['name'] for m in self.staff}
        # Known non-name Title Case phrases that may appear in answers
        known_non_names = {'Swim Instructor', 'Aquatics Director', 'Alex Rivera',
                           'Jamie Chen', 'Morgan Lee', 'Taylor Park', 'Sam Patel',
                           'Jordan Kim', 'Casey Brooks'}
        for query in [
            'Who can cover the lifeguard shift?',
            'List all employees',
            'Are certifications expiring?',
        ]:
            qr = answer_query(query, self.staff, self.shifts, self.assignments, self.today)
            # Extract Title Case two-word phrases and check any that look like names
            found_phrases = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', qr['answer'])
            for phrase in found_phrases:
                if phrase in known_non_names:
                    continue  # expected phrase, not an invented name
                self.assertIn(phrase, real_names,
                              f"Query '{query}' mentioned unexpected phrase '{phrase}'")


# ---------------------------------------------------------------------------
# Priority 2 tests — dynamic audit trail
# ---------------------------------------------------------------------------

class AuditTrailTests(unittest.TestCase):
    def setUp(self):
        self.staff = load_rows('staff.csv')
        self.shifts = load_rows('uncovered_shifts.csv')
        self.assignments = load_rows('assignments.csv')
        self.today = date(2026, 11, 18)

    def test_audit_has_five_steps(self):
        """The dynamic audit trail always has exactly 5 step entries (some with sub-entries)."""
        result = analyze(self.shifts[0], self.staff, self.assignments, self.today)
        step_lines = [line for line in result['audit'] if line.startswith('Step ')]
        self.assertEqual(len(step_lines), 5)

    def test_audit_step1_records_shift_id(self):
        """Step 1 of the audit trail names the shift being evaluated."""
        result = analyze(self.shifts[0], self.staff, self.assignments, self.today)
        self.assertTrue(any('SHIFT-001' in line for line in result['audit']))

    def test_audit_step2_records_exclusion_reasons(self):
        """Step 2 sub-entries name each excluded candidate and their reason."""
        result = analyze(self.shifts[0], self.staff, self.assignments, self.today)
        exclusion_lines = ' '.join(result['audit'])
        self.assertIn('Taylor Park', exclusion_lines)
        self.assertIn('Morgan Lee', exclusion_lines)
        self.assertIn('Jordan Kim', exclusion_lines)
        self.assertIn('Sam Patel', exclusion_lines)

    def test_audit_step4_names_recommended_candidate(self):
        """Step 4 of the audit trail names the recommended candidate."""
        result = analyze(self.shifts[0], self.staff, self.assignments, self.today)
        self.assertTrue(any('Alex Rivera' in line for line in result['audit']))

    def test_audit_escalation_step4_mentions_escalate(self):
        """When no candidate is found, Step 4 mentions escalation."""
        result = analyze(self.shifts[1], self.staff, self.assignments, self.today)
        self.assertTrue(any('Escalat' in line for line in result['audit']))

    def test_audit_step5_always_requires_supervisor(self):
        """Step 5 always requires supervisor review regardless of outcome."""
        for shift in self.shifts:
            result = analyze(shift, self.staff, self.assignments, self.today)
            self.assertTrue(any('supervisor' in line.lower() for line in result['audit']))


# ---------------------------------------------------------------------------
# Priority 3 tests — enriched recommendation fields
# ---------------------------------------------------------------------------

class EnrichedRecommendationTests(unittest.TestCase):
    def setUp(self):
        self.staff = load_rows('staff.csv')
        self.shifts = load_rows('uncovered_shifts.csv')
        self.assignments = load_rows('assignments.csv')
        self.today = date(2026, 11, 18)

    def test_candidate_count_matches_qualified_list(self):
        """candidate_count equals the length of the qualified list."""
        result = analyze(self.shifts[0], self.staff, self.assignments, self.today)
        self.assertEqual(result['candidate_count'], len(result['qualified']))

    def test_runner_up_is_second_in_qualified_list(self):
        """runner_up is the second-lowest-hours eligible candidate."""
        result = analyze(self.shifts[0], self.staff, self.assignments, self.today)
        self.assertIsNotNone(result['runner_up'])
        self.assertEqual(result['runner_up']['name'], result['qualified'][1]['name'])

    def test_runner_up_has_more_hours_than_recommended(self):
        """runner_up has equal or more hours than the recommended candidate."""
        result = analyze(self.shifts[0], self.staff, self.assignments, self.today)
        self.assertGreaterEqual(
            result['runner_up']['hours_this_week'],
            result['recommended']['hours_this_week'],
        )

    def test_hours_remaining_is_correct(self):
        """hours_remaining = max_weekly_hours - hours_this_week for the recommended candidate."""
        result = analyze(self.shifts[0], self.staff, self.assignments, self.today)
        recommended = result['recommended']
        staff_row = next(s for s in self.staff if s['staff_id'] == recommended['staff_id'])
        expected = float(staff_row['max_weekly_hours']) - recommended['hours_this_week']
        self.assertAlmostEqual(result['hours_remaining'], expected)

    def test_no_candidate_fields_are_none(self):
        """When no candidate is found, runner_up and hours_remaining are None."""
        result = analyze(self.shifts[1], self.staff, self.assignments, self.today)
        self.assertIsNone(result['runner_up'])
        self.assertIsNone(result['hours_remaining'])
        self.assertEqual(result['candidate_count'], 0)


# ---------------------------------------------------------------------------
# ML predictor tests
# ---------------------------------------------------------------------------

class PredictorTests(unittest.TestCase):
    """Tests for predictor.py — the experimental ML shortage-prediction module."""

    @classmethod
    def setUpClass(cls):
        """Train the model once for the whole test class (saves time)."""
        from predictor import (
            train_and_evaluate, predict_shift_risk,
            features_from_shift_and_staff, FEATURES, LABEL,
        )
        cls.train_and_evaluate = staticmethod(train_and_evaluate)
        cls.predict_shift_risk = staticmethod(predict_shift_risk)
        cls.features_from_shift_and_staff = staticmethod(features_from_shift_and_staff)
        cls.FEATURES = FEATURES
        cls.LABEL = LABEL
        cls.trained = train_and_evaluate()
        cls.staff = load_rows('staff.csv')
        cls.shifts = load_rows('uncovered_shifts.csv')

    def test_training_produces_required_keys(self):
        """train_and_evaluate() returns all required keys."""
        required = {'model', 'feature_names', 'train_size', 'test_size',
                    'metrics', 'confusion_matrix', 'classes',
                    'both_classes_in_test', 'report_text'}
        self.assertTrue(required.issubset(self.trained.keys()))

    def test_train_test_sizes_correct(self):
        """Chronological split: 160 train rows, 40 test rows."""
        self.assertEqual(self.trained['train_size'], 160)
        self.assertEqual(self.trained['test_size'],  40)

    def test_both_classes_present_in_test(self):
        """Both risk classes (0 and 1) must appear in the test set."""
        self.assertTrue(self.trained['both_classes_in_test'],
                        'Test set is missing one class — increase dataset size or check generation seed.')

    def test_metrics_are_valid_floats(self):
        """All four evaluation metrics are floats in [0, 1]."""
        for name, val in self.trained['metrics'].items():
            self.assertIsInstance(val, float, f'{name} is not a float')
            self.assertGreaterEqual(val, 0.0, f'{name} < 0')
            self.assertLessEqual(val, 1.0,    f'{name} > 1')

    def test_f1_score_above_threshold(self):
        """F1 on the synthetic test set should exceed 0.60 (sanity check)."""
        self.assertGreater(self.trained['metrics']['f1'], 0.60,
                           'F1 unusually low — check data generation or model.')

    def test_confusion_matrix_shape(self):
        """Confusion matrix is 2 rows × 2 columns."""
        cm = self.trained['confusion_matrix']
        self.assertEqual(len(cm), 2)
        self.assertEqual(len(cm[0]), 2)
        self.assertEqual(len(cm[1]), 2)

    def test_confusion_matrix_totals_match_test_size(self):
        """All confusion matrix cells sum to the test set size."""
        cm = self.trained['confusion_matrix']
        total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
        self.assertEqual(total, self.trained['test_size'])

    def test_no_target_leakage_in_features(self):
        """staff_fully_eligible must NOT appear as a feature (direct label leakage)."""
        self.assertNotIn('staff_fully_eligible', self.trained['feature_names'])

    def test_predict_returns_required_keys(self):
        """predict_shift_risk() returns risk_label, probability, features_used."""
        feats = self.features_from_shift_and_staff(self.shifts[0], self.staff)
        pred = self.predict_shift_risk(self.trained, **feats)
        self.assertIn('risk_label',    pred)
        self.assertIn('probability',   pred)
        self.assertIn('features_used', pred)

    def test_predict_risk_label_is_valid(self):
        """risk_label must be one of Low, Medium, High."""
        for shift in self.shifts:
            feats = self.features_from_shift_and_staff(shift, self.staff)
            pred = self.predict_shift_risk(self.trained, **feats)
            self.assertIn(pred['risk_label'], ('Low', 'Medium', 'High'))

    def test_predict_probability_in_range(self):
        """Predicted probability is a float in [0, 1]."""
        for shift in self.shifts:
            feats = self.features_from_shift_and_staff(shift, self.staff)
            pred = self.predict_shift_risk(self.trained, **feats)
            self.assertGreaterEqual(pred['probability'], 0.0)
            self.assertLessEqual(pred['probability'],    1.0)

    def test_shift001_lower_risk_than_shift002(self):
        """SHIFT-001 (Lifeguard, large pool) should score lower risk than SHIFT-002 (single Swim Instructor)."""
        feats1 = self.features_from_shift_and_staff(self.shifts[0], self.staff)
        feats2 = self.features_from_shift_and_staff(self.shifts[1], self.staff)
        prob1  = self.predict_shift_risk(self.trained, **feats1)['probability']
        prob2  = self.predict_shift_risk(self.trained, **feats2)['probability']
        self.assertLess(prob1, prob2,
                        f'Expected SHIFT-001 ({prob1}) < SHIFT-002 ({prob2})')

    def test_features_from_shift_and_staff_keys(self):
        """features_from_shift_and_staff() returns all expected feature keys."""
        expected_keys = {
            'shift_duration_hrs', 'role', 'day_of_week', 'start_hour',
            'available_staff_count', 'staff_cert_valid', 'staff_hours_ok',
        }
        for shift in self.shifts:
            feats = self.features_from_shift_and_staff(shift, self.staff)
            self.assertEqual(set(feats.keys()), expected_keys,
                             f'Unexpected feature keys for {shift["shift_id"]}')

    def test_predictor_independent_of_engine(self):
        """predictor.py must not import from engine.py (isolation check)."""
        import importlib, ast
        from pathlib import Path
        src = Path('predictor.py').read_text(encoding='utf-8')
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, 'module', '') or ''
                for alias in getattr(node, 'names', []):
                    self.assertNotEqual(module, 'engine',
                                        'predictor.py must not import from engine.py')
                    self.assertNotEqual(getattr(alias, 'name', ''), 'engine',
                                        'predictor.py must not import engine')


if __name__ == '__main__':
    unittest.main()

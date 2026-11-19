"""Deterministic, auditable coverage recommendation workflow (synthetic data only)."""
from __future__ import annotations
from datetime import date, datetime, timedelta
from pathlib import Path
import csv
import re

BASE = Path(__file__).parent
DATE_FORMAT = '%Y-%m-%d'
TIME_FORMAT = '%Y-%m-%d %H:%M'


def load_rows(filename: str) -> list[dict]:
    with (BASE / 'data' / filename).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def shift_times(shift: dict) -> tuple[datetime, datetime]:
    day = shift['date']
    start = datetime.strptime(f"{day} {shift['start']}", TIME_FORMAT)
    end = datetime.strptime(f"{day} {shift['end']}", TIME_FORMAT)
    if end <= start:
        end += timedelta(days=1)
    return start, end


def eligible(staff: dict, shift: dict, assignments: list[dict], as_of: date) -> tuple[bool, list[str]]:
    reasons = []
    start, end = shift_times(shift)
    if staff['role'] != shift['role']:
        reasons.append('Role mismatch')
    if staff['available'].strip().lower() != 'yes':
        reasons.append('Not available')
    try:
        cert = date.fromisoformat(staff['cert_expires'])
        if cert < end.date():
            reasons.append('Certification expired before shift ends')
    except ValueError:
        reasons.append('Invalid certification date')
    try:
        hours = float(staff['hours_this_week'])
        maximum = float(staff['max_weekly_hours'])
        requested = (end - start).total_seconds() / 3600
        if hours + requested > maximum:
            reasons.append('Would exceed weekly hours limit')
    except ValueError:
        reasons.append('Invalid hours data')
    if start.date() < as_of:
        reasons.append('Shift date is in the past')
    for assignment in assignments:
        if assignment['staff_id'] == staff['staff_id']:
            astart, aend = shift_times(assignment)
            if start < aend and astart < end:
                reasons.append('Already assigned to overlapping shift')
                break
    return not reasons, reasons


def analyze(shift: dict, staff_list: list[dict], assignments: list[dict], as_of: date | None = None) -> dict:
    """Evaluate every candidate, sort qualified staff by existing weekly workload."""
    as_of = as_of or date.today()
    audit: list[str] = []

    # Step 1 — read inputs
    audit.append(
        f"Step 1 · Loaded shift {shift['shift_id']} ({shift['role']}, "
        f"{shift['date']} {shift['start']}–{shift['end']}) and "
        f"{len(staff_list)} staff record(s) with {len(assignments)} existing assignment(s). "
        f"Review date: {as_of}."
    )

    # Step 2 — evaluate every candidate
    checks = []
    for staff in staff_list:
        ok, reasons = eligible(staff, shift, assignments, as_of)
        checks.append({
            'staff_id': staff['staff_id'], 'name': staff['name'],
            'eligible': ok, 'reasons': reasons,
            'hours_this_week': float(staff['hours_this_week']),
            'cert_expires': staff['cert_expires'],
        })

    ineligible = [c for c in checks if not c['eligible']]
    eligible_count = len(checks) - len(ineligible)

    audit.append(
        f"Step 2 · Evaluated {len(checks)} candidate(s) against 5 rules: "
        "role match, availability, certification validity, weekly hours limit, schedule conflict."
    )
    for c in ineligible:
        audit.append(f"   ✗ {c['name']} excluded — {'; '.join(c['reasons'])}")
    if eligible_count == 0:
        audit.append("   No candidates passed all checks.")
    else:
        passing = [c['name'] for c in checks if c['eligible']]
        audit.append(f"   ✓ {eligible_count} candidate(s) passed all checks: {', '.join(passing)}")

    # Step 3 — rank qualified candidates
    qualified = sorted((c for c in checks if c['eligible']), key=lambda c: (c['hours_this_week'], c['name']))
    if qualified:
        ranking = ', '.join(f"{c['name']} ({c['hours_this_week']} hrs)" for c in qualified)
        audit.append(
            f"Step 3 · Ranked {len(qualified)} eligible candidate(s) by fewest current weekly hours: {ranking}."
        )
    else:
        audit.append("Step 3 · No eligible candidates to rank.")

    # Step 4 — recommend or escalate
    recommended = qualified[0] if qualified else None
    if recommended:
        hours_remaining = float(
            next(s['max_weekly_hours'] for s in staff_list if s['staff_id'] == recommended['staff_id'])
        ) - recommended['hours_this_week']
        shift_start, shift_end = shift_times(shift)
        shift_hrs = (shift_end - shift_start).total_seconds() / 3600
        audit.append(
            f"Step 4 · Recommended {recommended['name']} — lowest weekly hours among eligible candidates "
            f"({recommended['hours_this_week']} hrs scheduled; {hours_remaining:.1f} hrs remaining under limit; "
            f"this shift adds {shift_hrs:.1f} hrs)."
        )
    else:
        audit.append(
            f"Step 4 · No eligible candidate found for a {shift['role']} shift. "
            "Escalating to Aquatics Director."
        )

    # Step 5 — human oversight gate
    audit.append(
        "Step 5 · Awaiting supervisor review. No schedule change occurs automatically. "
        "Supervisor must confirm availability and approve before any assignment is made."
    )

    status = 'Pending supervisor approval' if qualified else 'Escalate: no eligible candidate'

    # Extra fields for richer UI display (Priority 3)
    candidate_count = len(qualified)
    runner_up = qualified[1] if len(qualified) > 1 else None
    if recommended:
        max_hrs = float(next(s['max_weekly_hours'] for s in staff_list if s['staff_id'] == recommended['staff_id']))
        hours_remaining = max_hrs - recommended['hours_this_week']
    else:
        hours_remaining = None

    return {
        'shift': shift, 'checks': checks, 'qualified': qualified,
        'recommended': recommended,
        'status': status,
        'audit': audit,
        'candidate_count': candidate_count,
        'runner_up': runner_up,
        'hours_remaining': hours_remaining,
    }


def expiring_certifications(staff_list: list[dict], as_of: date, within_days: int = 30) -> list[dict]:
    result = []
    for staff in staff_list:
        try:
            remaining = (date.fromisoformat(staff['cert_expires']) - as_of).days
            if remaining <= within_days:
                result.append({'name': staff['name'], 'cert_expires': staff['cert_expires'], 'days_remaining': remaining})
        except ValueError:
            result.append({'name': staff['name'], 'cert_expires': staff['cert_expires'], 'days_remaining': 'Invalid date'})
    return result


# ---------------------------------------------------------------------------
# Natural-language query dispatcher
# ---------------------------------------------------------------------------

_COVERAGE_KEYWORDS = re.compile(
    r'\b(cover|coverage|fill|staff|who can|available for|lifeguard|instructor|shift)\b', re.I
)
_CERT_KEYWORDS = re.compile(
    r'\b(cert|certification|certifications|expir|expire|expiring|expired|renew|renewal|license)\b', re.I
)
_STAFF_KEYWORDS = re.compile(
    r'\b(staff|employee|employees|team|roster|list|who works|who is)\b', re.I
)
_SHIFT_KEYWORDS = re.compile(
    r'\b(shift|shifts|uncovered|open|schedule)\b', re.I
)

_SUPPORTED_TOPICS = (
    '• Coverage / staffing questions — e.g. "Who can cover the Saturday morning shift?"'
    '\n• Certification questions — e.g. "Are any certifications expiring soon?"'
    '\n• Staff roster — e.g. "List all staff" or "Who is on the team?"'
    '\n• Shift list — e.g. "What shifts are uncovered?"'
)


def _classify_intent(query: str) -> str:
    """Return one of: 'coverage', 'certification', 'staff_info', 'shift_list', 'unknown'."""
    q = query.strip()
    if not q:
        return 'unknown'
    cert_score = len(_CERT_KEYWORDS.findall(q))
    coverage_score = len(_COVERAGE_KEYWORDS.findall(q))
    staff_score = len(_STAFF_KEYWORDS.findall(q))
    shift_score = len(_SHIFT_KEYWORDS.findall(q))
    best = max(cert_score, coverage_score, staff_score, shift_score)
    if best == 0:
        return 'unknown'
    if cert_score == best:
        return 'certification'
    if coverage_score == best:
        return 'coverage'
    if shift_score == best:
        return 'shift_list'
    return 'staff_info'


def _match_shift(query: str, shifts: list[dict]) -> dict | None:
    """Try to find a shift mentioned in the query by ID, role, date, or time keywords."""
    q = query.lower()
    for shift in shifts:
        if shift['shift_id'].lower() in q:
            return shift
        if shift['date'] in q:
            return shift
    for shift in shifts:
        role_words = shift['role'].lower().split()
        if any(w in q for w in role_words):
            return shift
    return None


def answer_query(
    query: str,
    staff_list: list[dict],
    shifts: list[dict],
    assignments: list[dict],
    as_of: date,
) -> dict:
    """
    Parse a natural-language staffing question and return a structured response.

    Returns a dict with keys:
      intent    – classified intent string
      answer    – plain-English answer string
      detail    – optional list of dicts for tabular display (may be empty)
      supported – True if the question was understood
    """
    intent = _classify_intent(query)

    if intent == 'certification':
        expiring = expiring_certifications(staff_list, as_of)
        if expiring:
            lines = [f"• {c['name']}: expires {c['cert_expires']} ({c['days_remaining']} days remaining)"
                     for c in expiring]
            answer = (
                f"{len(expiring)} certification(s) are expiring or already expired within 30 days "
                f"of {as_of}:\n" + '\n'.join(lines)
            )
        else:
            answer = f"No certifications are expiring within 30 days of {as_of}."
        return {'intent': intent, 'answer': answer, 'detail': expiring, 'supported': True}

    if intent == 'coverage':
        matched_shift = _match_shift(query, shifts)
        if matched_shift:
            result = analyze(matched_shift, staff_list, assignments, as_of)
            if result['recommended']:
                r = result['recommended']
                answer = (
                    f"For shift {matched_shift['shift_id']} ({matched_shift['role']}, "
                    f"{matched_shift['date']} {matched_shift['start']}–{matched_shift['end']}): "
                    f"{len(result['qualified'])} eligible candidate(s) found. "
                    f"Suggested: {r['name']} ({r['hours_this_week']} hrs scheduled this week). "
                    f"Supervisor approval required before any schedule change."
                )
            else:
                answer = (
                    f"For shift {matched_shift['shift_id']} ({matched_shift['role']}, "
                    f"{matched_shift['date']}): no eligible candidates found. "
                    f"Escalate to Aquatics Director."
                )
            return {'intent': intent, 'answer': answer, 'detail': result['checks'], 'supported': True}
        else:
            # No specific shift identified — summarise all uncovered shifts
            lines = [
                f"• {s['shift_id']}: {s['date']} {s['start']}–{s['end']} ({s['role']})"
                for s in shifts
            ]
            answer = (
                f"There are {len(shifts)} uncovered shift(s) in the sample data:\n"
                + '\n'.join(lines)
                + "\nTo get a coverage recommendation, mention a specific shift ID, date, or role."
            )
            return {'intent': intent, 'answer': answer, 'detail': [], 'supported': True}

    if intent == 'shift_list':
        lines = [
            f"• {s['shift_id']}: {s['date']} {s['start']}–{s['end']} ({s['role']})"
            for s in shifts
        ]
        answer = (
            f"{len(shifts)} uncovered shift(s) in sample data:\n" + '\n'.join(lines)
        )
        return {'intent': intent, 'answer': answer, 'detail': [], 'supported': True}

    if intent == 'staff_info':
        lines = [
            f"• {m['name']} ({m['role']}) — available: {m['available']}, "
            f"cert expires: {m['cert_expires']}, {m['hours_this_week']}/{m['max_weekly_hours']} hrs this week"
            for m in staff_list
        ]
        answer = f"{len(staff_list)} staff member(s) in sample data:\n" + '\n'.join(lines)
        return {'intent': intent, 'answer': answer, 'detail': staff_list, 'supported': True}

    # unknown intent
    answer = (
        "This prototype can only answer questions about:\n"
        + _SUPPORTED_TOPICS
        + "\n\nPlease rephrase your question using one of those topics. "
        "This system does not connect to live data or external AI services."
    )
    return {'intent': 'unknown', 'answer': answer, 'detail': [], 'supported': False}


def report(result: dict, expiring: list[dict]) -> str:
    shift = result['shift']
    lines = [
        '# Aquatics Operations Report',
        f"Shift: {shift['shift_id']} — {shift['date']} {shift['start']}–{shift['end']} ({shift['role']})",
        f"Status: {result['status']}",
        f"Eligible candidates: {len(result['qualified'])}",
    ]
    if result['recommended']:
        c = result['recommended']
        lines.append(f"Recommendation: {c['name']} (current weekly hours: {c['hours_this_week']})")
        lines.append('Supervisor must confirm availability and approve before editing the schedule.')
    else:
        lines.append('No eligible candidate found. Escalate to Aquatics Director; do not assign an unqualified staff member.')
    lines.append('\n## Candidate evaluation')
    for c in result['checks']:
        lines.append(f"- {c['name']}: {'Eligible' if c['eligible'] else '; '.join(c['reasons'])}")
    lines.append('\n## Certification alerts (next 30 days, including expired)')
    lines.extend(f"- {c['name']}: {c['cert_expires']} ({c['days_remaining']} days remaining)" for c in expiring)
    if not expiring:
        lines.append('- None')
    lines.append('\n## Safety and limitations')
    lines.append('- Demonstration uses fictional personnel data. Verify facility policies and current records before real use.')
    lines.append('- This workflow does not automatically edit schedules, send notifications, or connect to a production system.')
    lines.append('- Availability is an example yes/no field, not a live day-specific availability calendar.')
    return '\n'.join(lines) + '\n'

"""Pins the Metaculus peer-score delta convention, per question type.

Why this file exists: the same arithmetic has been got wrong in two opposite directions by
residual-round scripts, and the repo had nothing that stated the convention in executable
form. Both errors are silent — they produce a plausible number with the wrong magnitude.

* **Continuous, doubled.** ``scratch/residual_2026-08-31/q45065_capbug_replay.py`` (lines
  297 and 307) printed a ``d_peer(100ln)`` column as ``2 * (baseline_cf - baseline_pub)``.
  Metaculus HALVES a continuous peer score, and ``numeric_log_score`` already carries that
  halving (it returns ``50 * ln(...)``), so the difference was already in spot-peer points
  and the doubling made every figure in the column 2x too large. It priced a one-bin-miss
  counterfactual at +404 when the truth is +202.
* **Binary / multiple choice, mis-scaled by ln(K).** ``binary_log_score`` and
  ``mc_log_score`` are log-base-K BASELINE scores, so their differences are in ``log_K``
  units and reaching peer points takes a ``ln(K)`` factor. Which way an uncorrected figure
  errs depends on K, so this bullet carries no direction: the quoted-over-true ratio is
  ``1/ln(K)``, which is 1.44 at K=2 (binary OVER-states, correct it by multiplying by
  ``ln 2`` ≈ 0.693) but 0.91 at K=3 and 0.40 at K=12 (multiple choice with three or more
  options UNDER-states). Thirteen dossier scripts in the 2026-09-01 round used a log2 form
  for a binary peer delta, so every one of those figures is 1.44x too large.

The platform formulas these assert against were read from Metaculus's own
``scoring/score_math.py`` on 2026-09-02 (fetched copy:
``scratch/residual_2026-09-01/dossiers/44798_verify_metaculus_score_math.py``):
``evaluate_forecasts_peer_spot_forecast`` computes ``100 * (N/(N-1)) * ln(p/gmp)``, then
``/= 2`` when the question type is in ``QUESTION_CONTINUOUS_TYPES``
(``[numeric, date, discrete]``). The crowd's geometric mean includes us, so the
``N/(N-1)`` factor collapses the expression to
``100 * (ln p_us - mean_others ln p_i)``: changing only OUR forecast moves the score by
``100 * ln(new/old)``, halved for continuous, with no crowd term surviving.

Receipt for the corrected q45065 figures:
``scratch/residual_2026-09-01/DOSSIER_SYNTHESIS.md`` section 7.2.
"""

import math

import numpy as np
import pytest

from metaculus_bot.scoring_common import (
    CONTINUOUS_PEER_DIVISOR,
    CONTINUOUS_QUESTION_TYPES,
    binary_log_score,
    mc_log_score,
    numeric_log_score,
    spot_peer_delta,
)

# A closed-bound 201-point grid over [0, 200] whose resolution lands in one known bucket,
# so the whole pmf can be described by "how much mass sits on the resolving outcome".
GRID_POINTS = 201
N_INBOUND = GRID_POINTS - 1
RESOLUTION = 100.5
RESOLVING_STEP_INDEX = 100
LOWER_BOUND = 0.0
UPPER_BOUND = 200.0


def _cdf_with_resolving_mass(mass: float) -> list[float]:
    """A legal closed-bound CDF placing ``mass`` on the bucket ``RESOLUTION`` falls in."""
    steps = np.full(N_INBOUND, (1.0 - mass) / (N_INBOUND - 1))
    steps[RESOLVING_STEP_INDEX] = mass
    return np.concatenate([[0.0], np.cumsum(steps)]).tolist()


def _numeric_baseline(mass: float) -> float:
    return numeric_log_score(
        _cdf_with_resolving_mass(mass),
        RESOLUTION,
        LOWER_BOUND,
        UPPER_BOUND,
        open_lower_bound=False,
        open_upper_bound=False,
    )


class TestContinuousPeerDeltaIsNotDoubled:
    """The exact shape of the q45065 replay bug: a difference of two continuous baseline
    scores is ALREADY a spot-peer delta."""

    def test_numeric_log_score_difference_equals_the_peer_delta(self):
        old_mass, new_mass = 0.00351, 0.20000

        baseline_delta = _numeric_baseline(new_mass) - _numeric_baseline(old_mass)
        peer_delta = spot_peer_delta(old_prob=old_mass, new_prob=new_mass, question_type="discrete")

        assert baseline_delta == pytest.approx(peer_delta, rel=1e-12), (
            "a continuous baseline-score difference is already on the spot-peer scale; "
            "multiplying it by 2 to 'convert' it is the 2026-08-31 replay bug"
        )

    def test_the_q45065_near_miss_is_priced_at_two_hundred_not_four_hundred(self):
        # Masses from scratch/residual_2026-08-31/q45065_capbug_replay_stdout.txt, resolution
        # row 12: published 0.00351, counterfactual 0.20000. The script printed 404.23 for
        # that row; the platform's continuous halving makes it 202.11. The tolerance is 0.1
        # because the stdout rounds the masses to five decimals and its own 202.11 came from
        # the unrounded pair, so these inputs land at 202.14.
        peer_delta = spot_peer_delta(old_prob=0.00351, new_prob=0.20000, question_type="discrete")

        assert peer_delta == pytest.approx(202.11, abs=0.1)
        assert peer_delta != pytest.approx(404.23, abs=1.0)

    def test_the_realized_q45065_outcome_moved_half_a_point_not_one(self):
        # Same table, the row marked ACTUAL: 0.19780 -> 0.20000, printed as 1.11.
        assert spot_peer_delta(old_prob=0.19780, new_prob=0.20000, question_type="discrete") == pytest.approx(
            0.55, abs=0.01
        )


class TestPerTypeFormula:
    """``100 * ln(new/old)``, halved for the continuous types and only those."""

    @pytest.mark.parametrize("question_type", sorted(CONTINUOUS_QUESTION_TYPES))
    def test_continuous_types_are_halved(self, question_type):
        expected = 100.0 * math.log(0.3 / 0.1) / 2.0
        assert spot_peer_delta(old_prob=0.1, new_prob=0.3, question_type=question_type) == pytest.approx(expected)

    @pytest.mark.parametrize("question_type", ["binary", "multiple_choice"])
    def test_binary_and_mc_are_not_halved(self, question_type):
        expected = 100.0 * math.log(0.3 / 0.1)
        assert spot_peer_delta(old_prob=0.1, new_prob=0.3, question_type=question_type) == pytest.approx(expected)

    def test_a_continuous_delta_is_exactly_half_its_binary_twin(self):
        binary = spot_peer_delta(old_prob=0.2, new_prob=0.5, question_type="binary")
        numeric = spot_peer_delta(old_prob=0.2, new_prob=0.5, question_type="numeric")
        assert binary / numeric == pytest.approx(CONTINUOUS_PEER_DIVISOR)

    def test_an_unchanged_forecast_scores_zero(self):
        assert spot_peer_delta(old_prob=0.42, new_prob=0.42, question_type="numeric") == pytest.approx(0.0)

    def test_losing_mass_on_the_resolving_outcome_is_negative(self):
        assert spot_peer_delta(old_prob=0.5, new_prob=0.1, question_type="binary") < 0.0


class TestBaselineToPeerConversionOnTheOtherTypes:
    """``binary_log_score`` / ``mc_log_score`` are log-base-K, so their deltas need ln(K)."""

    def test_binary_baseline_delta_times_ln_two_is_the_peer_delta(self):
        old_prob, new_prob = 0.12, 0.63

        baseline_delta = binary_log_score(new_prob, True) - binary_log_score(old_prob, True)
        peer_delta = spot_peer_delta(old_prob=old_prob, new_prob=new_prob, question_type="binary")

        assert baseline_delta * math.log(2.0) == pytest.approx(peer_delta, rel=1e-12)
        assert baseline_delta / peer_delta == pytest.approx(1.0 / math.log(2.0), rel=1e-12), (
            "the two scales differ by 1/ln2 ~ 1.44; quoting a log2 binary delta as peer points "
            "OVER-states it, and the correction is multiplying by ln 2 ~ 0.693"
        )

    @pytest.mark.parametrize("n_options", [3, 5, 12])
    def test_mc_baseline_delta_times_ln_k_is_the_peer_delta(self, n_options):
        correct = 0
        old_probs = [0.10, *[0.90 / (n_options - 1)] * (n_options - 1)]
        new_probs = [0.55, *[0.45 / (n_options - 1)] * (n_options - 1)]

        baseline_delta = mc_log_score(new_probs, correct) - mc_log_score(old_probs, correct)
        peer_delta = spot_peer_delta(
            old_prob=old_probs[correct], new_prob=new_probs[correct], question_type="multiple_choice"
        )

        assert baseline_delta * math.log(n_options) == pytest.approx(peer_delta, rel=1e-12)
        # The error direction flips with K, which is why the module docstring's bullet states
        # none: at three or more options an uncorrected log_K delta UNDER-states the peer delta
        # (ratio 1/ln K < 1), the opposite of binary's 1.44x inflation.
        assert baseline_delta < peer_delta


class TestFailFast:
    """Both guards block a silently-wrong number rather than returning one."""

    def test_an_unrecognized_question_type_raises(self):
        with pytest.raises(ValueError, match="unrecognized question_type"):
            spot_peer_delta(old_prob=0.1, new_prob=0.2, question_type="conditional_binary")

    @pytest.mark.parametrize(("old_prob", "new_prob"), [(0.0, 0.2), (0.2, 0.0), (-0.1, 0.2)])
    def test_a_non_positive_probability_raises(self, old_prob, new_prob):
        with pytest.raises(ValueError, match="positive probabilities"):
            spot_peer_delta(old_prob=old_prob, new_prob=new_prob, question_type="numeric")

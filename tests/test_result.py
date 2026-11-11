import json
import math
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

from verification import VerificationResult, VerificationStatus


def test_correct_construction() -> None:
    result = VerificationResult(
        status=VerificationStatus.INCONCLUSIVE,
        target_class=7,
        target_lower_bound=0.3,
        max_competitor_upper_bound=0.5,
        explanation="bounds overlap",
        perturbation_epsilon=0.01,
    )

    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.target_class == 7
    assert result.target_lower_bound == 0.3
    assert result.max_competitor_upper_bound == 0.5
    assert result.explanation == "bounds overlap"
    assert result.perturbation_epsilon == 0.01
    assert result.verification_margin == pytest.approx(-0.2)


def test_perturbation_epsilon_defaults_to_none() -> None:
    result = VerificationResult(
        VerificationStatus.INCONCLUSIVE, 0, 0.3, 0.5, "bounds overlap"
    )

    assert result.perturbation_epsilon is None


def test_correct_certified_result() -> None:
    result = VerificationResult(
        status=VerificationStatus.CERTIFIED,
        target_class=3,
        target_lower_bound=0.5,
        max_competitor_upper_bound=0.2,
        explanation="strict dominance",
        perturbation_epsilon=0.05,
    )

    assert result.status is VerificationStatus.CERTIFIED
    assert result.verification_margin == pytest.approx(0.3)
    assert result.verification_margin > 0.0


@pytest.mark.parametrize(
    ("target_lower", "max_competitor_upper"),
    [(0.4, 0.4), (0.3, 0.5), (-1.0, 0.0)],
)
def test_correct_inconclusive_result(
    target_lower: float, max_competitor_upper: float
) -> None:
    result = VerificationResult(
        VerificationStatus.INCONCLUSIVE,
        1,
        target_lower,
        max_competitor_upper,
        "target not separated from competitors",
    )

    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.verification_margin <= 0.0


def test_single_class_vacuous_certification_is_representable() -> None:
    result = VerificationResult(
        VerificationStatus.CERTIFIED, 0, 0.7, -math.inf, "no competitor exists"
    )

    assert result.status is VerificationStatus.CERTIFIED
    assert result.max_competitor_upper_bound == -math.inf
    assert result.verification_margin == math.inf


def test_result_is_immutable() -> None:
    result = VerificationResult(VerificationStatus.INCONCLUSIVE, 0, 0.3, 0.5, "overlap")

    with pytest.raises(FrozenInstanceError):
        result.status = VerificationStatus.CERTIFIED


def test_verification_margin_is_not_stored() -> None:
    stored_fields = {field.name for field in fields(VerificationResult)}

    assert "verification_margin" not in stored_fields
    assert stored_fields == {
        "status",
        "target_class",
        "target_lower_bound",
        "max_competitor_upper_bound",
        "explanation",
        "perturbation_epsilon",
    }


@pytest.mark.parametrize("status", ["CERTIFIED", "UNKNOWN", 1, None])
def test_invalid_status_is_rejected(status: object) -> None:
    with pytest.raises(TypeError, match="VerificationStatus member"):
        VerificationResult(status, 0, 0.5, 0.2, "ok")


@pytest.mark.parametrize("target_class", [True, 1.5, "1", None])
def test_non_integer_target_class_is_rejected(target_class: object) -> None:
    with pytest.raises(TypeError, match="integer class index"):
        VerificationResult(VerificationStatus.CERTIFIED, target_class, 0.5, 0.2, "ok")


def test_negative_target_class_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        VerificationResult(VerificationStatus.CERTIFIED, -1, 0.5, 0.2, "ok")


@pytest.mark.parametrize("target_lower", [math.nan, math.inf, -math.inf])
def test_nonfinite_target_lower_bound_is_rejected(target_lower: float) -> None:
    with pytest.raises(ValueError, match="target_lower_bound"):
        VerificationResult(VerificationStatus.INCONCLUSIVE, 0, target_lower, 0.5, "ok")


@pytest.mark.parametrize("max_competitor_upper", [math.nan, math.inf])
def test_nonfinite_max_competitor_upper_bound_is_rejected(
    max_competitor_upper: float,
) -> None:
    with pytest.raises(ValueError, match="max_competitor_upper_bound"):
        VerificationResult(
            VerificationStatus.INCONCLUSIVE, 0, 0.3, max_competitor_upper, "ok"
        )


@pytest.mark.parametrize(
    ("target_lower", "max_competitor_upper"),
    [("0.5", 0.2), (0.5, [0.2]), (None, 0.2)],
)
def test_non_scalar_bounds_are_rejected(
    target_lower: object, max_competitor_upper: object
) -> None:
    with pytest.raises(TypeError, match="real scalar"):
        VerificationResult(
            VerificationStatus.CERTIFIED, 0, target_lower, max_competitor_upper, "ok"
        )


@pytest.mark.parametrize("explanation", [None, 5, b"bytes"])
def test_non_string_explanation_is_rejected(explanation: object) -> None:
    with pytest.raises(TypeError, match="explanation must be a string"):
        VerificationResult(VerificationStatus.CERTIFIED, 0, 0.5, 0.2, explanation)


@pytest.mark.parametrize("explanation", ["", "   "])
def test_blank_explanation_is_rejected(explanation: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        VerificationResult(VerificationStatus.CERTIFIED, 0, 0.5, 0.2, explanation)


def test_negative_perturbation_epsilon_is_rejected() -> None:
    with pytest.raises(ValueError, match="perturbation_epsilon must be nonnegative"):
        VerificationResult(
            VerificationStatus.CERTIFIED, 0, 0.5, 0.2, "ok", perturbation_epsilon=-0.1
        )


@pytest.mark.parametrize("epsilon", [math.nan, math.inf])
def test_nonfinite_perturbation_epsilon_is_rejected(epsilon: float) -> None:
    with pytest.raises(ValueError, match="perturbation_epsilon"):
        VerificationResult(
            VerificationStatus.CERTIFIED, 0, 0.5, 0.2, "ok", perturbation_epsilon=epsilon
        )


def test_non_scalar_perturbation_epsilon_is_rejected() -> None:
    with pytest.raises(TypeError, match="perturbation_epsilon must be a real scalar"):
        VerificationResult(
            VerificationStatus.CERTIFIED, 0, 0.5, 0.2, "ok", perturbation_epsilon="0.1"
        )


def test_certified_with_zero_margin_is_rejected() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        VerificationResult(VerificationStatus.CERTIFIED, 0, 0.4, 0.4, "tie is not proof")


def test_certified_with_negative_margin_is_rejected() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        VerificationResult(VerificationStatus.CERTIFIED, 0, 0.3, 0.5, "overlap")


def test_inconclusive_with_positive_margin_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-positive"):
        VerificationResult(VerificationStatus.INCONCLUSIVE, 0, 0.5, 0.2, "misreported")


def test_str_is_readable() -> None:
    result = VerificationResult(
        VerificationStatus.CERTIFIED, 7, 0.5, 0.2, "strict dominance", perturbation_epsilon=0.01
    )

    text = str(result)

    assert "CERTIFIED" in text
    assert "target_class=7" in text
    assert "target_lower_bound=0.5" in text
    assert "max_competitor_upper_bound=0.2" in text
    assert "margin=0.3" in text
    assert "epsilon=0.01" in text


def test_str_reports_missing_epsilon() -> None:
    result = VerificationResult(VerificationStatus.INCONCLUSIVE, 0, 0.3, 0.5, "overlap")

    assert "epsilon=none" in str(result)


def test_to_dict_contains_expected_fields() -> None:
    result = VerificationResult(
        VerificationStatus.CERTIFIED, 7, 0.5, 0.2, "strict dominance", perturbation_epsilon=0.01
    )

    payload = result.to_dict()

    assert set(payload) == {
        "status",
        "target_class",
        "target_lower_bound",
        "max_competitor_upper_bound",
        "verification_margin",
        "perturbation_epsilon",
        "explanation",
    }
    assert payload["status"] == "CERTIFIED"
    assert payload["target_class"] == 7
    assert payload["target_lower_bound"] == 0.5
    assert payload["max_competitor_upper_bound"] == 0.2
    assert payload["verification_margin"] == pytest.approx(0.3)
    assert payload["perturbation_epsilon"] == 0.01
    assert payload["explanation"] == "strict dominance"


def test_to_dict_is_json_serializable() -> None:
    result = VerificationResult(
        VerificationStatus.CERTIFIED, 7, 0.5, 0.2, "strict dominance", perturbation_epsilon=0.01
    )

    restored = json.loads(json.dumps(result.to_dict()))

    assert restored["status"] == "CERTIFIED"


def test_from_dict_round_trips() -> None:
    original = VerificationResult(
        VerificationStatus.INCONCLUSIVE, 7, 0.3, 0.5, "bounds overlap", perturbation_epsilon=0.02
    )

    assert VerificationResult.from_dict(original.to_dict()) == original


def test_from_dict_accepts_string_status() -> None:
    payload = {
        "status": "INCONCLUSIVE",
        "target_class": 2,
        "target_lower_bound": 0.1,
        "max_competitor_upper_bound": 0.4,
        "explanation": "overlap",
        "perturbation_epsilon": None,
    }

    result = VerificationResult.from_dict(payload)

    assert result.status is VerificationStatus.INCONCLUSIVE


def test_from_dict_ignores_stored_margin() -> None:
    original = VerificationResult(VerificationStatus.CERTIFIED, 0, 0.5, 0.2, "dominance")
    payload = original.to_dict()
    payload["verification_margin"] = 999.0

    restored = VerificationResult.from_dict(payload)

    assert restored == original
    assert restored.verification_margin == pytest.approx(0.3)


def test_from_dict_rejects_unknown_status_value() -> None:
    payload = {
        "status": "UNSAFE",
        "target_class": 0,
        "target_lower_bound": 0.5,
        "max_competitor_upper_bound": 0.2,
        "explanation": "unknown status",
    }

    with pytest.raises(ValueError):
        VerificationResult.from_dict(payload)


def test_status_enum_has_exactly_two_members() -> None:
    assert {status.value for status in VerificationStatus} == {
        "CERTIFIED",
        "INCONCLUSIVE",
    }
    assert VerificationStatus("CERTIFIED") is VerificationStatus.CERTIFIED
    assert VerificationStatus("INCONCLUSIVE") is VerificationStatus.INCONCLUSIVE


def test_package_exports_only_the_public_contract() -> None:
    import verification

    assert set(verification.__all__) == {
        "VerificationStatus",
        "VerificationResult",
        "LPConstraintBuilder",
        "LPProblem",
        "VariableLayout",
        "build_lp_problem",
    }


def test_results_with_identical_fields_are_equal() -> None:
    first = VerificationResult(
        VerificationStatus.CERTIFIED, 0, 0.5, 0.2, "dominance", perturbation_epsilon=0.01
    )
    second = VerificationResult(
        VerificationStatus.CERTIFIED, 0, 0.5, 0.2, "dominance", perturbation_epsilon=0.01
    )

    assert first == second
    assert first != VerificationResult(VerificationStatus.CERTIFIED, 0, 0.5, 0.2, "dominance")


def test_results_differing_in_status_are_not_equal() -> None:
    certified = VerificationResult(VerificationStatus.CERTIFIED, 0, 0.5, 0.2, "dominance")
    inconclusive = VerificationResult(VerificationStatus.INCONCLUSIVE, 0, 0.2, 0.5, "overlap")

    assert certified != inconclusive
    assert certified != "not a result"


def test_integer_bounds_are_coerced_to_float() -> None:
    result = VerificationResult(VerificationStatus.CERTIFIED, 0, 1, 0, "dominance")

    assert isinstance(result.target_lower_bound, float)
    assert isinstance(result.max_competitor_upper_bound, float)
    assert result.verification_margin == pytest.approx(1.0)


def test_numpy_scalar_bounds_are_accepted() -> None:
    result = VerificationResult(
        VerificationStatus.CERTIFIED, 0, np.float64(0.5), np.float64(0.2), "dominance"
    )

    assert result.target_lower_bound == 0.5
    assert result.max_competitor_upper_bound == 0.2


def test_zero_perturbation_epsilon_is_preserved() -> None:
    result = VerificationResult(
        VerificationStatus.CERTIFIED, 0, 0.5, 0.2, "dominance", perturbation_epsilon=0.0
    )

    assert result.perturbation_epsilon == 0.0
    assert result.to_dict()["perturbation_epsilon"] == 0.0


def test_replace_attaches_perturbation_epsilon() -> None:
    base = VerificationResult(VerificationStatus.CERTIFIED, 0, 0.5, 0.2, "dominance")

    attached = replace(base, perturbation_epsilon=0.03)

    assert base.perturbation_epsilon is None
    assert attached.perturbation_epsilon == 0.03
    assert attached.status is base.status
    assert attached.verification_margin == base.verification_margin


def test_replace_revalidates_the_status_margin_relationship() -> None:
    base = VerificationResult(VerificationStatus.CERTIFIED, 0, 0.5, 0.2, "dominance")

    with pytest.raises(ValueError, match="non-positive"):
        replace(base, status=VerificationStatus.INCONCLUSIVE)


def test_to_dict_returns_an_independent_mapping() -> None:
    result = VerificationResult(VerificationStatus.CERTIFIED, 0, 0.5, 0.2, "dominance")

    payload = result.to_dict()
    payload["status"] = "INCONCLUSIVE"

    assert result.to_dict()["status"] == "CERTIFIED"


def test_single_class_result_round_trips_through_json() -> None:
    result = VerificationResult(
        VerificationStatus.CERTIFIED, 0, 0.7, -math.inf, "no competitor exists"
    )

    restored = VerificationResult.from_dict(json.loads(json.dumps(result.to_dict())))

    assert restored == result
    assert restored.max_competitor_upper_bound == -math.inf


@pytest.mark.parametrize(
    "field",
    [
        "status",
        "target_class",
        "target_lower_bound",
        "max_competitor_upper_bound",
        "explanation",
    ],
)
def test_from_dict_requires_each_field(field: str) -> None:
    payload = {
        "status": "CERTIFIED",
        "target_class": 0,
        "target_lower_bound": 0.5,
        "max_competitor_upper_bound": 0.2,
        "explanation": "dominance",
    }
    del payload[field]

    with pytest.raises(KeyError):
        VerificationResult.from_dict(payload)


def test_from_dict_accepts_a_status_member() -> None:
    payload = {
        "status": VerificationStatus.CERTIFIED,
        "target_class": 0,
        "target_lower_bound": 0.5,
        "max_competitor_upper_bound": 0.2,
        "explanation": "dominance",
    }

    assert VerificationResult.from_dict(payload).status is VerificationStatus.CERTIFIED


def test_from_dict_preserves_missing_epsilon() -> None:
    original = VerificationResult(VerificationStatus.INCONCLUSIVE, 1, 0.1, 0.4, "overlap")
    payload = original.to_dict()
    payload.pop("perturbation_epsilon")

    assert VerificationResult.from_dict(payload).perturbation_epsilon is None


def test_str_for_inconclusive_result() -> None:
    result = VerificationResult(VerificationStatus.INCONCLUSIVE, 3, 0.1, 0.4, "overlap")

    text = str(result)

    assert text.startswith("INCONCLUSIVE")
    assert "target_class=3" in text
    assert "margin=-0.3" in text


def test_large_target_class_index_is_accepted() -> None:
    result = VerificationResult(VerificationStatus.CERTIFIED, 999, 0.5, 0.2, "dominance")

    assert result.target_class == 999


def test_status_enum_compares_with_its_string_value() -> None:
    assert VerificationStatus.CERTIFIED == "CERTIFIED"
    assert VerificationStatus.INCONCLUSIVE == "INCONCLUSIVE"


def test_randomized_status_and_margin_remain_consistent() -> None:
    # CERTIFIED is constructible exactly when the margin is positive, and
    # INCONCLUSIVE exactly when it is not, for randomly generated bounds.
    rng = np.random.default_rng(29)

    for _ in range(200):
        target_lower = float(rng.uniform(-5.0, 5.0))
        competitor_upper = float(rng.uniform(-5.0, 5.0))
        margin = target_lower - competitor_upper

        if margin > 0.0:
            result = VerificationResult(
                VerificationStatus.CERTIFIED, 0, target_lower, competitor_upper, "dominance"
            )
            assert result.status is VerificationStatus.CERTIFIED
        else:
            result = VerificationResult(
                VerificationStatus.INCONCLUSIVE, 0, target_lower, competitor_upper, "overlap"
            )
            assert result.status is VerificationStatus.INCONCLUSIVE
        assert result.verification_margin == pytest.approx(margin)

        mismatched = (
            VerificationStatus.INCONCLUSIVE
            if margin > 0.0
            else VerificationStatus.CERTIFIED
        )
        with pytest.raises(ValueError):
            VerificationResult(mismatched, 0, target_lower, competitor_upper, "mismatched status")


def test_randomized_invalid_bounds_are_rejected() -> None:
    rng = np.random.default_rng(37)
    invalid_values = [math.nan, math.inf, -math.inf]

    for _ in range(50):
        invalid = invalid_values[int(rng.integers(0, 3))]

        with pytest.raises(ValueError):
            VerificationResult(
                VerificationStatus.INCONCLUSIVE, 0, invalid, 0.5, "invalid lower bound"
            )
        if invalid != -math.inf:  # -inf is legal only for the competitor bound
            with pytest.raises(ValueError):
                VerificationResult(
                    VerificationStatus.INCONCLUSIVE, 0, 0.5, invalid, "invalid upper bound"
                )

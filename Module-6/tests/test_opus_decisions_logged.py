from src.opus import OPUSEngine


def test_opus_decisions_logged_with_statuses_and_reasons():
    opus = OPUSEngine(quality_threshold=0.70)

    # 1. High quality accepted
    d1 = opus.evaluate_candidate("c_001", score=0.88, lane="code")
    assert d1.status == "accepted"
    assert d1.reason == "quality_above_threshold"

    # 2. Low quality rejected
    d2 = opus.evaluate_candidate("c_002", score=0.45, lane="code")
    assert d2.status == "rejected"
    assert d2.reason == "low_quality_score"

    # 3. Quota pressure rejected
    d3 = opus.evaluate_candidate("c_003", score=0.90, lane="general", lane_quota_full=True)
    assert d3.status == "rejected"
    assert d3.reason == "quota_pressure"

    # 4. Deferred for later stage
    d4 = opus.evaluate_candidate("c_004", score=0.95, lane="reasoning", defer=True)
    assert d4.status == "deferred"
    assert d4.reason == "deferred_for_later_stage"

    # 5. Protected floor rescued (Indic lane with score >= 0.50 under quota pressure or threshold)
    d5 = opus.evaluate_candidate("c_005", score=0.55, lane="indic", lane_quota_full=True)
    assert d5.status == "protected-floor-rescued"
    assert d5.reason == "protected_floor_rescued"
    assert d5.rescued_by_protected_floor is True

    valid_statuses = {"accepted", "rejected", "deferred", "protected-floor-rescued"}
    for d in [d1, d2, d3, d4, d5]:
        assert d.status in valid_statuses
        assert d.reason is not None and len(d.reason) > 0

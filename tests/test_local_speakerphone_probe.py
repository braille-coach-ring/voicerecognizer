"""Verify the probe's common-task metric and rejection counts."""

from script.local_speakerphone_probe import metrics


def test_common_task_is_speaker_equal_and_normalizes_both_sides():
    rows = [
        {"speaker": "r2", "label": "di"},
        {"speaker": "r2", "label": "a"},
        {"speaker": "r4", "label": "di"},
        {"speaker": "r4", "label": "a"},
        {"speaker": "environment", "label": "other"},
    ]
    predictions = ["ji", "a", "ji", "other", "a"]
    common = {False: ["a", "di"], True: ["a", "ji"]}
    output = metrics(rows, predictions, common)
    assert output["raw"]["speakerphone_ba"] == 0.25
    assert output["normalized"]["speakerphone_ba"] == 0.75
    assert output["normalized"]["other_recall"] == 0
    assert output["normalized"]["environment_false_accept"] == 1
    assert output["normalized"]["per_speaker"]["r4"]["speech_false_reject"] == 0.5
    # Extra recordings of r2 must not give that person more aggregate weight.
    repeated = metrics(rows + rows[:2] * 4, predictions + predictions[:2] * 4, common)
    assert repeated["normalized"]["speakerphone_ba"] == 0.75

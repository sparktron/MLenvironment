from rl_framework.training.eval_runner import _was_truncated


def test_was_truncated_true_when_timelimit_flag_present() -> None:
    assert _was_truncated([{"TimeLimit.truncated": True}]) is True


def test_was_truncated_false_for_missing_or_false_flags() -> None:
    assert _was_truncated([{}]) is False
    assert _was_truncated([{"TimeLimit.truncated": False}]) is False
    assert _was_truncated({}) is False

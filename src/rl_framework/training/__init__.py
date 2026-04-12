__all__ = ["train", "evaluate"]


def __getattr__(name: str):
    if name == "train":
        from rl_framework.training.sb3_runner import train
        return train
    if name == "evaluate":
        from rl_framework.training.eval_runner import evaluate
        return evaluate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

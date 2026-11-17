"""Run the pinned legacy fit with all 20 epochs and retain every VAL metric."""
from pathlib import Path
import sys

import train

ARM = "aggr_full20"
original_detector = train.Detector


def detector(arm):
    if arm != ARM:
        raise ValueError(arm)
    return original_detector("aggr")


def load_training():
    source = Path(train.__file__).read_text()
    assert train.MAX_EPOCHS == 20 and train.EARLY_PATIENCE == 5
    before = """        if best is not None and epoch - best["epoch"] > EARLY_PATIENCE:
            break
"""
    assert source.count(before) == 1
    source = source.replace(before, "")
    source = source.replace("    start = 1\n    binding =", "    start = 1\n    curve = []\n    binding =", 1)
    source = source.replace('        best, start = state["best"], state["epoch"] + 1',
                            '        curve = state["val_curve"]\n        best, start = state["best"], state["epoch"] + 1', 1)
    source = source.replace('        statistics = val_stats(val, probability)',
                            '        statistics = val_stats(val, probability)\n        curve.append(dict(epoch=epoch, **statistics))', 1)
    source = source.replace('epoch=epoch, best=best,',
                            'epoch=epoch, best=best, val_curve=curve,', 1)
    source = source.replace('    atomic_checkpoint(checkpoint, {"model": best_model, "selection": best})',
                            '    assert [r["epoch"] for r in curve] == list(range(1, 21))\n'
                            '    atomic_checkpoint(checkpoint, {"model": best_model, "selection": best})', 1)
    source = source.replace('                  evaluation=val_stats(val, best_probability))',
                            '                  evaluation=val_stats(val, best_probability), val_curve=curve)', 1)
    assert source.count("val_curve") == 3 and source.count("curve.append") == 1
    namespace = {"__file__": train.__file__, "__name__": "_full20"}
    exec(compile(source, train.__file__, "exec"), namespace)
    namespace.update(Detector=detector, ARMS=(ARM,))
    return namespace


if __name__ == "__main__":
    command = sys.argv.pop(1)
    if command == "fit":
        load_training()["main"]()
    elif command == "check":
        load_training()
        print("PINNED_FULL20_RUNNER_OK")
    elif command == "test":
        train.Detector = detector
        from evaluate import main

        domain, output, index = sys.argv[1:]
        main(domain, ARM, Path(output), Path(index), "cuda")
    else:
        raise ValueError(command)

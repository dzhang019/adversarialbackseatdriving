from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adv_steering.run_poscon_negcon_pipeline import *  # noqa: F401,F403
else:
    from .run_poscon_negcon_pipeline import *  # noqa: F401,F403

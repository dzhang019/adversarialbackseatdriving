from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adv_steering.analyze_poscon_negcon_residuals import *  # noqa: F401,F403
else:
    from .analyze_poscon_negcon_residuals import *  # noqa: F401,F403


if __name__ == "__main__":
    main()

"""``python -m poseassess.wii``: Wii-only recorder (see ``record_cli``)."""

import sys

from poseassess.wii.record_cli import main

if __name__ == "__main__":
    sys.exit(main())

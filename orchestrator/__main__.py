"""Entry point so the package can be run with `python3 -m orchestrator`."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())

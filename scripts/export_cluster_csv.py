"""Export a result archive to a simple x/y CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from harmonic_dla.io import load_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = load_result(args.result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, result.positions, delimiter=",", header="x,y", comments="")


if __name__ == "__main__":
    main()

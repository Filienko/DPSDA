"""Generate epsilon variants of the four tabular examples for a full privacy sweep.

Each base example is written at epsilon=10. We copy it and change exactly two
things: the ``exp_folder`` suffix (so every budget writes to its own results
folder and never collides) and the privacy budget passed to ``pe_runner.run()``.

Budgets and the folder suffix each writes to (legacy suffixes are reused for
1 / 10 / inf so the runs generated earlier are picked up, not regenerated):

    eps     tag        folder suffix   budget
    0.25    eps0p25    _eps0p25        epsilon=0.25
    0.5     eps0p5     _eps0p5         epsilon=0.5
    1       eps1       _eps1           epsilon=1.0
    5       eps5       _eps5           epsilon=5.0
    10      eps10      (none)          epsilon=10.0
    100     eps100     _eps100         epsilon=100.0
    inf     epsinf     _nonoise        epsilon=None, noise_multiplier=0  (no privacy)

Writes variants into example/tabular/variants/<dataset>_<tag>.py.
"""

import os

BASES = ["adult", "breast_cancer", "artificial_characters", "person_activity"]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "variants")

# tag -> (folder_suffix, budget_block)
EPS = {
    "eps0p25": ("_eps0p25", "        epsilon=0.25,"),
    "eps0p5":  ("_eps0p5",  "        epsilon=0.5,"),
    "eps1":    ("_eps1",    "        epsilon=1.0,"),
    "eps5":    ("_eps5",    "        epsilon=5.0,"),
    "eps10":   ("",         "        epsilon=10.0,"),
    "eps100":  ("_eps100",  "        epsilon=100.0,"),
    "epsinf":  ("_nonoise", "        epsilon=None,\n        noise_multiplier=0,"),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    for base in BASES:
        with open(os.path.join(HERE, f"{base}.py")) as f:
            src = f.read()
        assert src.count("        epsilon=10.0,") == 1, f"{base}: epsilon marker not unique"
        assert src.count('_composite_population"') == 1, f"{base}: exp_folder marker not unique"

        for tag, (suffix, budget) in EPS.items():
            out = src.replace(
                '_composite_population"',
                f'_composite_population{suffix}"',
            ).replace(
                "        epsilon=10.0,",
                budget,
            )
            path = os.path.join(OUT, f"{base}_{tag}.py")
            with open(path, "w") as f:
                f.write(out)
            print(f"wrote {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    main()

"""Generate epsilon=1 and no-noise variants of the tabular examples.

For each base example we copy the script verbatim and change exactly two things:
  * the ``exp_folder`` suffix (so the run writes to its own results/checkpoint
    folder and never collides with or resumes the epsilon=10 runs), and
  * the privacy budget passed to ``pe_runner.run(...)``:
        eps1     -> epsilon=1.0
        nonoise  -> epsilon=None, noise_multiplier=0   (zero Gaussian noise)

Nothing else (datasets, iterations, sample schedule, callbacks) is touched.
Writes the variants into example/tabular/variants/.
"""

import os

BASES = ["adult", "breast_cancer", "artificial_characters", "person_activity"]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "variants")

VARIANTS = {
    "eps1": {
        "folder_suffix": "_eps1",
        "budget": "        epsilon=1.0,",
    },
    "nonoise": {
        "folder_suffix": "_nonoise",
        "budget": "        epsilon=None,\n        noise_multiplier=0,",
    },
}


def main():
    os.makedirs(OUT, exist_ok=True)
    for base in BASES:
        with open(os.path.join(HERE, f"{base}.py")) as f:
            src = f.read()

        assert src.count("        epsilon=10.0,") == 1, f"{base}: epsilon marker not unique"
        assert src.count('_composite_population"') == 1, f"{base}: exp_folder marker not unique"

        for name, cfg in VARIANTS.items():
            out = src.replace(
                '_composite_population"',
                f'_composite_population{cfg["folder_suffix"]}"',
            ).replace(
                "        epsilon=10.0,",
                cfg["budget"],
            )
            path = os.path.join(OUT, f"{base}_{name}.py")
            with open(path, "w") as f:
                f.write(out)
            print(f"wrote {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    main()

# DDuo soundness work: notes for picking up again

Working notes for the `soundness-and-packaging` branch of the fork
(`~/code/github/sbaldasty/dduo-python`), which will become one PR to
`uvm-plaid/dduo-python`. Last updated 2026-11-17, at commit `7061c68`; 271 tests
passing.

## Where things stand

Commits on the branch, oldest first:

| commit | what |
|---|---|
| `a62f58b` | `pyproject.toml` (uv, setuptools), `tests/` with fixtures, extras for sklearn/diffprivlib/tensorflow |
| `4939c1a` | `DistanceMetric.__eq__` compared only `bound`, so `L1() == LInf()` and mode checks never fired |
| `149f3d6` | `duet.list`/`duet.map` fell back to an *empty* sensitivity environment (even `map(lambda x: x + 1, ls)`) |
| `572a155` | `map`'s shared per-element environment was mutated by identity results; scaling by constants (division used `c` not `1/c`, negative constants gave negative sensitivities, `\|c\| < 1` shrank row counts) |
| `a66e784` | methods missing from `sens_effects` kept their input's sensitivity (`cumsum`); now unbounded. `ndarray.sum` used `sens_id`. Added `groupby`/`size`, `to_numpy`, sklearn `Bunch` rules |
| `d6af070` | `duet/numpy.py` audit: `sum` returned empty sensitivity for non-L2 input; elementwise functions relabelled value distances as rows; `dot` added sensitivities of a product; `sqrt` never ran; a global `numpy.linalg.norm` monkeypatch; `array_split` crashed |
| `8b082b4` | keyword arguments to tracked methods contributed no sensitivity; `clip(lower=, upper=)` and one-sided clips crashed; data-dependent clip limits were trusted |
| `222577a` | public `bounds` on every `DuetWrapper` (interval arithmetic, `bounds_effects` table) |
| `151532a` | `Rows()` mode: row distance vs value distance (`L1`/`L2`/`LInf`) |
| `e6bb3af` | rows become value distances lazily, at sums and releases, through bounds; `clip` only sets bounds |
| `93e741d` | elementwise binary operations on rows are row-local (products, comparisons, quotients) |
| `8b67d44` | `details=True` on mechanisms returns a `Release` (value, mechanism, scale, sensitivity, bounds) |
| `7061c68` | per-element noise in `gauss`/`renyi_gauss` (was one shared draw per vector); `above_threshold` checked sensitivity, always charges, charges every source |

Not committed: this file.

## The model, as it now stands

**Modes** (`duet/duet_vals.py`, comment above `class Rows`):

- `Rows()` is a *row distance*: sensitivity `s` means one change to a data source
  changes at most `s` rows/elements, each arbitrarily. Data sources start here
  (`read_csv`, `numpy.load`, sklearn loaders). Any per-row computation keeps the
  count.
- `L1`/`L2`/`LInf` are *value distances*: how far the value moves in that norm.
  Mechanisms release these.

**Bounds**: `DuetWrapper.bounds = (lo, hi)` holds for every element on *every*
possible dataset, so it comes only from public constants (clip limits, literals,
"counts are non-negative", comparisons are `[0, 1]`, public data's own values).
`(-inf, inf)` is unknown. `sum` keeps only the sign of its input's bounds
(`n * hi` would reveal the private `n`).

**Rows become value distances** only where one is needed: sums across rows and
mechanisms (`release_sensitivity`). Conversion is
`count * per_row_change(bounds) * elements_per_row`, with
`per_row_change = max(hi - lo, |lo|, |hi|)`. That covers both neighbour relations
(replace one row, add or remove one), and equals `hi - lo` when 0 lies within the
bounds. Unbounded rows are unbounded at release.

**Elementwise binary operations**: `binary_sensitivity` and `rows_combination`.

- If every sensitive operand is `Rows`, the result is row-local whatever the
  operation.
- pandas operands pair records by index label, so take the per-source maximum.
  Positional NumPy operands take the sum.
- Broadcasting across rows is unbounded.
- Value distances follow Lipschitz rules.
- Plain-number constants keep `scale_by_constant` and `floor_div_by_constant`.

**Sensitivity rules** in `sens_effects` take
`(senv, mode, args, kwargs=None, bounds=UNBOUNDED, value=None)` and return
`(senv, mode)`. Bounds rules in `bounds_effects` take `(bounds, args, kwargs)`. A
method missing from either table is unbounded / unknown.

**Migration for user code**: the per-row gradient clipping decorators become
`mode_switch(Rows, L2)` (updated in `examples/grad-renyi*.py` and the Gradient
Descent notebook). `examples/mwem.py` keeps `mode_switch(LInf, L2)`: it clips an
error vector built with `duet.list2`, not rows of data.

## Remaining work

1. **`laplace` on `LInf` vectors.** Laplace noise needs an L1 sensitivity;
   `laplace` takes whatever value distance it gets. Fine for scalars, `L1` values
   and bounded rows. Not fine for `LInf` vectors:
   - `df.count()` has one count per column, each `LInf` 1, but one added row
     moves every column's count. Its L1 sensitivity is the number of columns
     (public).
   - Histograms (`value_counts`, `groupby().size()`) have L1 sensitivity 1 under
     add/remove (DDuo's convention for counts) but 2 under replace-one.

   This needs a decision on the neighbour policy for counts. `per_row_change`
   already covers both relations, which argues for 2. Likely shape: `sens_count`
   returns `L1` with the right factor per method (`shape`: 1; `count()`:
   columns; histograms: 1 or 2).
2. **`map` over a value-distance list** sums other sources' sensitivities over all
   `n` elements (`newσ`), so the private `n` reaches the noise scale (and
   `Release.scale`). This predates the session. The rows path was fixed in
   `8b67d44` by making those sources unbounded. The value path needs the same
   thought, probably in `map`'s composition rules.
3. **In-place mutation isn't tracked.** `d.update(private)` puts private data in
   `d` without changing `d.senv` (only the returned wrapper gets it).
   `series.clip(..., inplace=True)` errs on the safe side. A fix needs `_missing`
   to update `self.senv` for a known list of mutating methods.
4. **Notebook *Additional Examples*, the `auto_stats_svt` cell**:
   - The "upper" SVT query, `clip(lower_b, b+1).sum() - clip(lower_b, b).sum()`,
     now fails the sensitivity check (analysis says 381; interval arithmetic
     can't see the correlation). It equals
     `(df[col].clip(lower=lower_b) - b).clip(0, 1).sum()` (sensitivity 1) only
     when `lower_b <= b + 1`. For smaller `b` the original relies on pandas'
     `clip` with `lower > upper`, which looks unintended. Ask the author.
   - `df[col].clip(upper_b, lower_b)` has its limits swapped.
   - `duet.laplace(df[col] - mean, ...)` releases the raw, unclipped column, one
     noisy value per person. Before this branch the noise was calibrated to
     sensitivity 1, which effectively published the data. Now it's unbounded
     (`inf`).
   - Unrelated to this branch: a `KeyError 0` cell, and a missing
     `census_data.csv`.
5. **Value-distance product formula** (option (a)):
   `|Δ(xy)| ≤ max|x|·Δy + max|y|·Δx` for products of genuine aggregates, as a
   fallback where the row-local rule doesn't apply. Not needed by anything yet.
6. **`dot` with public weights on value distances** is unbounded; it could scale
   by a norm of the weights (`‖w‖∞` for L1, `‖w‖₂` for L2). Low priority.

## Questions for the coauthor

- Does the paper's metric-annotated matrix type (as I remember it, the metric
  between rows is annotated separately from the metric within a row) already
  draw the `Rows`/value distinction? If so, cite it in the PR.
- Which neighbour relation should DDuo use (affects item 1)? `per_row_change`
  currently covers both.
- `examples/list_problem.py` and `examples/map_microbenchmark.py` wrap lists
  with `duet.LInf()`, which now means a value distance. Their outputs are
  unchanged, but were those lists meant as datasets (`Rows`)?
- The intent of the *Additional Examples* "upper" SVT queries for `b < lower_b`.

## noisyvalue follow-ups (`~/code/github/sbaldasty/noisyvalue`)

- `noisyvalue.dduo` (commit `affc384`) duplicates DDuo's calibration and
  accounting and reads `val.senv.get_max()` directly. Once the PR lands, rewrite
  each wrapper as a DDuo mechanism called with `details=True`, passed to
  `observe_release(r.value, mechanism=r.mechanism, scale=r.scale,
  low=r.bounds[0], high=r.bounds[1])`. That gives truncation at 0 for counts and
  for sums of non-negative values automatically, and drops the copied
  accounting.
- `noisyvalue/private.py`: per-unit sensitivity there is computed from the units
  present (data-dependent, so it leaks through the noise scale), and its
  truncation bound `sum(t.hi)` reveals `n`. The idea worth keeping, range
  tracking, now lives in DDuo as `bounds`. Agreed it can be deleted (only
  `test/test_private.py` depends on it).

## Checking a change

```
uv sync --extra examples     # Python 3.14 (.python-version); scikit-learn < 1.9 for diffprivlib
uv run pytest -q             # 271 tests
```

Then compare every example script's exit status and last line against the
previous commit. The privacy-cost lines from `adv_filter`, `adv_odometer`,
`filter_odometer(_2)`, `odometer`, `partition_example`, `renyi`, `rloop` and
`dpl` are deterministic and should match exactly. Also grep the output for
`inf`/`nan`, ignoring the `Rows(inf)`/`LInf(inf)` reprs.

```
cd examples && for f in *.py; do out=$(MPLBACKEND=Agg timeout 600 uv run --project .. python "$f" 2>&1); \
  echo "$f exit=$? :: $(echo "$out" | tail -1 | cut -c1-90)"; done > /tmp/examples.txt
```

For the notebooks, extract each one's code cells into a script that wraps every
cell in `try/except` (printing `CELL ERROR ...`), then run it from `notebooks/`.
Expected now:

- *Simple Examples*: no errors (SVT average about 38.6, total ε = 1.0).
- *Gradient Descent*: no errors (accuracy about 0.77–0.81).
- *Additional Examples*: `KeyError 0`, a missing `census_data.csv`, and the
  `above_threshold ... sensitivity 381.0` error from item 4.

Example scripts that fail regardless: `conditionals.py` (deliberate),
`dpl2.py`, `mwem.py` (needs `../data_long/adult_with_pii.csv`). Expected
gradient accuracies are about 0.82 (`grad-renyi-fast`), 0.81 (`-minibatching`)
and 0.76 (`grad-renyi`).

Local data (git-ignored) created by running things: `data_long/` (37 MB, written
by the Gradient Descent notebook; the `grad-renyi*` examples need it) and
`examples/archive.ics.uci.edu/`.

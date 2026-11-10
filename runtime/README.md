# Runtime inventory

Created by Codex GPT-6, session /root, 2026-11-06 UTC. Owner: Ampixa/user.
Original source cdjk `/Users/cdjk/github/hindi-detector/runtime`; runtime artifacts
k2 `/Volumes/T9/hindi-detector/runtime`. Status active. Next review 2026-11-13.
Current consumer: reproducible IndicF5 synthesis for the Hindi leakage goal.

`indicf5.in` is the authored compatibility dependency input for inspected pinned
IndicF5 source: Torch/Torchaudio 2.6.0, Transformers 4.49.0 and numpy 1.26.4.
PyArrow 19.0.1 is pinned because a newly resolved PyArrow build failed its runtime
import against numpy 1.26 despite resolution succeeding. The original failure
occurred while the upstream model package imported its unused training modules.
`indicf5.freeze` records all resolved installed distribution versions after
installation. `indicf5_install.log` records installation and explicit failures.
`uv-cache/` contains only uv-managed wheels, sdists and resolution metadata for
this task; exact logical paths belong to uv. It is disposable after installation
verification (within 24h of consumer completion), not a separate model backup.
`.venv-indicf5/` at the project root contains the isolated interpreter links and
installed dependencies for this task. Its inventory is `indicf5.freeze`; retain
while this goal needs F5 synthesis. No shared virtual environment is modified.
Package licences apply individually; all inputs are public Python packages.
No HF upload. Model weights are under the separate models inventory.

Reproduce using uv venv with Python 3.12, then uv pip install -r indicf5.freeze.
Until freeze is written, input constraints are not a complete reproducibility
lock. Stop before treating a failed installation as a usable runtime.

For word-localized work, uroman 1.3.1.1 and PyWORLD 0.3.5 are added without
upgrading existing IndicF5 dependencies. word_localization.freeze records the
resulting environment; word_install.log is temporary install output, retained
only until runtime verification (at most 24h afterward). runtime/word-cache is
a task-owned uv package/build cache, disposable once imports are verified.
PyWORLD wrapper is MIT; WORLD is modified BSD. Uroman’s actual distribution
licence/acknowledgment terms are retained in installed metadata. No shared runtime
is modified; original indicf5.freeze remains the reproduction record for that run.

Compatibility incidents: PyWORLD 0.3.6 had no cp312/macOS wheel or source package
usable by this runtime. Verified 0.3.5 source distribution was built instead. Its
legacy pkg_resources import requires setuptools==80.9.0; newer/no setuptools
failed import. The failed word_install.log and successful pyworld_install.log,
uroman_install.log and word_setuptools_install.log document this resolution.
These logs expire within 24h after imports and the pilot are verified.

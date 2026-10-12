# Contributing to FlowAnalytics

Thanks for your interest in improving FlowAnalytics. This guide covers how to
set up a development environment and what we expect in a pull request.

## Development setup

```bash
git clone https://github.com/CamHickling/FlowAnalytics.git
cd FlowAnalytics
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -r envs/requirements-analysis.txt   # if you work on analysis/
```

Set `FLOW_ANALYTICS_DATA_ROOT` to a local dataset (or a small test copy of one
trial) before running pipeline commands.

## Where things go

| Directory | Put here |
| --- | --- |
| `scripts/` | Pipeline tools that read raw trial data and produce outputs |
| `analysis/` | Study-specific statistics and coding workflows |
| `tools/` | Launchers and developer utilities |
| `experiments/` | Prototypes that are not part of the supported pipeline |
| `tests/` | `unittest` test modules named `test_*.py` |
| `docs/` | Specifications and long-form documentation |

## Making a change

1. Branch from `main` (`git checkout -b feature/short-description`).
2. Keep each pull request focused on one change.
3. Validate pipeline changes against **one trial** with the staged `mvp.py`
   commands before running anything in batch.
4. Run the tests: `python -m unittest discover -s tests`.
5. Update the README or `docs/` if you change a command, flag, or output path.

## Data and privacy

The dataset contains recordings and physiological data from study participants.

- **Never commit media, raw heart-rate CSVs, or other participant data.**
  `.gitignore` excludes common video, audio, CSV, and archive formats. Check
  `git status` before you commit anyway.
- Do not paste participant transcripts or identifiable details into issues or
  pull requests.
- Use synthetic or mocked inputs in tests.

## Reporting issues

When you open an issue, include the command you ran, the full error output,
your OS and Python version, and which trial (by ID only) triggered it.

Contributing to actrhythm
=========================

Thanks for your interest in contributing!

How to contribute
- Fork the repository and open a pull request against the `main` branch.
- Keep changes small and focused; one logical change per PR.

Running tests
- Install requirements (if any) and run the test suite:

  pytest -q

Commit messages
- Use Conventional Commits (e.g., feat(...):, fix(...):, docs(...):).
- Include a brief description and, where helpful, additional context in the
  body of the commit message.

Code style
- Keep changes readable and well-documented. Prefer small helper functions
  over large monolithic changes.

Testing
- Add unit tests for new behavior under tests/ and ensure the full suite
  passes locally before opening a PR.

Reporting bugs
- Open an issue with a minimal reproduction and expected vs actual behavior.

"""Multi-device execution for the reference engine (``docs/model_parallelism.md``).

The two seams every workstream shares live here and nowhere else:

- [`.placement`][causalab.neural.shared.parallel.placement] — where a tapped tensor lives across ranks, decided at
  plan time from the registry's parallel plan and never from the value (§4);
- [`.collective`][causalab.neural.shared.parallel.collective] — the one interface through which the engine talks to
  other ranks, with [`Solo`][causalab.neural.shared.parallel.collective.Solo] as the world-1 identity (§3).

Production code takes the protocols; the simulated implementations the tests
run under are ``tests/_helpers/simulated_world.py`` (§10).
"""

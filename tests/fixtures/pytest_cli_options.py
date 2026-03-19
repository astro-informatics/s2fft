from pathlib import Path

DEFAULT_SEED = 8966433580120847635


def pytest_addoption(parser):
    parser.addoption(
        "--seed",
        type=int,
        nargs="*",
        default=[DEFAULT_SEED],
        help=(
            "Seed(s) to use for random number generator fixture rng in tests. If "
            "multiple seeds are passed tests depending on rng will be run for all "
            "seeds specified."
        ),
    )
    parser.addoption(
        "--cache-directory",
        type=Path,
        default=Path(__file__).parent / "cached-test-data",
        help="Path to access / store cached test data from / to.",
    )
    parser.addoption(
        "--use-cache",
        action="store_true",
        help="Use cached test data rather than generating dynamically.",
    )
    parser.addoption(
        "--update-cache",
        action="store_true",
        help="Update cached test data values.",
    )


def pytest_generate_tests(metafunc):
    option = "seed"
    if option in metafunc.fixturenames:
        metafunc.parametrize(option, metafunc.config.getoption(option))


def pytest_collection_modifyitems(items):
    for item in items:
        if "cached_test_case_wrapper" in getattr(item, "fixturenames", ()):
            item.add_marker("uses_cached_data")

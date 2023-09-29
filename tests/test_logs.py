import s2fft.logs as lg
import pytest


def test_incorrect_log_yaml_path():
    dir_name = "random/incorrect/filepath/"

    # Check cannot add samples with different ndim.
    with pytest.raises(ValueError):
        lg.setup_logging(custom_yaml_path=dir_name)


def test_general_logging():
    lg.setup_logging()
    lg.critical_log("A random critical message")
    lg.debug_log("A random debug message")
    lg.warning_log("A random warning message")
    lg.info_log("A random warning message")

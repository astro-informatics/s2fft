import os
import logging.config
import logging
import yaml
import project_name
import colorlog


def setup_logging(custom_yaml_path=None, default_level=logging.DEBUG):
    """initialise and configure logging.

    Should be called at the beginning of code to initialise and configure the
    desired logging level. Logging levels can be ints in [0,50] where 10 is
    debug logging and 50 is critical logging.

    Args:

        custom_yaml_path (string): Complete pathname of desired yaml logging
            configuration. If empty will provide default logging config.

        default_level (int): Logging level at which to configure.

    Raises:

        ValueError: Raised if logging.yaml is not in ./logs/ directory.

    """
    if custom_yaml_path == None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(project_name.__file__)))
            + "/logs/logging.yaml"
        )
    if custom_yaml_path != None:
        path = custom_yaml_path
    value = os.getenv("LOG_CFG", None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())
        if custom_yaml_path == None:
            config["handlers"]["info_file_handler"]["filename"] = os.path.join(
                os.path.dirname(os.path.dirname(os.path.realpath(project_name.__file__)))
                + "/logs/info.log"
            )
            config["handlers"]["debug_file_handler"]["filename"] = os.path.join(
                os.path.dirname(os.path.dirname(os.path.realpath(project_name.__file__)))
                + "/logs/debug.log"
            )
            config["handlers"]["critical_file_handler"]["filename"] = os.path.join(
                os.path.dirname(os.path.dirname(os.path.realpath(project_name.__file__)))
                + "/logs/critical.log"
            )
            config["handlers"]["info_file_handler"]["filename"] = os.path.join(
                os.path.dirname(os.path.dirname(os.path.realpath(project_name.__file__)))
                + "/logs/info.log"
            )
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        raise ValueError("Logging config pathway incorrect.")
        critical_log("Using custom config from {}".format(path))


def debug_log(message):
    """Log a debug message (e.g. for background logs to assist debugging).

    Args:

        message: Message to log.

    """
    logger = logging.getLogger("ProjectName")
    logger.debug(message)


def warning_log(message):
    """Log a warning (e.g. for internal code warnings such as large dynamic
    ranges).

    Args:

        message: Warning to log.

    """
    logger = logging.getLogger("ProjectName")
    logger.warning(message)


def critical_log(message):
    """Log a critical message (e.g. core code failures etc).

    Args:

        message: Message to log.

    """
    logger = logging.getLogger("ProjectName")
    logger.critical(message)


def info_log(message):
    """Log an information message (e.g. evidence value printing, run completion
    etc).

    Args:

        message: Message to log.

    """
    logger = logging.getLogger("ProjectName")
    logger.info(message)

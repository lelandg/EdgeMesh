import logging
import datetime

the_logger = None

# Initialize the logger with options
def setup_logger(
        name=f"{__name__}",
        log_file=f"./logs/{__name__}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
        level=logging.INFO,
        format_string=None,
        date_format='%Y%m%d %H%M%S'):
    """
    Function to set up a logger.
    :param name: Logger name
    :param log_file: Path to the log file
    :param level: Logging level (e.g., logging.INFO, logging.ERROR)
    :param format_string: Log message format
    :return: Configured logger instance

    Example usage:
    my_logger = setup_logger(
        name="MyAppLogger",
        log_file="app.log",
        level=logging.DEBUG,
        format_string="%(asctime)s [%(levelname)s] %(message)s"
    )
    """
    global the_logger
    if the_logger:
        return the_logger
    # Create a logger
    the_logger = logging.getLogger(name)
    the_logger.setLevel(level)

    # Create a file handler for writing logs to a file
    file_handler = logging.FileHandler(log_file)

    # Set the format for the logger
    if not format_string:
        format_string = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_string, datefmt=date_format)
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    the_logger.addHandler(file_handler)

    return the_logger


def get_logger(
        name=f"./logs/{__name__}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
):
    global the_logger
    if not the_logger:
        return setup_logger(name, level=logging.DEBUG)
    return the_logger
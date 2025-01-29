import logging

PROJ_NAME = "bopt"


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_logger(level=logging.INFO, filename=None, add_console=True):
    fmt_str = "%(asctime)s, [%(levelname)s, %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=fmt_str)
    logger = logging.getLogger(PROJ_NAME)

    if add_console:
        logger.handlers.clear()
        console_handler = logging.StreamHandler()
        log_formatter = logging.Formatter(fmt_str)
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename, mode="a")
        log_formatter = logging.Formatter(fmt_str)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    if level is not None:
        logger.setLevel(level)
    logger.propagate = False
    return logger

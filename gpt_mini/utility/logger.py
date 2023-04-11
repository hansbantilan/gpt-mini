import logging


def init(log_name, level=logging.INFO):
    logger = logging.getLogger(log_name)
    logger.setLevel(level)

    # stream handler
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # format handler
    formatter = logging.Formatter(
        "[%(asctime)s - %(name)s - %(levelname)s] %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)

    return logger

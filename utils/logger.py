import logging
import os

from IPOMCP.ipomcp_config import get_config


def get_logger():
    logger = logging.getLogger("mylogger")
    config = get_config()
    environment = config.args.environment
    seed = config.args.seed
    manager_acting_agent = config.get_agent_tom_level("manager")
    worker_acting_agent = config.get_agent_tom_level("worker")
    configuration = f'{environment}_manager_{manager_acting_agent}_worker_{worker_acting_agent}_first_mover_{config.first_acting_agent}_seed_{seed}'
    logdir_path = os.path.join('logs', configuration)
    os.makedirs(logdir_path, exist_ok=True)
    log_path = os.path.join(logdir_path, 'info.log')
    if len(logger.handlers) == 0:
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        logger.addHandler(sh)
        logger.addHandler(fh)

    return logger



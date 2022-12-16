import logging
import os

from IPOMCP_solver.Solver.ipomcp_config import get_config


def get_logger():
    logger = logging.getLogger("mylogger")
    config = get_config()
    environment = config.args.environment
    seed = config.args.seed
    agent_tom_level = config.get_agent_tom_level("agent")
    subject_tom_level = config.get_agent_tom_level("subject")
    configuration = f'{environment}_agent_{agent_tom_level}_subject_{subject_tom_level}_seed_{seed}'
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



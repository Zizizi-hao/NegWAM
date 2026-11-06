import logging

from navsim.agents.negwam.evaluator.utils.scheduler import FlowMatchScheduler
from navsim.agents.negwam.evaluator.utils.utils import data_seq_to_patch, get_mesh_id, sample_timestep_id, warmup_constant_lambda


def init_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


logger = logging.getLogger("NegWAM")

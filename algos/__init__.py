# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

from utils import get_module

from .continuous.ode.cpgdm import ConjugatePGDMSampler, NoisyConjugatePGDMSampler
from .continuous.ode.ddim import VPConjugateSampler
from .continuous.ode.pgdm import PGDMSampler
from .ddim import DDIM
from .ddrm import DDRM
from .dps import DPS
from .identity import Identity
from .mcg import MCG
from .ndtm import NDTM
from .pgdm import PGDM
from .reddiff import REDDIFF
from .reddiff_parallel import REDDIFF_PARALLEL
from .sds import SDS
from .sds_var import SDS_VAR
from .ndtm_blind import NDTM_BLIND
from .dmplug_blind import DMPlug_Blind


def build_algo(cg_model, cfg):
    return get_module("algo", cfg.algo.name)(cg_model, cfg)

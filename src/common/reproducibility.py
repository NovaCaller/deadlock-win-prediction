from typing import Optional


def ensure_reproducibility(seed: Optional[int]):
    if seed is None:
        return

    import random
    import numpy as np
    # noinspection PyPackageRequirements
    import torch

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
"""
This package contains basic modules including data reader, trainers, models.
Supported models:
- RNP
- 3-layer player
- A2R. Attention-to-Rationale. (Yu et al., 2021)
- Invariant Rationalization
"""

from .data_reader import (
    BasicDataset,
    LJP_Bert_OneLable,
    read_cail_transformers,
    load_cail_dataset
)

from .model import RNP_Bert

from .trainer import (
    SimpleTrainingArguments,
    MetricHolder,
    BasicTrainer,
    FreezeGrad
)

from .rnp_trainer import RNP_Trainer

from .utils import (
    read_json_line,
    cuda,
    nest_batch,
    compute_accuracy_with_logits,
    set_random_seed,
    get_timestamp
)
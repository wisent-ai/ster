import os
import pkgutil

# Note: previous versions set os.environ["NUMBA_NUM_THREADS"] = "1" here,
# which caused RuntimeError when numba's reload_config (triggered by
# pynndescent import) detected the env and tried to re-launch threads at a
# different count than the one already running. Operators who actually want
# to constrain numba should set NUMBA_NUM_THREADS in the environment before
# launching python — not from inside our package init.

# pkgutil.extend_path lets sibling packages (wisent-extractors, wisent-evaluators)
# contribute to the wisent.* namespace from separate installed locations.
__path__ = pkgutil.extend_path(__path__, __name__)

# Legacy behavior: append direct child directories so existing code that relied
# on the custom __path__ layout keeps working.
_base = os.path.dirname(__file__)
for _entry in sorted(os.listdir(_base)):
    _path = os.path.join(_base, _entry)
    if os.path.isdir(_path) and not _entry.startswith((".", "_")):
        __path__.append(_path)

__version__ = "0.11.17"

from wisent.core.control.tasks.base.diversity_processors import (
    OpenerPenaltyProcessor,
    PhraseLedger,
    TriePenaltyProcessor,
    build_diversity_processors,
)
from wisent.core.primitives.model_interface.adapters import (
    AudioAdapter,
    BaseAdapter,
    MultimodalAdapter,
    RoboticsAdapter,
    TextAdapter,
    VideoAdapter,
)

# Multi-modal support
from wisent.core.primitives.model_interface.core.wisent import TraitConfig, Wisent
from wisent.core.primitives.models.modalities import (
    AudioContent,
    ImageContent,
    Modality,
    ModalityContent,
    MultimodalContent,
    RobotAction,
    RobotState,
    RobotTrajectory,
    TextContent,
    VideoContent,
)

__all__ = [
    # Version
    "__version__",
    # Diversity processors
    "OpenerPenaltyProcessor",
    "TriePenaltyProcessor",
    "PhraseLedger",
    "build_diversity_processors",
    # Main interface
    "Wisent",
    "TraitConfig",
    # Modalities
    "Modality",
    "ModalityContent",
    "TextContent",
    "AudioContent",
    "VideoContent",
    "ImageContent",
    "RobotState",
    "RobotAction",
    "RobotTrajectory",
    "MultimodalContent",
    # Adapters
    "BaseAdapter",
    "TextAdapter",
    "AudioAdapter",
    "VideoAdapter",
    "RoboticsAdapter",
    "MultimodalAdapter",
]

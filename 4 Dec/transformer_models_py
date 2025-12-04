# --- Imports for PyTorch Transformers (with checks) ---
# This file isolates all PyTorch-related dependencies.

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not found. Transformer models will be disabled.")

try:
    # pip install tab-transformer-pytorch
    from tab_transformer_pytorch import TabTransformer
    TAB_TRANSFORMER_AVAILABLE = True
except ImportError:
    TAB_TRANSFORMER_AVAILABLE = False
    print("Warning: tab-transformer-pytorch not found. TabTransformer model will be disabled.")
    
try:
    # pip install pytorch_tabular
    from pytorch_tabular.models import FTTransformerModel
    from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig, ModelConfig
    from pytorch_tabular.models.common.heads import LinearHeadConfig
    PYTORCH_TABULAR_AVAILABLE = True
except ImportError:
    PYTORCH_TABULAR_AVAILABLE = False
    print("Warning: pytorch-tabular not found. FTTransformer model will be disabled.")
# --- END NEW IMPORTS ---


# --- Tabular-Specific Transformers (Placeholders) ---
# NOTE: These models are NOT scikit-learn compatible and cannot be used
# in the standard Pipeline or CalibratedClassifierCV.
# They require a custom training loop and data handling logic
# which must be added to the `run_experiment` function itself.
# These are added as placeholders to show in the UI.

def build_tab_transformer():
    """
    Builds a TabTransformer model (Placeholder).
    
    This model requires a custom wrapper to be scikit-learn compatible,
    or a custom training loop in `run_experiment`.
    - It needs explicit `cat_dims` (categorical dimensions/cardinalities).
    - It needs explicit `con_dims` (number of continuous features).
    - It needs data to be passed as two separate tensors (categorical, continuous).
    """
    if not (TORCH_AVAILABLE and TAB_TRANSFORMER_AVAILABLE):
        raise NotImplementedError("tab-transformer-pytorch or torch is not installed.")
    
    # A full implementation would require a complex scikit-learn wrapper class
    # that internally handles data conversion, tensor creation, and a training loop.
    raise NotImplementedError(
        "TabTransformer is not scikit-learn compatible. "
        "It requires a custom wrapper and training loop."
    )

def build_ft_transformer_pytorch_tabular():
    """
    Builds an FT-Transformer model via pytorch-tabular (Placeholder).
    
    This is a high-level framework, not a simple model.
    The `run_experiment` function will bypass this and use custom logic.
    """
    if not (TORCH_AVAILABLE and PYTORCH_TABULAR_AVAILABLE):
        raise NotImplementedError("pytorch-tabular or torch is not installed.")

    # This error will be shown if the custom logic in run_experiment fails
    # or if this model is called by mistake.
    raise NotImplementedError(
        "FT-Transformer logic is handled directly in `run_experiment`."
    )

# --- Dictionary of models to be exported ---
# We only add models if their dependencies were successfully imported
TRANSFORMER_MODELS = {}
if TORCH_AVAILABLE:
    if TAB_TRANSFORMER_AVAILABLE:
        TRANSFORMER_MODELS["tab_transformer"] = build_tab_transformer
    if PYTORCH_TABULAR_AVAILABLE:
        TRANSFORMER_MODELS["ft_transformer"] = build_ft_transformer_pytorch_tabular


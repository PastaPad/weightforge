"""Hardware-matched settings, shared by the API and command-line config files."""
import json
from pathlib import Path

from .model import model_config, parameter_count


def hardware_presets():
    directory = Path(__file__).resolve().parent.parent / 'configs'
    settings = json.loads((directory / 'gtx1050ti.json').read_text())
    finetune = json.loads((directory / 'gtx1050ti-dialogue.json').read_text())
    parameters = parameter_count(model_config(8192, settings))
    return dict(presets=[dict(id='gtx1050ti', label='GTX 1050 Ti · 4 GB · modern 42M', settings=settings,
                             finetune_settings=finetune, recommended_vocab_size=8192, parameters=parameters,
                             fp32_weights_mib=parameters * 4 / 1024**2,
                             fp32_weights_gradients_adam_mib=parameters * 16 / 1024**2,
                             memory_estimate_scope='Parameter storage only; excludes activations, CUDA buffers and display usage')])

import logging
import torch
from typing import Union, List, Dict, Any
from allennlp.modules.elmo import _ElmoBiLm, _ElmoCharacterEncoder
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from language_models.allennlp_elmo.allennlp_elmolstm import ElmoLSTMFrozenStates as ElmoLstm

import json

class ELMOBiLM(_ElmoBiLm):
    """
    Substitution of _ElmoBiLM class by usage of ElmoLstm that does not update states
    """
    def __init__(
        self,
        options_file: str,
        weight_file: str,
        requires_grad: bool = False,
        vocab_to_cache: List[str] = None,
    ) -> None:
        super(_ElmoBiLm, self).__init__()

        self._token_embedder = _ElmoCharacterEncoder(
            options_file, weight_file, requires_grad=requires_grad
        )

        self._requires_grad = requires_grad
        if requires_grad and vocab_to_cache:
            logging.warning(
                "You are fine tuning ELMo and caching char CNN word vectors. "
                "This behaviour is not guaranteed to be well defined, particularly. "
                "if not all of your inputs will occur in the vocabulary cache."
            )
        # This is an embedding, used to look up cached
        # word vectors built from character level cnn embeddings.
        self._word_embedding = None
        self._bos_embedding: torch.Tensor = None
        self._eos_embedding: torch.Tensor = None
        if vocab_to_cache:
            logging.info("Caching character cnn layers for words in vocabulary.")
            # This sets 3 attributes, _word_embedding, _bos_embedding and _eos_embedding.
            # They are set in the method so they can be accessed from outside the
            # constructor.
            self.create_cached_cnn_embeddings(vocab_to_cache)

        with open(cached_path(options_file), "r") as fin:
            options = json.load(fin)
        if not options["lstm"].get("use_skip_connections"):
            raise ConfigurationError("We only support pretrained biLMs with residual connections")

        self._elmo_lstm = ElmoLstm(
            input_size=options["lstm"]["projection_dim"],
            hidden_size=options["lstm"]["projection_dim"],
            cell_size=options["lstm"]["dim"],
            num_layers=options["lstm"]["n_layers"],
            memory_cell_clip_value=options["lstm"]["cell_clip"],
            state_projection_clip_value=options["lstm"]["proj_clip"],
            requires_grad=requires_grad,
        )
        self._elmo_lstm.load_weights(weight_file)
        # Number of representation layers including context independent layer
        self.num_layers = options["lstm"]["n_layers"] + 1
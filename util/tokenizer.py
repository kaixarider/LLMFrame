from typing import Union,Optional,List,Tuple
import os
import logging
from pathlib import Path
from transformers import (AutoTokenizer,PreTrainedTokenizer,PreTrainedTokenizerFast)
logger = logging.getLogger(__name__)

AnyTokenizer=Union[AutoTokenizer,PreTrainedTokenizer,PreTrainedTokenizerFast]

def decode_tokens(
    tokenizer: AnyTokenizer,
    token_ids: list[int],
    *,
    skip_special_tokens: bool = False,
) -> str:
    """
    Backend-agnostic equivalent of HF's
    :code:`tokenizer.decode(token_ids, skip_special_tokens=...)`.
    """
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

def encode_tokens(
    tokenizer: AnyTokenizer,
    text: str,
    *,
    add_special_tokens: Optional[bool] = None,
) -> list[int]:
    """
    Backend-agnostic equivalent of HF's
    :code:`tokenizer.encode(text, add_special_tokens=...)`.
    """
    if add_special_tokens is not None:
        return tokenizer.encode(text, add_special_tokens=add_special_tokens)
    return tokenizer.encode(text)

def get_tokenizer(
    tokenizer_name:Union[str,Path],
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    **kwargs
)->AnyTokenizer:
    if tokenizer_mode=="slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False
    try:
        tokenizer=AutoTokenizer.from_pretrained(
            tokenizer_name, *args, trust_remote_code=trust_remote_code, **kwargs
        )
    
    except TypeError as e:
        # The LLaMA tokenizer causes a protobuf error in some environments.
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA-based "
            f"model, use llama_tokenizer instead of the original "
            "tokenizer."
        )
        raise RuntimeError(err_msg) from e
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if not trust_remote_code and (
            "does not exist or is not currently imported." in str(e)
            or "requires you to execute the tokenizer file" in str(e)
        ):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        logger.warning(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )
    return tokenizer

def _replace_none_with_empty(tokens: List[Optional[str]]):
    for i, token in enumerate(tokens):
        if token is None:
            tokens[i] = ""
            
INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET=5
def convert_prompt_ids_to_tokens(
    tokenizer: AnyTokenizer,
    prompt_ids: List[int],
    skip_special_tokens: bool = False,
) -> Tuple[List[str], int, int]:
    """Converts the prompt ids to tokens and returns the tokens and offsets
    for incremental detokenization.

    Note that not all tokens are converted to strings. Only the tokens that
    are necessary for incremental detokenization are converted to strings.
    """
    # We do not need to convert the whole prompt to tokens.
    # Offset a little more in case we have special tokens.
    new_tokens = tokenizer.convert_ids_to_tokens(
        prompt_ids[-INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET - 2:],
        skip_special_tokens=skip_special_tokens)
    read_offset = len(new_tokens)
    prefix_offset = max(
        read_offset - INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET, 0)
    # This is required to guard against out-of-vocab prompt token ids
    _replace_none_with_empty(new_tokens)  # type: ignore[arg-type]
    return new_tokens, prefix_offset, read_offset
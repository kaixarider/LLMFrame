from typing import Optional,Union,List
_SAMPLING_EPS = 1e-5
_MAX_TEMP = 1e-2
Sam_err_str="Sampling args Error!"
class SamplingParemeters:
    '''
    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).
    Args:
        n: Number of output sequences to return for the given prompt.
        best_of: Number of output sequences that are generated from the prompt.
            From these `best_of` sequences, the top `n` sequences are returned.
            `best_of` must be greater than or equal to `n`. This is treated as
            the beam width when `use_beam_search` is True. By default, `best_of`
            is set to `n`.
        presence_penalty: Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens.
        frequency_penalty: Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens.
        temperature: Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        top_p: Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        top_k: Integer that controls the number of top tokens to consider. Set
            to -1 to consider all tokens.
        use_beam_search: Whether to use beam search instead of sampling.
        stop: List of strings that stop the generation when they are generated.
            The returned output will not contain the stop strings.
        ignore_eos: Whether to ignore the EOS token and continue generating
            tokens after the EOS token is generated.
        max_tokens: Maximum number of tokens to generate per output sequence.
        logprobs: Number of log probabilities to return per output token.
    '''
    def __init__(
        self,
        n:int,
        best_of:Optional[int]=None,
        presence_penalty:float=0.0,
        frequency_penalty:float=0.0,
        temperature:float=1.0,
        top_p:float=1.0,
        top_k:int=-1,
        use_beam_search: bool = False,
        stop: Union[None, str, List[str]] = None,
        ignore_eos: bool = False,
        max_tokens: int = 16,
        logprobs: Optional[int] = None,
    )->None:
        self.n=n
        self.best_of=best_of
        self.presence_penalty=presence_penalty
        self.frequency_penalty=frequency_penalty
        self.temperature=temperature
        self.top_p=top_p
        self.top_k=top_k
        self.max_tokens=max_tokens
        if stop is None:
            self.stop=[]
        elif isinstance(stop,str):
            self.stop=[stop]
        else:
            self.stop=list[stop]
        self.use_beam_search=use_beam_search
        if use_beam_search==True:
            self._verity_beam_search()
        self.ignore_eos=ignore_eos
        self.max_tokens=max_tokens
        self.logprobs=logprobs
        self._verify_args()
    def _verify_args(self)->None:
        if self.n<1:
            raise ValueError(Sam_err_str+f"n must be at least 1,got {self.n}")
        if self.best_of and self.best_of<self.n:
            raise ValueError(Sam_err_str+f"besr_of must be greater than n.\
                            n is {self.n} and best_of is {self.best_of}")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(Sam_err_str+f"presence_penalty must be in [-2,2].\
                            It is {self.presence_penalty}")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(Sam_err_str+f"frequency_penalty must be in [-2,2].\
                            It is {self.frequency_penalty}")
        if self.temperature<0.0:
            raise ValueError(Sam_err_str+f"temperature must be greater than 0.\
                            temperature is {self.temperature}")
        if not 0.0<=self.top_p<=1.0:
            raise ValueError(Sam_err_str+"top_p must be in [0,1].")
        if self.top_k<-1 or self.top_k==0:
            raise ValueError(Sam_err_str+"top_k must be -1 (default) or positive number.")
        if self.max_tokens<1:
            raise ValueError(Sam_err_str+f"max_tokens must be not less than 1.\
                            It is {self.max_tokens}")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError(Sam_err_str+f"logprobs must be non-negitive.It is {self.logprobs}")
    def _verity_beam_search(self) -> None:
        if self.best_of == 1:
            raise ValueError(
                "best_of must be greater than 1 when using beam "
                f"search. Got {self.best_of}."
            )
        if self.temperature > _SAMPLING_EPS:
            raise ValueError("temperature must be 0 when using beam search.")
        if self.top_p < 1.0 -_SAMPLING_EPS:
            raise ValueError("top_p must be 1 when using beam search.")
        if self.top_k != -1:
            raise ValueError("top_k must be -1 when using beam search.")
    def _verify_greedy_sampling(self) -> None:
        if self.best_of > 1:
            raise ValueError(
                "best_of must be 1 when using greedy sampling." f"Got {self.best_of}."
            )
        if self.top_p < 1.0 - _SAMPLING_EPS:
            raise ValueError("top_p must be 1 when using greedy sampling.")
        if self.top_k != -1:
            raise ValueError("top_k must be -1 when using greedy sampling.")
    def __repr__(self) -> str:
        return (
            f"SamplingParams(n={self.n}, "
            f"best_of={self.best_of}, "
            f"presence_penalty={self.presence_penalty}, "
            f"frequency_penalty={self.frequency_penalty}, "
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"top_k={self.top_k}, "
            f"use_beam_search={self.use_beam_search}, "
            f"stop={self.stop}, "
            f"ignore_eos={self.ignore_eos}, "
            f"max_tokens={self.max_tokens}, "
            f"logprobs={self.logprobs})"
        )
class LLMProviderError(Exception):
    """Base error for failures while calling an LLM provider."""


class LLMProviderTimeoutError(LLMProviderError):
    """The provider did not respond before the configured timeout."""


class LLMProviderRateLimitError(LLMProviderError):
    """The provider rejected the request because of a rate limit."""


class LLMProviderUnavailableError(LLMProviderError):
    """The provider could not be reached or returned a server-side failure."""


class LLMProviderRequestError(LLMProviderError):
    """The provider rejected an otherwise completed request."""


class LLMInvalidResponseError(LLMProviderError):
    """The provider returned content that could not satisfy our contract."""
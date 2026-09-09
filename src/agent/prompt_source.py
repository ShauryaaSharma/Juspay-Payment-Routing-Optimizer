"""Where a prompt comes from: LangSmith, or the versioned registry in git.

`prompts.py` holds immutable named versions in source. That is the right default
and it stays the default, because it makes a prompt change a reviewable diff
with the code that consumes it, and it is what lets the eval gate mean
something: CI can assert that *this commit* scores 75% because the prompt is
part of the commit.

Hosted prompt management buys something git cannot: editing a prompt without a
deploy, and letting someone who does not open a pull request do it. That is a
real need on a real team, and this module supports it.

## The tension, stated rather than glossed

A prompt that can change without a code change means **the eval gate no longer
gates the thing that actually ran**. CI green on commit abc123 stops implying
"this prompt scores 75%" and starts implying "some prompt did, once".

Three things keep that honest here:

* **Pinning.** `AGENT_PROMPT_COMMIT` pins an exact LangSmith commit. Unpinned,
  you get whatever is live, and that is a deliberate choice you have to make.
* **Recording.** The resolved source, identifier and commit hash go into every
  trace, so a stored run always says which prompt text produced it.
* **Falling back.** No credentials, no network, or an unknown name all fall
  back to the git registry rather than failing. Offline reproducibility is not
  negotiable.

## Why not `client.pull_prompt`

The obvious call returns a LangChain `ChatPromptTemplate`, which drags in
`langchain-core`. `pull_prompt_commit` returns the raw manifest, from which the
template text is extractable with no extra dependency -- consistent with the
rest of this project, which uses the SDK it needs and not the ecosystem around
it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from . import prompts
from .config import Settings, settings


@dataclass(frozen=True)
class ResolvedPrompt:
    """A prompt plus the provenance of where it came from."""

    version: str
    system: str
    source: str  # "registry" or "langsmith"
    identifier: str | None = None  # LangSmith prompt name
    commit: str | None = None  # LangSmith commit hash, when known
    fallback_reason: str | None = None  # why LangSmith was not used

    def describe(self) -> dict[str, Any]:
        return {
            "prompt_version": self.version,
            "prompt_source": self.source,
            "prompt_identifier": self.identifier,
            "prompt_commit": self.commit,
            "prompt_fallback_reason": self.fallback_reason,
        }


def _extract_system_text(manifest: Any) -> str | None:
    """Pull the system message out of a LangSmith prompt manifest.

    The manifest is a serialised LangChain object and its exact shape depends on
    how the prompt was authored, so this walks it defensively and gives up
    cleanly rather than guessing. Giving up means falling back to the registry,
    which is a working prompt -- far better than a half-parsed one.
    """
    if manifest is None:
        return None
    if isinstance(manifest, str):
        return manifest or None

    if isinstance(manifest, dict):
        # ChatPromptTemplate: kwargs.messages[].prompt.kwargs.template
        kwargs = manifest.get("kwargs") or {}
        messages = kwargs.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                identifier = message.get("id") or []
                is_system = any("System" in str(part) for part in identifier)
                inner = (message.get("kwargs") or {}).get("prompt") or {}
                template = (inner.get("kwargs") or {}).get("template")
                if template and (is_system or len(messages) == 1):
                    return str(template)
        # PromptTemplate: kwargs.template
        template = kwargs.get("template")
        if template:
            return str(template)
        for key in ("template", "system", "text"):
            if isinstance(manifest.get(key), str) and manifest[key]:
                return manifest[key]
    return None


class PromptSource:
    """Resolves a prompt version, preferring LangSmith when configured.

    Results are cached in process. Pulling over the network on every
    investigation would put a third-party outage on the incident-response path,
    which is exactly the dependency an on-call engineer does not want.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or settings()
        self._cache: dict[str, ResolvedPrompt] = {}
        self._lock = threading.Lock()
        self._client: Any = None
        self._client_error: str | None = None

    # -- client -----------------------------------------------------------

    def _langsmith(self) -> Any | None:
        if self._client is not None or self._client_error is not None:
            return self._client
        if not self.config.prompts.api_key:
            self._client_error = "LANGSMITH_API_KEY not set"
            return None
        try:
            from langsmith import Client
        except ImportError:
            self._client_error = "langsmith not installed: pip install langsmith"
            return None
        try:
            self._client = Client(
                api_key=self.config.prompts.api_key,
                api_url=self.config.prompts.endpoint or None,
            )
        except Exception as exc:
            self._client_error = f"{type(exc).__name__}: {exc}"
        return self._client

    # -- resolution -------------------------------------------------------

    def resolve(self, version: str | None = None) -> ResolvedPrompt:
        key = version or self.config.prompt_version or prompts.DEFAULT_VERSION
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        resolved = self._resolve_uncached(key)
        with self._lock:
            self._cache[key] = resolved
        return resolved

    def _resolve_uncached(self, key: str) -> ResolvedPrompt:
        settings_ = self.config.prompts
        if settings_.source != "langsmith":
            return self._from_registry(key, reason=None)

        client = self._langsmith()
        if client is None:
            return self._from_registry(key, reason=self._client_error)

        identifier = settings_.identifier or key
        if settings_.commit:
            identifier = f"{identifier}:{settings_.commit}"

        try:
            commit = client.pull_prompt_commit(identifier)
        except Exception as exc:
            return self._from_registry(
                key, reason=f"pull failed ({type(exc).__name__}: {str(exc)[:120]})"
            )

        text = _extract_system_text(getattr(commit, "manifest", None))
        if not text:
            return self._from_registry(
                key, reason="LangSmith manifest had no extractable system prompt"
            )

        return ResolvedPrompt(
            version=f"langsmith:{settings_.identifier or key}",
            system=text,
            source="langsmith",
            identifier=settings_.identifier or key,
            commit=str(getattr(commit, "commit_hash", None) or settings_.commit or "")[:12] or None,
        )

    def _from_registry(self, key: str, reason: str | None) -> ResolvedPrompt:
        prompt = prompts.get(key)
        return ResolvedPrompt(
            version=prompt.version, system=prompt.system,
            source="registry", fallback_reason=reason,
        )

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def status(self) -> dict[str, Any]:
        settings_ = self.config.prompts
        return {
            "source": settings_.source,
            "identifier": settings_.identifier,
            "pinned_commit": settings_.commit,
            "client_error": self._client_error,
            "cached_versions": sorted(self._cache),
        }

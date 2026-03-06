from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from rag_intelligence.providers import ProviderRegistry
from rag_intelligence.settings import AppSettings


def get_settings(request: Request) -> AppSettings:
    return request.app.state.settings


def get_registry(request: Request) -> ProviderRegistry:
    return request.app.state.registry


SettingsDep = Annotated[AppSettings, Depends(get_settings)]
RegistryDep = Annotated[ProviderRegistry, Depends(get_registry)]

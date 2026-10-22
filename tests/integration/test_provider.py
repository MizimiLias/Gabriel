"""Optional live provider smoke test."""

import os

import pytest
from langchain_core.messages import HumanMessage

from app.configs.llm import get_chat_model


@pytest.mark.provider
@pytest.mark.skipif(
    os.getenv("RUN_PROVIDER_TESTS") != "true",
    reason="RUN_PROVIDER_TESTS is not true",
)
async def test_configured_provider_responds() -> None:
    model = await get_chat_model()
    response = await model.ainvoke([HumanMessage(content="Reply with OK only.")])

    assert response.text.strip()

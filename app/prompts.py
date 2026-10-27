"""Prompt definitions for the chat agent."""

TITLE_SYSTEM_PROMPT = (
    "Create a concise conversation title from the supplied first exchange. "
    "Use the same language as the user's message. Keep it natural and specific, "
    "at most 8 words or 24 CJK characters. Return only the title, without quotes, "
    "labels, or ending punctuation."
)

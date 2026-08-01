from __future__ import annotations


def should_deliver(*, dry_run: bool, enabled: bool) -> bool:
    """Single choke point for "is it OK to send this to an external service".

    Both DiscordSender and NotionSender delegate their send-gate to this function
    instead of each re-implementing the same `not dry_run and enabled` boolean, so
    there is exactly one place to break the safe-by-default guarantee (deleting a
    single `if not should_deliver(...): return` line is loud and localized, instead
    of two independent copy-pasted conditions that can silently drift apart).
    """
    return not dry_run and enabled

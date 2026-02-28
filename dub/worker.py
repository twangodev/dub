"""TaskIQ worker entrypoint.

Run with: taskiq worker dub.worker:broker
"""

from dub.tasks.broker import broker  # noqa: F401

# Import tasks so they get registered with the broker
import dub.tasks.dubbing  # noqa: F401

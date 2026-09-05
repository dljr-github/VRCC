"""Frame geometry every sample counter shares: the capture rate and the
frame length. Kept apart from :mod:`vrcc.audio.source` so a module that only
counts samples can import them without loading PortAudio.
"""

from __future__ import annotations

FRAME_LEN = 512
SAMPLE_RATE = 16000

"""Two equal-slot visual memories: nearest-similarity merge and recent-only."""

from dataclasses import dataclass

import numpy as np


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("feature must be a finite 1-D vector")
    length = np.linalg.norm(vector)
    if length < 1e-8:
        raise ValueError("feature must be nonzero")
    return vector / length


def pool_patches(patches):
    """Average each 4x4 region of a 16x16 patch grid into one memory token."""
    patches = np.asarray(patches, dtype=np.float32)
    if patches.ndim != 3 or patches.shape[:2] != (16, 16):
        raise ValueError("expected a 16x16 patch grid")
    return patches.reshape(4, 4, 4, 4, -1).mean(axis=(1, 3))


@dataclass(frozen=True)
class Entry:
    id: int
    vector: np.ndarray
    preview: np.ndarray
    first_frame: int
    last_frame: int
    mass: int = 1
    position: tuple[int, int] | None = None


@dataclass(frozen=True)
class Match:
    entry: Entry
    score: float


class Memory:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.entries = []
        self.next_id = 0

    def snapshot(self):
        # Entries are replaced rather than mutated, so this stays valid after observe().
        return tuple(self.entries)

    def new_entry(self, vector, preview, frame, position=None):
        result = Entry(self.next_id, vector, np.asarray(preview).copy(), frame, frame, position=position)
        self.next_id += 1
        return result

    def search(self, feature, before_frame=None, entries=None):
        vector = unit(feature)
        choices = self.entries if entries is None else entries
        if before_frame is not None:
            choices = [entry for entry in choices if entry.first_frame < before_frame]
        if not choices:
            return None
        scores = np.stack([entry.vector for entry in choices]) @ vector
        best = int(np.argmax(scores))
        return Match(choices[best], float(scores[best]))

    def assign(self, features):
        """Return a stable slot ID for each feature in a frame."""
        if not self.entries:
            raise ValueError("memory is empty")
        features = np.asarray(features, dtype=np.float32)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        scores = (features / np.maximum(norms, 1e-8)) @ np.stack(
            [entry.vector for entry in self.entries]
        ).T
        return np.array([self.entries[i].id for i in np.argmax(scores, axis=1)])


def merged(a, b):
    center = a.mass * a.vector + b.mass * b.vector
    if np.linalg.norm(center) < 1e-8:
        center = a.vector if a.mass >= b.mass else b.vector
    preview = a.preview if a.mass >= b.mass else b.preview
    return Entry(
        a.id, unit(center), preview,
        min(a.first_frame, b.first_frame), max(a.last_frame, b.last_frame),
        a.mass + b.mass, a.position if a.mass >= b.mass else b.position,
    )


class MergeMemory(Memory):
    def observe(self, feature, preview, frame, position=None):
        incoming = self.new_entry(unit(feature), preview, frame, position)
        if len(self.entries) < self.capacity:
            self.entries.append(incoming)
            return

        vectors = np.stack([entry.vector for entry in self.entries])
        similarity = vectors @ incoming.vector
        nearest = int(np.argmax(similarity))

        # When the incoming patch is genuinely new, combine the most similar
        # existing pair to make space. Otherwise fold it into its closest slot.
        if len(self.entries) > 1:
            pairs = vectors @ vectors.T
            np.fill_diagonal(pairs, -np.inf)
            i, j = np.unravel_index(np.argmax(pairs), pairs.shape)
            if pairs[i, j] > similarity[nearest]:
                self.entries[i] = merged(self.entries[i], self.entries[j])
                self.entries.pop(j)
                self.entries.append(incoming)
                return

        self.entries[nearest] = merged(self.entries[nearest], incoming)


class RecentMemory(Memory):
    def observe(self, feature, preview, frame, position=None):
        self.entries.append(self.new_entry(unit(feature), preview, frame, position))
        if len(self.entries) > self.capacity:
            self.entries.pop(0)

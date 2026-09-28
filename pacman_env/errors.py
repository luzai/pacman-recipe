"""Public exceptions raised by the Pacman environment."""


class PacmanEnvError(RuntimeError):
    """Base class for Pacman environment errors."""


class InvalidConfigurationError(PacmanEnvError, ValueError):
    """The requested environment configuration is unsupported or invalid."""


class LevelLoadError(PacmanEnvError, ValueError):
    """A level resource is absent, malformed, or fails its integrity check."""


class InvalidActionError(PacmanEnvError, ValueError):
    """An action is not one of the canonical environment action tokens."""


class EpisodeNotStartedError(PacmanEnvError):
    """An episode operation was attempted before reset()."""


class EpisodeFinishedError(PacmanEnvError):
    """step() was called after the current episode finished."""


class EnvironmentClosedError(PacmanEnvError):
    """An operation was attempted after close()."""

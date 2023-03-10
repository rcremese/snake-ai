class BaseValidationError(ValueError):
    pass

class CollisionError(BaseValidationError):
    pass

class ConfigurationError(BaseValidationError):
    pass
class ShapeError(BaseValidationError):
    pass

class EmptyEnvironmentError(BaseValidationError):
    pass

class InitialisationError(BaseValidationError):
    pass

class ResolutionError(BaseValidationError):
    pass

class OutOfBoundsError(BaseValidationError):
    pass
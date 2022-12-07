class BaseValidationError(ValueError):
    pass

class CollisionError(BaseValidationError):
    pass

class ShapeError(BaseValidationError):
    pass
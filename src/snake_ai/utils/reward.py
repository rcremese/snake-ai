import enum

class Reward(enum.Enum):
    FOOD = 10
    COLLISION = -10
    COLLISION_FREE = -1

import enum


class Reward(enum.Enum):
    FOOD = 100
    COLLISION = -100
    COLLISION_FREE = -1

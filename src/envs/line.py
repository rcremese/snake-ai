import pygame
from typing import Tuple, List, Optional
import logging
import numpy as np

class Line():
    def __init__(self, start : Tuple[int], end : Tuple[int]) -> None:
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        return f'Line({self.start}, {self.end})'

    def intersect(self, rect : pygame.Rect) -> Tuple[Tuple[int]]:
        return rect.clipline(self.start, self.end)

    def __len__(self):
        return np.linalg.norm(np.array(self.end) - np.array(self.start))

    def draw(self, display : pygame.display, line_color : Tuple[int], point_color : Tuple[int]):
        pygame.draw.line(display, line_color, self.start, self.end)
        pygame.draw.circle(display, point_color, self.end, radius=5)


def intersection_with_obstacles(initial_line : Line, rect_list : List[pygame.Rect]) -> Line:
    min_distance = initial_line.length
    shortest_line = None

    for rect in rect_list:
        intersection_line = intersect_obstacle(rect)
        if intersection_line is not None:
            dist = len(intersection_line)
            if dist < min_distance:
                min_distance = dist
                shortest_line = intersection_line
    # return the initial line if no intersection has been found
    if shortest_line is None:
        logging.debug(f"No intersection of the line {initial_line} with the obstacles")
        return initial_line
    return shortest_line

def intersect_obstacle(initial_line : Line, rect : pygame.Rect) -> Optional[Line]:
    starting_point = np.array(initial_line)
    intersection = initial_line.intersect(rect)
    if intersection is None:
        return None
    # find shortest distance between starting point and possible intersection points
    dist_1 = np.linalg.norm(starting_point - np.array(intersection[0]))
    dist_2 = np.linalg.norm(starting_point - np.array(intersection[1]))
    if dist_1 < dist_2:
        return Line(initial_line.start, intersection[0])
    return Line(initial_line.start, intersection[1])

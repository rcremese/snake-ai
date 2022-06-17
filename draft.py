from cmath import rect
import pygame


def collision(rect, line):
    clipped_line = rect.clipline(line)

    if clipped_line:
        # If clipped_line is not an empty tuple then the line
        # collides/overlaps with the rect. The returned value contains
        # the endpoints of the clipped line.
        start, end = clipped_line
        x1, y1 = start
        x2, y2 = end
        print(f'start : {start} , end :{end}')
    else:
        print("No clipping. The line is fully outside the rect.")

rect = pygame.Rect(10,10, 5, 5)
line1 = ((0,0), (rect.centerx, rect.centery))
line2 = ((0,0), (3, 3))

collision(rect, line1)
collision(rect, line2)

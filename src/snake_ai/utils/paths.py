##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Paths used for pygame
 # @desc Created on 2022-11-10 10:52:18 pm
 # @copyright https://mit-license.org/
 #
from pathlib import Path

FONT_PATH = Path(__file__).parents[1].joinpath('graphics', 'arial.ttf').resolve(strict=True)

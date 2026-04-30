'''
Predefined colors for assembly meshes
'''

import itertools

import distinctipy
import numpy as np

import settings


def get_color(part_ids, normalize=True, scheme='default'):
    color_map = {}

    if scheme=="default":
        if len(part_ids) <= 2:
            colors = np.array([
                [107, 166, 161, 255],
                [209, 184, 148, 255],
            ], dtype=int)
        else:
            colors = np.array([
                [210, 87, 89, 255],
                [237, 204, 73, 255],
                [60, 167, 221, 255],
                [190, 126, 208, 255],
                [108, 192, 90, 255],
            ], dtype=int)
        if normalize: colors = colors.astype(float) / 255.0
    
    elif scheme=="distinctipy": # Based on CAM02-UCS, which maximizes distinguishability for the human eye
        raw_colors = distinctipy.get_colors(len(part_ids))

        colors = np.zeros((len(part_ids), 4))
        for i, (r, g, b) in enumerate(raw_colors):
            colors[i] = [r, g, b, settings.alpha]
        colors[:, 3] *= settings.brightness

        if not normalize:
            colors = (raw_colors * 255).astype(int)

        colors = np.array(colors)

    elif scheme=="max_contrast": # Maximize mathematical contrast, might be better for VLM
        values = [0, 128, 255]
    
        rgb_combinations = list(itertools.product(values, repeat=3))
        filtered_rgb = [rgb for rgb in rgb_combinations if rgb not in [(0, 0, 0), (255, 255, 255)]] # remove black and white
        sorted_rgb = sorted(filtered_rgb, key=lambda x: x.count(128)) # Have most extreme colors (like 255 0 0) be at the top
        colors = [[r, g, b, int(settings.opacity * 255)] for r, g, b in sorted_rgb]

    for i, part_id in enumerate(part_ids):
        color_map[part_id] = colors[i % len(colors)]
    return color_map

# mefd-maps
Scripts and such involved in the 2023 rework of MEFD's map book

Unless otherwise noted, these are scripts meant to be run via the QGIS
processing toolbox.

`usng_to_skagit_grid` converts a nice, normal US National Grid 1km tile
layer into a Skagit-labeled ~~monstrosity~~ layer by doing some fancy
modular arithmetic on the northings + eastings for the starting layer.

`neighboring_grid_tiles.py` takes a Skagit-like grid (see image below) and
adds four new fields to each feature: `east` gives the map ID of the tile
just to the east, `west`, `north`, and `south` the map IDs of the tiles in
the expected directions. This is useful for marking neighboring tiles on an
atlas.

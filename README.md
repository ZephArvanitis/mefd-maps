# mefd-maps
Scripts and such involved in the 2023 rework of MEFD's map book

Unless otherwise noted, these are scripts meant to be run via the QGIS
processing toolbox.

### Processing scripts

`usng_to_skagit_grid` converts a nice, normal US National Grid 1km tile
layer into a Skagit-labeled ~~monstrosity~~ layer by doing some fancy
modular arithmetic on the northings + eastings for the starting layer.

`neighboring_grid_tiles.py` takes a Skagit-like grid (see image below) and
adds four new fields to each feature: `east` gives the map ID of the tile
just to the east, `west`, `north`, and `south` the map IDs of the tiles in
the expected directions. This is useful for marking neighboring tiles on an
atlas.

### Skagit county grid overview
The grid shapes are all based on the 1km US National grid, but with some additional subgridding that is standard across Skagit County. Here's the basic structure:

 - there's a larger grid of 3km x 3km squares, lettered north to south and numbered west to east
 - within each of those grid tiles, there are nine 1km x 1km squares, lettered A B C in the top row, D E F in the middle, and G H I on the bottom.

![grid-structure](https://user-images.githubusercontent.com/4411956/213900531-b28ea587-0d0f-4323-a11f-ad913df0bfa9.png)

CRS note: this grid, like the National Grid, is based off of a [Universal Trans Mercator]([url](https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system)) (UTM) coordinate system. For Skagit County, that means EPSG:26910 - NAD83 / UTM zone 10N.

# mefd-maps
Scripts and such involved in the 2023 rework of MEFD's map book

Unless otherwise noted, these are scripts meant to be run via the QGIS
processing toolbox.

## Processing scripts
Processing scripts in this repo fall into a number of categories:

### Address table generation
A key part of our map book is the ability to quickly look up any address and find what grid tile it's present in. This processing algorithm generates that table and outputs it.

Some notes on it:

- It has an obscene number of inputs, sorry! Each of them is important in
    some way though.
- It is *very important* that all the inputs cover all the area you're
    interested in – one step of the algorithm involves intersecting with
    the grid, the districts, and the "shelter bay" layer. Now you might be
    thinking "Zeph, I know for a fact that Shelter Bay doesn't cover all of
    Fidalgo Island". And you'd be right! ⭐ The shelter bay layer is a
    polygon layer like any other, with a polygon depicting Shelter Bay
    having a non-null value in a field (you'll chooose the field when you
    configure the script for its run), and, importantly, a polygon covering
    the area of interest (e.g. Fidalgo Island only).
    - what this means is that the Shelter Bay layer serves two purposes:
        one is to designate streets as within Shelter Bay (relevant because
        navigation there is famously difficult), and the other is to limit
        the address points and streets that will be shown. For instance, it
        can be used to eliminate address points and roads in La Conner from the
        address table.
    - I know this is kind of gross, but in my opinion it was less gross
        than adding yet ANOTHER input layer. Sorry future self.

Features of this table:

- lists district(s) in which addresses fall or through which roads travel
- calls out when all addresses for a street are in a single tile + district
- highlights when only odd/even addresses appear in a grid tile
- includes roads with no addresses (useful if you need to navigate there
    even though there's not an address on the street)
- corrects for minor differences in street name between the address +
    street layers, like FIRST to 1ST, MARYS to MARY'S, and so on.


### Hydrant reconciliation
`reconcile_hydrants_multi` takes multiple hydrant datasets and does its best to wrangle them into one final dataset.

**Use notes**

The processing algorithm accepts multiple point layers as input. There are some annoying ways the algorithm depends on the exact names of these layers:
- the output layers will have the union of input layers' fields, with the layer name prepended to each field name.
- when multiple datasets "agree" on a hydrant, the *first layer alphabetically* is the location that will be used for the proposed hydrant location.

Based on this, layer names like "0 2023-01 manual hydrants", "1 2023-02 manual hydrants", "2 2023-02 anacortes hydrants", "3 2022-04 MEFD hydrants", etc. are recommended.

**Algorithm notes**

The merging of hydrant datasets is, it turns out, kinda tricky. For some background info, see a [blog post](https://wxyzeph.com/posts/spatial-join/) from mid-2022 with my first attempt. This algorithm basically constructs an [undirected graph](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)) with a node per dataset/hydrant tuple, and edges representing association between two hydrants across their two layers.

Broadly speaking:

- for each pair of input layers:
  - calculates all cross-pairwise distances (that is, distances from every hydrant in layer A to every hydrant in layer B)
  - finds pairs of hydrants that are:
    - each closest to one another (that is, hydrant i in layer A is closest to hydrant j in layer B of all layer B hydrants, and hydrant j in layer B is closest to hydrant i in layer A of all layer A hydrants)
    - within a fixed distance of one another (as of 2023-02-17, hardcoded to 250 feet)
  - all pairs get added as an edge in the undirected graph
- now that the graph is constructed, iterate over [connected components](https://en.wikipedia.org/wiki/Component_(graph_theory)), each of which represents one proposed hydrant, with potentially multiple nearby locations from different datasets. For each component:
  - if the hydrant is only present in one dataset:
    - add it to the *distinct* output layer with a bit of info
    - add it as-is to the *proposed* output layer
  - if the hydrant is present in multiple datasets (that is, the connected component contains more than one node):
    - add lines connecting the multiple locations to the *correspondence* output layer
    - take a best guess as to actual location, based on the alphabetical priority described above. Add that point and other hydrant info to the *proposed* output layer.



### Grid generation + labeling

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

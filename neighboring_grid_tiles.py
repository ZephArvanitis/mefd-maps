# -*- coding: utf-8 -*-

"""
***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""
import string

from qgis.PyQt.QtCore import QCoreApplication, QVariant  # pylint: disable=import-error
from qgis.core import (QgsProcessing,  # pylint: disable=import-error
                       QgsProcessingParameterField,
                       QgsFeatureSink,
                       QgsField,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink)


def get_rowcol_from_skagit_id(cell_label):
    """Skagit ID -> row/column
    """
    cell_skagit_id = cell_label.lower()
    first_letter, number, second_letter = cell_skagit_id
    number = int(number)
    # first_letter = string.ascii_uppercase[row // 3]
    # number = col // 3
    # second_letter = string.ascii_uppercase[(row % 3) * 3 + col % 3]
    row = (ord(first_letter) - ord('a')) * 3 + (ord(second_letter) - ord('a')) // 3
    col = number * 3 + (ord(second_letter) - ord('a')) % 3
    return row, col

def get_skagit_id_from_rowcol(row, col):
    """row/column -> Skagit ID
    """
    first_letter = string.ascii_uppercase[row // 3]
    number = col // 3
    second_letter = string.ascii_uppercase[(row  % 3) * 3 + col % 3]
    grid_label = f"{first_letter}{number}{second_letter}"
    return grid_label

def get_josh_mapbook_id_from_rowcol(row, col):
    """row/col -> mapbook ID (like A15, G-D7, W-B12)

    Fidalgo: from (18, 6) to (35, 23)
    Guemes: from (11, 11) to (17, 17)
    Whidbey: from (32, 7) to (42, 22)
    """
    col_number = col - 5
    col_letter = string.ascii_uppercase[col_number]
    row_number = row - 11

    return f"{col_letter}{row_number:02d}"

def get_rowcol_from_josh_mapbook_id(cell_label):
    """Josh mapbook ID -> rol/column"""
    row_number = int(cell_label[1:])
    col_letter = cell_label[0].lower()
    col_number = ord(col_letter) - ord('a')
    col = col_number + 5
    row = row_number + 11
    return row, col


def get_rowcol_from_usng(northings, eastings):
    """USNG -> row/column
    """
    col = eastings - 14
    row = 92 - northings
    return row, col


class GridNeighborsProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    This takes a Skagit-like grid and adds four new fields to each feature:
    east gives the map ID of the tile just to the east, west, north,
    and south the map IDs of the tiles in the expected directions. This
    is useful for marking neighboring tiles on an atlas.

    All Processing algorithms should extend the QgsProcessingAlgorithm
    class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
    NAME_FIELD = 'name_field'
    EAST_FIELD_NAME = 'east'
    WEST_FIELD_NAME = 'west'
    NORTH_FIELD_NAME = 'north'
    SOUTH_FIELD_NAME = 'south'
    DESTINATION_GRID = 'destination_grid'

    def tr(self, string):  # pylint: disable=invalid-name
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):  # pylint: disable=invalid-name
        return GridNeighborsProcessingAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'neighboring_grid_tiles'

    def displayName(self):  # pylint: disable=invalid-name
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('Add neighboring grid tiles')

    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr('MEFD mapbook scripts')

    def groupId(self):  # pylint: disable=invalid-name
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'mefdmapbookscripts'

    def shortHelpString(self):  # pylint: disable=invalid-name
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it..
        """
        return self.tr("Add neighboring grid tiles to a Skagit-like grid")

    def initAlgorithm(self, config=None):  # pylint: disable=invalid-name
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        # We add the input vector features source. It can have any kind of
        # geometry.
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                self.tr('Input layer'),
                [QgsProcessing.TypeVectorAnyGeometry]
            )
        )

        # Field as parameter
        self.addParameter(
        QgsProcessingParameterField(
                self.NAME_FIELD,
                'Choose Grid ID Field',
                '',
                self.INPUT))

        # Select destination grid from dropdown
        self.addParameter(QgsProcessingParameterEnum(
            self.DESTINATION_GRID,
            'Destination grid',
            options=['Skagit grid', 'Josh grid'],
            defaultValue=0,
            optional=True
            ))

        # We add a feature sink in which to store our processed features (this
        # usually takes the form of a newly created vector layer when the
        # algorithm is run in QGIS).
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr('Output layer')
            )
        )

    def get_all_included_names(self, source, name_field):
        """Return a set of all skagit_ids in the source layer.

        This is so we can omit irrelevant east/west/north/south neighbor
        annotations and avoid confusion.
        """
        features = source.getFeatures()
        all_names = {feature[name_field] for feature in features}
        return all_names

    def processAlgorithm(self, parameters, context, feedback):  # pylint: disable=invalid-name
        """
        Here is where the processing itself takes place.
        """

        # Retrieve the feature source and sink. The 'dest_id' variable is used
        # to uniquely identify the feature sink, and must be included in the
        # dictionary returned by the processAlgorithm function.
        source = self.parameterAsSource(
            parameters,
            self.INPUT,
            context
        )

        # If source was not found, throw an exception to indicate that the algorithm
        # encountered a fatal error. The exception text can be any string, but in this
        # case we use the pre-built invalidSourceError method to return a standard
        # helper text for when a source cannot be evaluated
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        name_field = self.parameterAsString(
            parameters,
            self.NAME_FIELD,
            context)

        destination_grid = self.parameterAsInt(parameters,
                                               self.DESTINATION_GRID,
                                               context)
        grid_functions = {0: (get_skagit_id_from_rowcol,
                              get_rowcol_from_skagit_id),
                          1: (get_josh_mapbook_id_from_rowcol,
                              get_rowcol_from_josh_mapbook_id)}
        from_rowcol, to_rowcol = grid_functions[destination_grid]

        fields = source.fields()
        fields.append(QgsField(self.EAST_FIELD_NAME, QVariant.String))
        fields.append(QgsField(self.WEST_FIELD_NAME, QVariant.String))
        fields.append(QgsField(self.NORTH_FIELD_NAME, QVariant.String))
        fields.append(QgsField(self.SOUTH_FIELD_NAME, QVariant.String))

        (sink, dest_id) = self.parameterAsSink(
                parameters,
                self.OUTPUT,
                context, 
                fields,
                source.wkbType(),
                source.sourceCrs())

        # Send some information to the user
        feedback.pushInfo('CRS is {}'.format(source.sourceCrs().authid()))

        # If sink was not created, throw an exception to indicate that the algorithm
        # encountered a fatal error. The exception text can be any string, but in this
        # case we use the pre-built invalidSinkError method to return a standard
        # helper text for when a sink cannot be evaluated
        if sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT))

        # Compute the number of steps to display within the progress bar and
        # get features from source
        total = 100.0 / source.featureCount() if source.featureCount() else 0
        features = source.getFeatures()
        all_names = self.get_all_included_names(source, name_field)
        feedback.pushInfo(f"All names included: {all_names}")

        for current, feature in enumerate(features):
            # Stop the algorithm if cancel button has been clicked
            if feedback.isCanceled():
                break
            feedback.pushInfo(f'working on feature {feature}, {feature.attributes()}')

            feature.setFields(source.fields(), False)

            # Get rid of pesky middle-of-the-string spaces
            tile_map_id = "".join(feature[name_field].split())
            tile_row, tile_col = to_rowcol(tile_map_id)
            east = from_rowcol(tile_row, tile_col + 1)
            west = from_rowcol(tile_row, tile_col - 1)
            north = from_rowcol(tile_row - 1, tile_col)
            south = from_rowcol(tile_row + 1, tile_col)

            # Drop any values not in the full grid we're neighboring
            east = east if east in all_names else ""
            west = west if west in all_names else ""
            north = north if north in all_names else ""
            south = south if south in all_names else ""

            # Make sure these extra features are in the same order we added
            # the fields above!
            feature.setAttributes(feature.attributes() + [east, west, north, south])

            # Add a feature in the sink
            sink.addFeature(feature, QgsFeatureSink.FastInsert)

            # Update the progress bar
            feedback.setProgress(int(current * total))

        # Return the results of the algorithm.
        return {self.OUTPUT: dest_id}

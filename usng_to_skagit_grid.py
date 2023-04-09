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

def get_rowcol_from_usng(northings, eastings):
    """USNG -> row/column
    """
    col = eastings - 14
    row = 92 - northings
    return row, col

def get_mapbook_id_from_rowcol(row, col):
    """row/col -> mapbook ID (like A15, G-D7, W-B12)

    Fidalgo: from (18, 6) to (35, 23)
    Guemes: from (11, 11) to (17, 17)
    Whidbey: from (32, 7) to (42, 22)
    """
    prefix = ""
    # Guemes: anything with a row <= 17
    col_number = col - 5
    if row <= 17:
        prefix = "G"
        row_letter = string.ascii_uppercase[row - 11]

    else:
        # Gross conditional to catch Whidbey as not-Fidalgo
        is_whidbey = False
        if row > 35:
            is_whidbey = True
        if col < 18:
            if row > 31 or (col > 12 and row > 30):
                is_whidbey = True

        if is_whidbey:
            prefix = "W"
            row_letter = string.ascii_uppercase[row - 31]
        else:
            row_letter = string.ascii_uppercase[row - 18]

    return f"{prefix + '-' if prefix else ''}{row_letter}{col_number:02d}"

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

class USNGtoSkagitGrid(QgsProcessingAlgorithm):
    """
    Add fields to a full 1km USNG grid to depict the 'G4C'-like map ID used
    in the Mt. Erie mapbook
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
    NAME_FIELD = 'name_field'
    NORTHINGS_FIELD = 'Northings'
    EASTINGS_FIELD = 'Eastings'
    DESTINATION_GRID = 'destination_grid'
    MAP_ID_FIELD = 'map_id2'

    def tr(self, string):  # pylint: disable=invalid-name
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):  # pylint: disable=invalid-name
        return USNGtoSkagitGrid()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'usng_to_skagit_grid'

    def displayName(self):  # pylint: disable=invalid-name
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('Convert USNG to Skagit grid')

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
        return self.tr("Example algorithm short description")

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
                    self.NORTHINGS_FIELD,
                    'Field with northings',
                    '',
                    self.INPUT))
        self.addParameter(
            QgsProcessingParameterField(
                    self.EASTINGS_FIELD,
                    'Field with eastings',
                    '',
                    self.INPUT))

        # Which end result?
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

        northings_field = self.parameterAsString(
            parameters,
            self.NORTHINGS_FIELD,
            context)
        eastings_field = self.parameterAsString(
            parameters,
            self.EASTINGS_FIELD,
            context)

        destination_grid = self.parameterAsInt(parameters,
                                               self.DESTINATION_GRID,
                                               context)
        grid_function = {0: get_skagit_id_from_rowcol,
                         1: get_josh_mapbook_id_from_rowcol}
        destination_grid_function = grid_function[destination_grid]

        fields = source.fields()
        fields.append(QgsField(self.MAP_ID_FIELD, QVariant.String))

        (sink, dest_id) = self.parameterAsSink(
                parameters,
                self.OUTPUT,
                context,
                fields,
                source.wkbType(),
                source.sourceCrs())

        # Send some information to the user
        feedback.pushInfo("CRS is {source.sourceCrs().authid()}")

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

        for current, feature in enumerate(features):
            # Stop the algorithm if cancel button has been clicked
            if feedback.isCanceled():
                break

            feature.setFields(source.fields(), False)

            northings = int(feature[northings_field])
            eastings = int(feature[eastings_field])
            row, col = get_rowcol_from_usng(northings, eastings)
            map_id = destination_grid_function(row, col)
            feedback.pushInfo(f'working on feature {feature}, {feature.attributes()}')
            feedback.pushInfo(f"layer has fields {[field.name() for field in source.fields()]}")

            # Make sure these extra features are in the same order we added the fields above!
            feature.setAttributes(feature.attributes() + [map_id])

            # Add a feature in the sink
            sink.addFeature(feature, QgsFeatureSink.FastInsert)

            # Update the progress bar
            feedback.setProgress(int(current * total))

        # Return the results of the algorithm. In this case our only result is
        # the feature sink which contains the processed features, but some
        # algorithms may return multiple feature sinks, calculated numeric
        # statistics, etc. These should all be included in the returned
        # dictionary, with keys matching the feature corresponding parameter
        # or output names.
        return {self.OUTPUT: dest_id}

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

from qgis.PyQt.QtCore import QCoreApplication, QVariant 
from qgis.core import (QgsProcessing,
                       QgsProcessingParameterField,
                       QgsFeatureSink,
                       QgsField,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink)
from qgis import processing


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
    MAP_ID_FIELD = 'map_id'

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
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

    def displayName(self):
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

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'mefdmapbookscripts'

    def shortHelpString(self):
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it..
        """
        return self.tr("Example algorithm short description")

    def initAlgorithm(self, config=None):
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

        # We add a feature sink in which to store our processed features (this
        # usually takes the form of a newly created vector layer when the
        # algorithm is run in QGIS).
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr('Output layer')
            )
        )

    def get_rowcol_from_northings_eastings(self, northings, eastings):
        col = eastings - 14
        row = 92 - northings
        return row, col
    
    def get_rowcol_from_1kid(self, cell_label):
        cell_1kid = cell_label.lower()
        first_letter, number, second_letter = cell_1kid
        number = int(number)
        # first_letter = string.ascii_uppercase[row // 3]
        # number = col // 3
        # second_letter = string.ascii_uppercase[(row % 3) * 3 + col % 3]
        row = (ord(first_letter) - ord('a')) * 3 + (ord(second_letter) - ord('a')) // 3
        col = number * 3 + (ord(second_letter) - ord('a')) % 3
        return row, col

    def get_1kid_from_rowcol(self, row, col):
        first_letter = string.ascii_uppercase[row // 3]
        number = col // 3
        second_letter = string.ascii_uppercase[(row  % 3) * 3 + col % 3]
        grid_label = f"{first_letter}{number}{second_letter}"
        return grid_label
        
    def processAlgorithm(self, parameters, context, feedback):
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

        for current, feature in enumerate(features):
            # Stop the algorithm if cancel button has been clicked
            if feedback.isCanceled():
                break

            feedback.pushInfo(f'working on feature {feature}, {feature.attributes()}')
            feedback.pushInfo(f"layer has fields {[field.name() for field in source.fields()]}")
            field_names = [field.name() for field in source.fields()]
            feature.setFields(source.fields(), False)
            
            northings = int(feature[northings_field])
            eastings = int(feature[eastings_field])
            row, col = self.get_rowcol_from_northings_eastings(northings, eastings)
            map_id = self.get_1kid_from_rowcol(row, col)
                
            # Make sure these extra features are in the same order we added the fields above!
            feature.setAttributes(feature.attributes() + [map_id])
                
            # Add a feature in the sink
            sink.addFeature(feature, QgsFeatureSink.FastInsert)
                
            # Update the progress bar
            feedback.setProgress(int(current * total))
            
            
            
        # To run another Processing algorithm as part of this algorithm, you can use
        # processing.run(...). Make sure you pass the current context and feedback
        # to processing.run to ensure that all temporary layer outputs are available
        # to the executed algorithm, and that the executed algorithm can send feedback
        # reports to the user (and correctly handle cancellation and progress reports!)

        # Return the results of the algorithm. In this case our only result is
        # the feature sink which contains the processed features, but some
        # algorithms may return multiple feature sinks, calculated numeric
        # statistics, etc. These should all be included in the returned
        # dictionary, with keys matching the feature corresponding parameter
        # or output names.
        return {self.OUTPUT: dest_id}


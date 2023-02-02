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
from collections import defaultdict

from qgis.PyQt.QtCore import QCoreApplication, QVariant 
from qgis.core import (QgsProcessing,
                       QgsProcessingParameterField,
                       QgsProcessingParameterExpression,
                       QgsExpression,
                       QgsExpressionContext,
                       QgsExpressionContextUtils,
                       QgsFeatureSink,
                       QgsField,
                       QgsFields,
                       QgsProcessingContext,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink)
from qgis import processing


class AddressRangeTableProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    This takes a grid (polygon layer) and addresses (points layer for now)
    and constructs a mapbook-like table showing what ranges of addresses
    are in what map grid tile.

    All Processing algorithms should extend the QgsProcessingAlgorithm
    class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    GRID = 'GRID'
    ADDRESS = 'ADDRESS'
    OUTPUT = 'OUTPUT'
    NAME_FIELD = 'name_field'
    ADDRESS_FIELD = 'address_field'
    STREET_FIELD = 'street_field'

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return AddressRangeTableProcessingAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'address_range_table'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('Generate address range table')

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
        return self.tr("Given a polygon grid and a layer of address points, "
                       "generate a table showing address ranges per grid "
                       "tile")

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """
        # Grid
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.GRID,
                self.tr('Grid layer'),
                [QgsProcessing.TypeVectorPolygon]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.NAME_FIELD,
                'Choose Grid ID Field',
                '',
                self.GRID))

        # Address points
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.ADDRESS,
                self.tr('Address layer'),
                [QgsProcessing.TypeVectorPoint]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.ADDRESS_FIELD,
                'Choose street number field',
                '',
                self.ADDRESS))
        self.addParameter(
            QgsProcessingParameterExpression(
                self.STREET_FIELD,
                'Define full street name',
                '',
                self.ADDRESS)
            )

        # TODO: add district layer and computation.

        # We add a feature sink in which to store our processed features (this
        # usually takes the form of a newly created vector layer when the
        # algorithm is run in QGIS).
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr('Output layer')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """
        # Retrieve the feature source and sink. The 'dest_id' variable is used
        # to uniquely identify the feature sink, and must be included in the
        # dictionary returned by the processAlgorithm function.
        # Grid info
        grid_source = self.parameterAsSource(
            parameters,
            self.GRID,
            context
        )
        if grid_source is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.GRID)
            )

        name_field = self.parameterAsString(
            parameters,
            self.NAME_FIELD,
            context)

        # Address info
        address_source = self.parameterAsSource(
            parameters,
            self.ADDRESS,
            context
        )
        if address_source is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.ADDRESS)
            )

        street_number_field = self.parameterAsString(
            parameters,
            self.ADDRESS_FIELD,
            context)
        street_name_exp_str = self.parameterAsString(
            parameters,
            self.STREET_FIELD,
            context)
        street_name_expression = QgsExpression(street_name_exp_str)

        # Generate list of fields
        fields = QgsFields()
        fields.append(QgsField("Street name", QVariant.String))
        fields.append(QgsField("Start address", QVariant.String))
        fields.append(QgsField("End address", QVariant.String))
        fields.append(QgsField("Notes", QVariant.String))
        fields.append(QgsField("Map page", QVariant.String))

        # Create sink object
        (sink, dest_id) = self.parameterAsSink(
                parameters,
                self.OUTPUT,
                context,
                fields,
                crs=address_source.sourceCrs())
        if sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT))

        # Send some information to the user
        grid_crs = grid_source.sourceCrs().authid()
        address_crs = address_source.sourceCrs().authid()
        feedback.pushInfo(
            f'Grid CRS is {grid_crs}, Address CRS is {address_crs}')

        grid_features = grid_source.getFeatures()
        address_features = address_source.getFeatures()

        # Generate spatial join dict. Should have form
        # {'INPUT': address_layer,
        #  'PREDICATE': [0], # 0 = intersect, 1 = contain, 2 = equal, 3 = touch,
                             # 4 = overlap, 5 = are within, 6 = cross
        #  'JOIN': grid_layer,
        #  'JOIN_FIELDS':[],
        #  'METHOD': 0,  # This is 0 for 'attributes of each matching feature'
                         # 1 for 'first feature only', 2 for 'largest overlap'
        #  'DISCARD_NONMATCHING': False,
        #  'PREFIX': '',
        #  'OUTPUT': ????}
        spatial_join_dict = {
            "INPUT": parameters[self.ADDRESS],
            "PREDICATE": [0,],  # 0 means intersection
            "JOIN": parameters[self.GRID],
            "JOIN_FIELDS": [],  # no field join, only spatial
            "METHOD": 0,  # generates attribute for each matching feature
            "DISCARD_NONMATCHING": False,
            "PREFIX": "",
            "OUTPUT": "memory:"  # We'll store this in memory and steal it
            # back from the context after the algorithm completes
        }

        # Method from https://gis.stackexchange.com/a/426338/161588
        address_with_grid_info = processing.run(
            "native:joinattributesbylocation",
            spatial_join_dict,
            is_child_algorithm=True,
            context=context, feedback=feedback)["OUTPUT"]
        address_with_grid_layer = QgsProcessingContext.takeResultLayer(
            context, address_with_grid_info)

        address_with_grid_features = address_with_grid_layer.getFeatures()
        # Compute the number of steps to display within the progress bar and
        # get features from source
        total = 100.0 / address_with_grid_layer.featureCount() if address_with_grid_layer.featureCount() else 0

        # Initialize expression context
        expression_context = QgsExpressionContext()
        expression_context.appendScopes(
            QgsExpressionContextUtils.globalProjectLayerScopes(
                address_with_grid_layer))

        # We're going to iterate over attributes and add 'em to a dict of
        # form:
        # {
        #   (street_name, grid_tile): (min_number, max_number),
        #   ...
        # }
        number_grid_dict = defaultdict(set)
        for i, feature in enumerate(address_with_grid_features):
            if feedback.isCanceled():
                break

            # Configure field names for index access to attributes
            feature.setFields(address_with_grid_layer.fields(), False)
            map_id = feature[name_field]
            address_number = feature[street_number_field]

            # Set context and evaluate street name expression
            expression_context.setFeature(feature)
            street_name = street_name_expression.evaluate(expression_context)
            # feedback.pushInfo(f'working on feature {feature}, {feature.attributes()}')
            # feedback.pushInfo(f'fields {[field.name() for field in address_with_grid_layer.fields()]}')
            # feedback.pushInfo(f'map id {map_id}')
            # feedback.pushInfo(f'just computed street name {street_name}')

            # skip streets outside the grid tiles
            if map_id is None:
                continue

            index_tuple = (street_name, map_id)

            # add current address to address set
            number_grid_dict[index_tuple] |= {address_number}

            feedback.setProgress(int(i * total))


        feedback.pushInfo(f"resulting dict {number_grid_dict}")
        for (street_name, map_id), address_number_set in number_grid_dict.items():
            # Order of attributes matters! Make sure this matches order
            # defined above
            # fields.append(QgsField("Street name", QVariant.String))
            # fields.append(QgsField("Start address", QVariant.String))
            # fields.append(QgsField("End address", QVariant.String))
            # fields.append(QgsField("Map page", QVariant.String))
            address_min = min(address_number_set)
            address_max = max(address_number_set)

            # generate notes
            notes = []
            addresses_even = {address for address in address_number_set
                              if address % 2 == 0}
            addresses_odd = {address for address in address_number_set
                             if address % 2 == 1}
            n_even = len(addresses_even)
            n_odd = len(addresses_odd)
            if n_even == 0 and n_odd > 0:
                notes.append("ODD ONLY")
            if n_odd == 0 and n_even > 0:
                notes.append("EVEN ONLY")
            all_grids_for_street = [grid_tile
                                    for street, grid_tile in number_grid_dict.keys()
                                    if street == street_name]
            if len(all_grids_for_street) == 1:
                # intentionally override even/odd only, since all addresses
                # is strictly more informative
                notes = ["ALL ADDRESSES"]
            notes_str = ", ".join(notes)

            attributes = [street_name, address_min, address_max,
                          notes_str, map_id]
            feature.setAttributes(attributes)
            sink.addFeature(feature, QgsFeatureSink.FastInsert)

        # Return the results of the algorithm.
        return {self.OUTPUT: dest_id}


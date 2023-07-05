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
import functools
import re
from collections import defaultdict

from qgis.PyQt.QtCore import QCoreApplication, QVariant  # pylint: disable=import-error
from qgis.core import (  # pylint: disable=import-error
        QgsCoordinateReferenceSystem,
        QgsCoordinateTransform,
        QgsProcessing,
        QgsProcessingParameterField,
        QgsProcessingParameterExpression,
        QgsProject,
        QgsExpression,
        QgsExpressionContext,
        QgsExpressionContextUtils,
        QgsFeature,
        QgsFeatureSink,
        QgsField,
        QgsFields,
        QgsProcessingContext,
        QgsProcessingException,
        QgsProcessingAlgorithm,
        QgsProcessingParameterFeatureSource,
        QgsProcessingParameterFeatureSink)
from qgis import processing  # pylint: disable=import-error


class LandingZoneTableProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    This takes a grid (polygon layer) and landing zones (point layer)
    and constructs a mapbook-like table showing information about each
    landing zone.

    All Processing algorithms should extend the QgsProcessingAlgorithm
    class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    GRID = 'GRID'
    NAME_FIELD = 'name_field'

    LANDING_ZONE = "LANDING_ZONE"
    LZ_NAME_FIELD = 'lz_name_field'

    DISTRICTS = "DISTRICTS"
    DISTRICT_NAME_FIELD = "district_name_field"

    OUTPUT = 'OUTPUT'

    # CRS for conventional lat/long transformation
    LATLNG_CRS = QgsCoordinateReferenceSystem("EPSG:4326")

    def __init__(self):
        self.sink = None
        self.output_fields = None
        self.address_to_street_name_map = {}
        super().__init__()

    def tr(self, string):  # pylint: disable=invalid-name
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):  # pylint: disable=invalid-name
        """Initialize instance (used by QGIS processing toolbox)
        """
        return LandingZoneTableProcessingAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'landing_zone_range_table'

    def displayName(self):  # pylint: disable=invalid-name
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('Generate landing zone table')

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
        return self.tr("Given a landing zone layer, a grid, and district "
                       "polygons, generate a useful table of landing zone "
                       "information")

    def initAlgorithm(self, config=None):  # pylint: disable=invalid-name
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

        # Landing zone points
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.LANDING_ZONE,
                self.tr('Landing zones'),
                [QgsProcessing.TypeVectorPoint]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.LZ_NAME_FIELD,
                'Choose name field',
                '',
                self.LANDING_ZONE))

        # Districts
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.DISTRICTS,
                self.tr('Districts layer'),
                [QgsProcessing.TypeVectorPolygon]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.DISTRICT_NAME_FIELD,
                'Field giving the name of the district',
                '',
                self.DISTRICTS))

        # We add a feature sink in which to store our processed features (this
        # usually takes the form of a newly created vector layer when the
        # algorithm is run in QGIS).
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr('Output layer')
            )
        )

    def _get_grid_inputs(self, parameters, context):
        """Get grid-related inputs
        """
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

        return grid_source, name_field

    def _get_lz_inputs(self, parameters, context):
        """Get landing zone-related inputs
        """
        # Address info
        lz_source = self.parameterAsSource(
            parameters,
            self.LANDING_ZONE,
            context
        )
        if lz_source is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.LANDING_ZONE)
            )

        name_field = self.parameterAsString(
            parameters,
            self.LZ_NAME_FIELD,
            context)

        return (lz_source, name_field)

    def _get_district_inputs(self, parameters, context):
        """Get district-related inputs
        """
        district_source = self.parameterAsSource(
            parameters,
            self.DISTRICTS,
            context
        )
        if district_source is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.DISTRICTS)
            )

        name_field = self.parameterAsString(
            parameters,
            self.DISTRICT_NAME_FIELD,
            context)

        return district_source, name_field

    def _configure_sink(self, parameters, context, crs):
        """Configure output layer
        """
        # Generate list of fields
        fields = QgsFields()
        fields.append(QgsField("Landing zone name", QVariant.String))
        fields.append(QgsField("District", QVariant.String))
        fields.append(QgsField("Map page", QVariant.String))
        fields.append(QgsField("Latitude", QVariant.String))
        fields.append(QgsField("Longitude", QVariant.String))

        # Create sink object
        (self.sink, dest_id) = self.parameterAsSink(
                parameters,
                self.OUTPUT,
                context,
                fields,
                crs=crs)
        if self.sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT))

        self.output_fields = fields
        return {self.OUTPUT: dest_id}

    def _intersection_join(self, input1, input2, context, feedback):
        """Run a spatial join with two inputs, storing result in memory
        """
        feedback.pushInfo("About to try an intersection with "
                          f"{input1} (type {type(input1)}) x {input2}")
        intersection_dict = {
                'INPUT': input1,
                'OVERLAY':input2,
                'INPUT_FIELDS':[],
                'OVERLAY_FIELDS':[],
                'OVERLAY_FIELDS_PREFIX':'',
                'OUTPUT':"TEMPORARY_OUTPUT"}  # We'll store this in memory and steal it
                                    # back from the context after the algorithm completes
        # Method from https://gis.stackexchange.com/a/426338/161588
        intersect_result = processing.run("native:intersection",
                       intersection_dict,
                       is_child_algorithm=True,
                       context=context, feedback=feedback)["OUTPUT"]
        intersect_layer = QgsProcessingContext.takeResultLayer(
            context, intersect_result)

        return intersect_layer

    def _perform_joins(self, starting_layer, parameters, context, feedback):
        """Perform necessary joins

        Inputs:
        - starting layer: the landing zone layer to join against
        - parameters: the processing algorithm parameters, from which the
          method will retrieve other layers needed for the joins
        - context: the processing context, used for the actual joins
        - feedback: the feedback for the user, useful for printing messages
          or just to pass to the joins

        Returns:
        - a layer of the starting layer, joined with districts and grid

        We're going to do several spatial joins:
        1. address layer -> district boundaries
           - will add district_f attribute like 'MEFD' or '13'
        2. address layer -> grid cells
           - will add a whole bunch of attributes, but we only care about the
             grid ID one, which the user will choose from a dropdown.
        """
        # Spatial join #1: address layer -> district boundaries
        lz_with_district_layer = self._intersection_join(
                starting_layer,
                parameters[self.DISTRICTS], context, feedback)

        # Spatial join #2: address layer -> grid cells
        lz_with_grid_layer = self._intersection_join(
                 lz_with_district_layer, parameters[self.GRID],
                 context, feedback)

        return lz_with_grid_layer

    def processAlgorithm(self, parameters, context, feedback):  # pylint: disable=invalid-name
        """
        Here is where the processing itself takes place.
        """
        # Retrieve the feature source and sink. The 'dest_id' variable is used
        # to uniquely identify the feature sink, and must be included in the
        # dictionary returned by the processAlgorithm function.
        # Grid info
        grid_source, name_field = self._get_grid_inputs(
                parameters, context)

        (lz_source,
         lz_name_field) = self._get_lz_inputs(
                parameters, context)

        _, district_field = self._get_district_inputs(
                parameters, context)

        dest_ids = self._configure_sink(parameters, context, lz_source.sourceCrs())

        # Send some information to the user
        grid_crs = grid_source.sourceCrs().authid()
        lz_crs = lz_source.sourceCrs()
        feedback.pushInfo(
            f'Grid CRS is {grid_crs}, landing zone CRS is {lz_crs.authid()}')

        lz_with_grid_layer = self._perform_joins(
                parameters[self.LANDING_ZONE],
                parameters, context, feedback)

        for i, feature in enumerate(lz_with_grid_layer.getFeatures()):
            if feedback.isCanceled():
                break

            feature.setFields(lz_with_grid_layer.fields(), False)
            map_id = feature[name_field]
            district = feature[district_field]

            lz_name = feature[lz_name_field]
            geometry = feature.geometry()
            # Convert geometry to lat/long (instead of meters north/east of
            # a specific point)
            transform = QgsCoordinateTransform(
                lz_crs, self.LATLNG_CRS, QgsProject.instance())
            geometry.transform(transform)
            lng, lat = (geometry.asPoint().x(),
                          geometry.asPoint().y())

            lat_str = f"{int(lat)}° {(abs(lat) % 1) * 60:.4f}"
            lng_str = f"{int(lng)}° {(abs(lng) % 1) * 60:.4f}"

            feedback.pushInfo(f"{lng} -> {lng_str}, {lat} -> {lat_str}")

            attributes = [lz_name, district, map_id, lat_str, lng_str]
            feature = QgsFeature(self.output_fields)
            feature.setAttributes(attributes)
            self.sink.addFeature(feature, QgsFeatureSink.FastInsert)

        # Return the results of the algorithm.
        return dest_ids

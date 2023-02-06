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
import itertools
import numpy as np
import string
from collections import defaultdict

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsCoordinateReferenceSystem,
                       QgsCoordinateTransform,
                       QgsDistanceArea,
                       QgsExpression,
                       QgsExpressionContext,
                       QgsExpressionContextUtils,
                       QgsFeature,
                       QgsFeatureSink,
                       QgsField,
                       QgsFields,
                       QgsGeometry,
                       QgsPoint,
                       QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingContext,
                       QgsProcessingException,
                       QgsProcessingParameterExpression,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterField,
                       QgsProject,
                       QgsUnitTypes,
                       QgsWkbTypes,
                       )
from qgis import processing


class ReconcileHydrantsProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    This takes two layers of hydrant point data (including IDs) and creates:
        1. a mapping between the layers of hydrants the algorithm thinks
        are the same
        2. a layer of points highlighting where the two datasets differ
        significantly

    All Processing algorithms should extend the QgsProcessingAlgorithm
    class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    HYDRANTS1 = 'HYDRANTS1'
    HYDRANTS2 = 'HYDRANTS2'
    HYDRANTS1_ID_FIELD = 'hydrant1_address_field'
    HYDRANTS2_ID_FIELD = 'hydrant2_address_field'
    CORRESPONDENCE = "CORRESPONDENCE"
    DISTINCT = "DISTINCT"
    DISTANCE_CRS = QgsCoordinateReferenceSystem("EPSG:2285")

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return ReconcileHydrantsProcessingAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'reconcile_hydrants'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('Reconcile hydrant data sets')

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
        return self.tr("Given two layers of hydrants, create a mapping "
                       "between them and highlight differences")

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """
        # Layer 1
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.HYDRANTS1,
                self.tr("First hydrant layer"),
                [QgsProcessing.TypeVectorPoint]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.HYDRANTS1_ID_FIELD,
                "Choose this layer's ID Field",
                '',
                self.HYDRANTS1))

        # Layer 2
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.HYDRANTS2,
                self.tr("Second hydrant layer"),
                [QgsProcessing.TypeVectorPoint]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.HYDRANTS2_ID_FIELD,
                "Choose this layer's ID Field",
                '',
                self.HYDRANTS2))

        # We add a feature sink in which to store our processed features (this
        # usually takes the form of a newly created vector layer when the
        # algorithm is run in QGIS).
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.CORRESPONDENCE,
                self.tr('Correspondence between hydrant datasets'),
                QgsProcessing.TypeVectorLine
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.DISTINCT,
                self.tr('Distinctive hydrants'),
            )
        )


    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """
        # Retrieve the feature source and sink. The 'dest_id' variable is used
        # to uniquely identify the feature sink, and must be included in the
        # dictionary returned by the processAlgorithm function.
        # Hydrant 1 info
        hydrant1_source = self.parameterAsSource(
            parameters,
            self.HYDRANTS1,
            context
        )
        if hydrant1_source is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.HYDRANTS1)
            )

        hydrant1_id_field = self.parameterAsString(
            parameters,
            self.HYDRANTS1_ID_FIELD,
            context)

        # Hydrant 2
        hydrant2_source = self.parameterAsSource(
            parameters,
            self.HYDRANTS2,
            context
        )
        if hydrant2_source is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.HYDRANTS2)
            )

        hydrant2_id_field = self.parameterAsString(
            parameters,
            self.HYDRANTS2_ID_FIELD,
            context)

        # Generate list of fields
        correspondence_fields = QgsFields()
        correspondence_fields.append(QgsField("Source 1 id", QVariant.String))
        correspondence_fields.append(QgsField("Source 2 id", QVariant.String))
        correspondence_fields.append(QgsField("Distance", QVariant.Double))

        distinct_fields = QgsFields()
        distinct_fields.append(QgsField("Source data set", QVariant.String))
        distinct_fields.append(QgsField("Source hydrant id",
                                        QVariant.String))
        distinct_fields.append(QgsField("Nearest hydrant in other dataset",
                                        QVariant.Double))

        # Create sink object
        (correspondence_sink, correspondence_dest_id) = self.parameterAsSink(
                parameters,
                self.CORRESPONDENCE,
                context,
                correspondence_fields,
                QgsWkbTypes.LineString,
                crs=self.DISTANCE_CRS)
        if correspondence_sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters,
                                                               self.CORRESPONDENCE))

        # (sink, dest_id) = self.parameterAsSink(
        #         parameters,
        #         self.DISTINCT,
        #         context,
        #         distinct_fields,
        #         crs=self.DISTANCE_CRS)
        # if sink is None:
        #     raise QgsProcessingException(self.invalidSinkError(parameters,
        #                                                        self.DISTINCT))

        # Send some information to the user
        hydrant1_crs = hydrant1_source.sourceCrs()
        hydrant2_crs = hydrant2_source.sourceCrs()
        feedback.pushInfo(
            f"Hydrant data 1 CRS is {hydrant1_crs.authid()}, "
            f"Hydrant data 2 CRS is {hydrant2_crs.authid()}, "
            f"calculating distances in {self.DISTANCE_CRS.authid()}")

        hydrant1_features = list(hydrant1_source.getFeatures())
        hydrant2_features = list(hydrant2_source.getFeatures())

        n1 = len(hydrant1_features)
        n2 = len(hydrant2_features)

        # Step 1: pairwise distances
        # distances[i, j] is distance from hydrant i in first datset to
        # hydrant j in second
        distances = np.zeros((n1, n2))
        d = QgsDistanceArea()
        d.setSourceCrs(self.DISTANCE_CRS,
                       QgsProject.instance().transformContext())
        d.setEllipsoid(QgsProject.instance().ellipsoid())

        # Cache transformed geometries
        hydrant1_geoms = {}  # will be i: geometry
        hydrant2_geoms = {}

        # Hardcoding distance calculation CRS for now, to Washington North
        transform1 = QgsCoordinateTransform(
            hydrant1_crs, self.DISTANCE_CRS, QgsProject.instance())
        transform2 = QgsCoordinateTransform(
            hydrant2_crs, self.DISTANCE_CRS, QgsProject.instance())
        feedback.pushInfo(f"Calculating {n1} x {n2} = {n1 * n2} distances")
        total = 100.0 / (n1 * n2)
        for counter, (i, j) in enumerate(itertools.product(range(n1),
                                                           range(n2))):
            if feedback.isCanceled():
                break
            if i in hydrant1_geoms:
                g1 = hydrant1_geoms[i]
            else:
                hydrant1 = hydrant1_features[i]
                # Note: transform transforms in place so we need a separate
                # geometry variable here
                g1 = hydrant1.geometry()
                g1.transform(transform1)
                hydrant1_geoms[i] = g1

            if j in hydrant2_geoms:
                g2 = hydrant2_geoms[j]
            else:
                hydrant2 = hydrant2_features[j]
                g2 = hydrant2.geometry()
                g2.transform(transform2)
                hydrant2_geoms[j] = g2

            # NOTE: distance in EPSG:2285 is in feet, so that's what we get
            # here, but convert anyway, just to be safe
            distance = d.measureLine(g1.asPoint(), g2.asPoint())
            distance_feet = d.convertLengthMeasurement(
                distance, QgsUnitTypes.DistanceUnit.DistanceFeet)
            distances[i, j] = distance_feet
            feedback.setProgress(int(counter * total))

        feedback.pushInfo(f"Done calculating distances: {distances}")

        # Step 2: find hydrants that are mutually closest to each other,
        # flag distinct hydrants
        min_in_col = distances == distances.min(axis=0)
        min_in_row = (distances.T == distances.min(axis=1)).T

        # Matching hydrants are mutually closest to each other...
        mutually_close = min_in_col & min_in_row
        # ...and are separated by a distance less than x
        MAX_MATCH_DISTANCE = 250  # IDK, seems legit?
        within_buffer = distances < MAX_MATCH_DISTANCE
        matching_hydrants = mutually_close & within_buffer

        # 2a: matching hydrants:
        pairs = list(zip(*matching_hydrants.nonzero()))  # (i, j) pairs
        feedback.pushInfo(f"Building matching hydrant layer ({len(pairs)} "
                          " total)")
        total = 100.0 / (n1 * n2)
        feedback.pushInfo(f"fields {[f for f in correspondence_fields]}")
        for counter, (i, j) in enumerate(pairs):
            if feedback.isCanceled():
                break
            hydrant1 = hydrant1_features[i]
            # All hydrants should be in the cache at this point; stop
            # explicitly checking
            hydrant1_point = hydrant1_geoms[i].asPoint()
            hydrant1.setFields(hydrant1_source.fields(), False)
            hydrant1_id = hydrant1[hydrant1_id_field]

            hydrant2 = hydrant2_features[j]
            hydrant2_point = hydrant2_geoms[j].asPoint()
            hydrant2.setFields(hydrant2_source.fields(), False)
            hydrant2_id = hydrant2[hydrant2_id_field]

            distance = distances[i, j]

            # Remember, order here must match the field order.
            # For reasons I don't understand, passing `distance` as it is
            # rather than casting it to a string causes an error when
            # adding the feature to the layer. So convert to a string
            # first, and it'll get converted back to a number when added.
            attributes = [str(hydrant1_id), str(hydrant2_id), str(distance)]
            feedback.pushInfo(f"{attributes}")

            # Generate geometry: a line connecting the two points
            match_geometry = QgsGeometry.fromPolyline(
                [QgsPoint(hydrant1_point), QgsPoint(hydrant2_point)]);

            feature = QgsFeature(correspondence_fields)
            feature.setGeometry(match_geometry)
            feature.setAttributes(attributes)
            feedback.pushInfo(f"feature is {feature}, fields {[f for f in feature.fields()]}, attributes {feature.attributes()}, geometry {feature.geometry()}")
            correspondence_sink.addFeature(feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(counter * total))

        feedback.pushInfo("Done with matching hydrant layer")


        # 2b: non-matching hydrants, that is, any row or column without a
        # True in `matching_hydrants`
        hydrant1_nonmatches = (matching_hydrants.sum(axis=1) == 0).nonzero()
        hydrant2_nonmatches = (matching_hydrants.sum(axis=0) == 0).nonzero()
            # distinct_fields = QgsFields()
            # distinct_fields.append(QgsField("Source data set", QVariant.String))
            # distinct_fields.append(QgsField("Source hydrant id",
            #                                 QVariant.String))
            # distinct_fields.append(QgsField("Nearest hydrant in other dataset",
            #                                 QVariant.Double))
        feedback.pushInfo("Processing distinct hydrants: "
                          f"{len(hydrant1_nonmatches)} in dataset 1, "
                          f"{len(hydrant2_nonmatches)} in dataset 2")

        return {self.CORRESPONDENCE: correspondence_dest_id}
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
            notes_str = ", ".join(notes)
            attributes = [street_name, address_min, address_max,
                          notes_str, map_id]
            feature.setAttributes(attributes)
            sink.addFeature(feature, QgsFeatureSink.FastInsert)

        # Return the results of the algorithm.
        return {self.OUTPUT: dest_id}

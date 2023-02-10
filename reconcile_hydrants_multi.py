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
import networkx as nx
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
                       QgsProcessingParameterMultipleLayers,
                       QgsProject,
                       QgsUnitTypes,
                       QgsWkbTypes,
                       )
from qgis import processing


class ReconcileHydrantsMultiProcessingAlgorithm(QgsProcessingAlgorithm):
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
    # Hardcode the CRS we'll use for calculating distances: Washington
    # North
    DISTANCE_CRS = QgsCoordinateReferenceSystem("EPSG:2285")
    MAX_MATCH_DISTANCE = 250  # IDK, seems legit?

    HYDRANTS = 'HYDRANTS'
    # HYDRANTS1 = 'HYDRANTS1'
    # HYDRANTS2 = 'HYDRANTS2'
    # TODO: figure out what to do re id fields, can I select one for each?
    # HYDRANTS1_ID_FIELD = 'hydrant1_address_field'
    # HYDRANTS2_ID_FIELD = 'hydrant2_address_field'

    # Output layers
    CORRESPONDENCE = "CORRESPONDENCE"
    DISTINCT = "DISTINCT"

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return ReconcileHydrantsMultiProcessingAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'reconcile_hydrants_multi'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('Reconcile 3+ hydrant data sets')

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
        return self.tr("Given 2+ layers of hydrants, create a mapping "
                       "between them and highlight differences")

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """
        # All layers in one source! Wooooooo
        self.addParameter(
                QgsProcessingParameterMultipleLayers(
                    name=self.HYDRANTS,
                    description=self.tr("Select hydrant layers"),
                    layerType=QgsProcessing.TypeVectorPoint))

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
                self.tr('Distinct hydrants'),
            )
        )

    def getPairwiseCorrespondence(
            self,
            dataset1, dataset2, correspondence_distance_ft, feedback):
        """Get correspondence between hydrants in two layers.

        Input:
        - dataset1: a QgsProcessingFeatureSource of points
        - dataset2: another QgsProcessingFeatureSource of points
        - correspondence_distance_ft: max distance (in feet) for two
          hydrants to be considered matching.

        Output:
        - a list of tuples of indices like [(i1, j1), (i2, j2)...], meaning
          that dataset1[i1] and dataset2[j1] are considered matching.
        """
        feedback.pushInfo(
                "Getting pairwise correspondence between "
                f"{dataset1.sourceName()} and {dataset2.sourceName()}..."
                )
        hydrant1_crs = dataset1.sourceCrs()
        hydrant2_crs = dataset2.sourceCrs()
        feedback.pushInfo(
            f"Hydrant data 1 CRS is {hydrant1_crs.authid()}, "
            f"Hydrant data 2 CRS is {hydrant2_crs.authid()}, "
            f"calculating distances in {self.DISTANCE_CRS.authid()}")

        hydrant1_features = list(dataset1.getFeatures())
        hydrant2_features = list(dataset2.getFeatures())

        n1 = len(hydrant1_features)
        n2 = len(hydrant2_features)

        # Step 1: pairwise distances
        # distances[i, j] is distance from hydrant i in first dataset to
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

        feedback.pushInfo(f"Done calculating distances")

        # Step 2: find hydrants that are mutually closest to each other,
        # flag distinct hydrants
        min_in_col = distances.min(axis=0)
        is_min_in_col = distances == min_in_col
        min_in_row = distances.min(axis=1)
        is_min_in_row = (distances.T == min_in_row).T

        # Matching hydrants are mutually closest to each other...
        mutually_close = is_min_in_col & is_min_in_row
        # ...and are separated by a distance less than x (this is feet)
        within_buffer = distances < correspondence_distance_ft
        matching_hydrants = mutually_close & within_buffer

        # 2a: matching hydrants:
        pairs = list(zip(*matching_hydrants.nonzero()))  # (i, j) pairs
        return pairs


    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """
        # Retrieve the feature source and sink. The 'dest_id' variable is used
        # to uniquely identify the feature sink, and must be included in the
        # dictionary returned by the processAlgorithm function.
        # All hydrant data
        hydrants_sources = self.parameterAsLayerList(
                parameters, self.HYDRANTS, context)
        feedback.pushInfo(f"hydrant sources {hydrants_sources}")
        if hydrants_sources is None:
            raise QgsProcessingException(
                    self.invalidSourceError(parameters, self.HYDRANTS)
                    )
        if len(hydrants_sources) < 2:
            # TODO find correct error to raise here
            raise Exception("Invalid number of hydrant layers – got "
                    f"{len(hydrants_sources)} but expected 2+")

        # We're going to construct a graph representing which hydrants
        # "correspond" to which across different data sets. The keys in
        # this graph will be (dataset_name, index_in_dataset), and an edge
        # between (dataset_1, i) and (dataset_2, j) indicates that hydrant
        # i within dataset_1 "matches" with hydrant j in dataset_2. That
        # means two things:
        #   - They are both each other's closest neighbor across datasets
        #   - They are within MAX_MATCH_DISTANCE feet of each other

        # Before we can do anything fancy, we need to populate the graph
        # with its vertices
        G = nx.Graph()
        for dataset in hydrants_sources:
            name = dataset.name()
            feedback.pushInfo(f"name {dataset.name()}, type {dataset.type()}, source {dataset.source()}")
            n = dataset.featureCount()
            nodes_to_add = [(name, i) for i in range(n)]
            G.add_nodes_from(nodes_to_add)

        feedback.pushInfo(f"{len(G)} nodes before edges")

        # Iterate over every pair of data sources and find connections. Use
        # those to populate a graph
        for dataset1, dataset2 in itertools.combinations(hydrants_sources, 2):
            pairs = self.getPairwiseCorrespondence(dataset1, dataset2,
                    self.MAX_MATCH_DISTANCE, feedback)
            # Add dataset names to pairs
            name1 = dataset1.name()
            name2 = dataset2.name()
            pairs_with_names = [((name1, i), (name2, j)) for i, j in pairs]
            G.add_edges_from(pairs_with_names)
            feedback.pushInfo(f"{len(G)} nodes after correspondence "
                    f"between {name1} and {name2}, {len(G.edges())} edges")

        # TODO: look at graph's connected components and add each to either
        # distinct or a correspondence layer. (correspondence should
        # probably have each pair in lines?? idk)
        for component in nx.connected_components(G):
            # Each component is just a set of nodes
            feedback.pushInfo(f"component {component} has {len(component)} nodes!")
            # TODO: error checking: no component should have multiple nodes
            # from the same original data set.

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

        (distinct_sink, distinct_dest_id) = self.parameterAsSink(
                parameters,
                self.DISTINCT,
                context,
                distinct_fields,
                QgsWkbTypes.Point,
                crs=self.DISTANCE_CRS)
        if distinct_sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters,
                                                               self.DISTINCT))


        # Send some information to the user
        feedback.pushInfo(f"Building matching hydrant layer ({len(pairs)} "
                          " total)")
        total = 100.0 / (n1 * n2)
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

            # Generate geometry: a line connecting the two points
            match_geometry = QgsGeometry.fromPolyline(
                [QgsPoint(hydrant1_point), QgsPoint(hydrant2_point)]);

            feature = QgsFeature(correspondence_fields)
            feature.setGeometry(match_geometry)
            feature.setAttributes(attributes)
            correspondence_sink.addFeature(feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(counter * total))

        feedback.pushInfo("Done with matching hydrant layer")


        # 2b: non-matching hydrants, that is, any row or column without a
        # True in `matching_hydrants`
        hydrant1_nonmatches = (matching_hydrants.sum(axis=1) == 0
                               ).nonzero()[0]
        hydrant2_nonmatches = (matching_hydrants.sum(axis=0) == 0
                               ).nonzero()[0]
        feedback.pushInfo("Processing distinct hydrants: "
                          f"{len(hydrant1_nonmatches)} in dataset 1, "
                          f"{len(hydrant2_nonmatches)} in dataset 2")

        total = 100.0 / (len(hydrant1_nonmatches) + len(hydrant2_nonmatches))
        hydrant1_source_name = hydrant1_source.sourceName()
        for counter, i in enumerate(hydrant1_nonmatches):
            if feedback.isCanceled():
                break

            hydrant1 = hydrant1_features[i]
            hydrant1_point = hydrant1_geoms[i]
            hydrant1.setFields(hydrant1_source.fields(), False)
            hydrant1_id = hydrant1[hydrant1_id_field]

            nearest_distance = min_in_row[i]

            attributes = [hydrant1_source_name, hydrant1_id,
                          str(nearest_distance)]

            feature = QgsFeature(correspondence_fields)
            feature.setGeometry(hydrant1_point)
            feature.setAttributes(attributes)
            distinct_sink.addFeature(feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(counter * total))


        hydrant2_source_name = hydrant2_source.sourceName()
        for counter, j in enumerate(hydrant2_nonmatches):
            if feedback.isCanceled():
                break
            counter = counter + len(hydrant2_nonmatches)
            hydrant2 = hydrant2_features[j]
            hydrant2_point = hydrant2_geoms[j]
            hydrant2.setFields(hydrant2_source.fields(), False)
            hydrant2_id = hydrant2[hydrant2_id_field]

            nearest_distance = min_in_col[j]

            attributes = [hydrant2_source_name, hydrant2_id,
                          str(nearest_distance)]

            feature = QgsFeature(correspondence_fields)
            feature.setGeometry(hydrant2_point)
            feature.setAttributes(attributes)
            distinct_sink.addFeature(feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(counter * total))

        feedback.pushInfo("Done processing distinct hydrants")

        return {self.CORRESPONDENCE: correspondence_dest_id,
                self.DISTINCT: distinct_dest_id}

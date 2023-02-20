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
from collections import Counter
import itertools

import networkx as nx
import numpy as np

from qgis.PyQt.QtCore import (  # pylint: disable=import-error
        QCoreApplication,
        QVariant)
from qgis.core import (  # pylint: disable=import-error
        QgsCoordinateReferenceSystem,
        QgsCoordinateTransform,
        QgsDistanceArea,
        QgsFeature,
        QgsFeatureSink,
        QgsField,
        QgsFields,
        QgsGeometry,
        QgsPoint,
        QgsProcessing,
        QgsProcessingAlgorithm,
        QgsProcessingException,
        QgsProcessingParameterFeatureSink,
        QgsProcessingParameterMultipleLayers,
        QgsProject,
        QgsUnitTypes,
        QgsWkbTypes,
        )


class ReconcileHydrantsMultiProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    This takes two or more layers of hydrant point data and creates:
        1. a mapping between the layers of hydrants the algorithm thinks
        are the same
        2. a layer of points highlighting where the two datasets differ
        significantly
        3. a "proposed hydrant location" layer with best guesses as to
        final hydrant locations, with the location chosen based on the
        first (alphabetical) matching data set.
    """
    # Hardcode the CRS we'll use for calculating distances: Washington
    # North
    DISTANCE_CRS = QgsCoordinateReferenceSystem("EPSG:2285")
    MAX_MATCH_DISTANCE = 250  # IDK, seems legit?

    HYDRANTS = 'HYDRANTS'
    # CONFIRMED_NEGATIVES = 'CONFIRMED_NEGATIVES'

    # Output layers
    CORRESPONDENCE = "CORRESPONDENCE"
    DISTINCT = "DISTINCT"
    PROPOSED = "PROPOSED"

    hydrant_geometries = {}  # Will be dict mapping dataset name ->
                             # dict mapping index i -> a qgs geometry

    def __init__(self):
        self.distinct_fields = None
        self.correspondence_fields = None
        self.proposed_fields = None

        self.distinct_sink = None
        self.correspondence_sink = None
        self.proposed_sink = None

        super().__init__()

    def tr(self, string):  # pylint: disable=invalid-name
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):  # pylint: disable=invalid-name
        """
        Returns an instance of this algorithm, used by QGIS processing
        toolbox
        """
        return ReconcileHydrantsMultiProcessingAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'reconcile_hydrants'

    def displayName(self):  # pylint: disable=invalid-name
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
        return self.tr("Given 2+ layers of hydrants, create a mapping "
                       "between them, highlight differences, and generate "
                       "a proposed merged hydrant dataset")

    def initAlgorithm(self, config=None):  # pylint: disable=invalid-name
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

        # self.addParameter(
        #         QgsProcessingParameterMultipleLayers(
        #             name=self.CONFIRMED_NEGATIVES,
        #             description=self.tr("Layers with confirmed non-hydrants"),
        #             layerType=QgsProcessing.TypeVectorPoint))

        # Add feature sinks for output layers
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
                QgsProcessing.TypeVectorPoint
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.PROPOSED,
                self.tr('Proposed hydrant locations'),
                QgsProcessing.TypeVectorPoint
            )
        )

    def _populate_geometries(self, hydrant_sources):
        """Populate hydrant geometries in a single CRS.
        """
        for hydrant_source in hydrant_sources:
            crs = hydrant_source.sourceCrs()
            name = hydrant_source.name()

            hydrant_features = list(hydrant_source.getFeatures())

            feature_count = len(hydrant_features)

            # Cache transformed geometries
            self.hydrant_geometries[name] = {}  # will be i: geometry

            transform = QgsCoordinateTransform(
                crs, self.DISTANCE_CRS, QgsProject.instance())
            for i in range(feature_count):
                hydrant = hydrant_features[i]
                # Note: transform transforms in place so we need a separate
                # geometry variable here
                geometry = hydrant.geometry()
                geometry.transform(transform)
                self.hydrant_geometries[name][i] = geometry

    def _get_pairwise_distances(self, dataset1, dataset2, feedback):
        """Construct pairwise distance matrix and return it.
        """
        name1 = dataset1.name()
        name2 = dataset2.name()

        assert name1 in self.hydrant_geometries
        assert name2 in self.hydrant_geometries

        hydrant1_count = len(self.hydrant_geometries[name1])
        hydrant2_count = len(self.hydrant_geometries[name2])

        # Step 1: pairwise distances
        # distances[i, j] is distance from hydrant i in first dataset to
        # hydrant j in second
        distances = np.zeros((hydrant1_count, hydrant2_count))
        distance_calculator = QgsDistanceArea()
        distance_calculator.setSourceCrs(
                self.DISTANCE_CRS, QgsProject.instance().transformContext())
        distance_calculator.setEllipsoid(QgsProject.instance().ellipsoid())

        feedback.pushInfo(f"    Calculating {hydrant1_count} x "
                          f"{hydrant2_count} "
                          f"= {hydrant1_count * hydrant2_count} distances")
        total = 100.0 / (hydrant1_count * hydrant2_count)
        for counter, (i, j) in enumerate(
                itertools.product(range(hydrant1_count),
                                  range(hydrant2_count))):
            if feedback.isCanceled():
                break

            # Get transformed geometries
            geom1 = self.hydrant_geometries[name1][i]
            geom2 = self.hydrant_geometries[name2][j]

            # NOTE: distance in EPSG:2285 is in feet, so that's what we get
            # here, but convert anyway, just to be safe
            distance = distance_calculator.measureLine(geom1.asPoint(),
                                                       geom2.asPoint())
            distance_feet = distance_calculator.convertLengthMeasurement(
                distance, QgsUnitTypes.DistanceUnit.DistanceFeet)
            distances[i, j] = distance_feet
            feedback.setProgress(int(counter * total))

        return distances

    def _get_pairwise_correspondence(
            self,
            dataset1, dataset2, correspondence_distance_ft, feedback):
        """Get correspondence between hydrants in two layers.

        Input:
        - dataset1: a QgsProcessingFeatureSource of points
        - dataset2: another QgsProcessingFeatureSource of points
        - correspondence_distance_ft: max distance (in feet) for two
          hydrants to be considered matching.

        Output:
        - a list of tuples of indices like [(i1, j1, d1), (i2, j2, d2)...],
          meaning that dataset1[i1] and dataset2[j1] are considered
          matching, with a distance of d1, and so on.
        """
        feedback.pushInfo(
                "Comparing "
                f"{dataset1.sourceName()} and {dataset2.sourceName()} "
                f"using {self.DISTANCE_CRS.authid()}"
                )
        distances = self._get_pairwise_distances(dataset1, dataset2,
                                                feedback)

        # Step 2: find hydrants that are mutually closest to each other
        is_min_in_col = distances == distances.min(axis=0)
        is_min_in_row = (distances.T == distances.min(axis=1)).T

        # Matching hydrants are mutually closest to each other...
        mutually_close = is_min_in_col & is_min_in_row
        # ...and are separated by a distance less than x feet
        within_buffer = distances < correspondence_distance_ft
        matching_hydrants = mutually_close & within_buffer

        pairs = list(zip(*matching_hydrants.nonzero()))  # (i, j) pairs
        pairs_with_distances = [(i, j, distances[i, j]) for i, j in pairs]
        return pairs_with_distances

    def _generate_hydrant_graph(self, hydrant_sources, feedback):
        """
        We're going to construct a graph representing which hydrants
        "correspond" to which across different data sets. The keys in
        this graph will be (dataset_name, index_in_dataset), and an edge
        between (dataset_1, i) and (dataset_2, j) indicates that hydrant
        i within dataset_1 "matches" with hydrant j in dataset_2. That
        means two things:
          - They are both each other's closest neighbor across datasets
          - They are within MAX_MATCH_DISTANCE feet of each other
        """
        hydrant_graph = nx.Graph()
        for dataset in hydrant_sources:
            name = dataset.name()
            feature_count = dataset.featureCount()
            nodes_to_add = [(name, i) for i in range(feature_count)]
            hydrant_graph.add_nodes_from(nodes_to_add)

        # Iterate over every pair of data sources and find connections. Use
        # those to populate a graph
        for dataset1, dataset2 in itertools.combinations(hydrant_sources, 2):
            pairs = self._get_pairwise_correspondence(dataset1, dataset2,
                    self.MAX_MATCH_DISTANCE, feedback)
            # Add dataset names to pairs
            name1 = dataset1.name()
            name2 = dataset2.name()
            pairs_with_names = [((name1, i), (name2, j),
                                 {"distance": d})
                                for i, j, d in pairs]
            hydrant_graph.add_edges_from(pairs_with_names)
            feedback.pushInfo(
                f"    Hydrant graph has {len(hydrant_graph)} nodes "
                f"and {len(hydrant_graph.edges())} edges")

        return hydrant_graph

    def _generate_merged_fields(self, hydrant_sources):
        """Generate merged fields for all hydrant sources

        Input:
        - list of hydrant sources

        Output:
        - merged_fields: list of fields from all sources, prepended with
          source name
        - index_range_dict: indexes in merged_fields for each dataset
          (indexed by name)
        """
        # First, generate a merged list of fields from all datasets
        dataset_names = []
        merged_fields = []
        all_dataset_fields = []
        for dataset in hydrant_sources:
            dataset_name = dataset.name()
            dataset_names.append(dataset_name)
            dataset_fields = []
            for field in dataset.fields():
                field.setName(dataset_name + "-" + field.name())
                dataset_fields.append(field)
            merged_fields.extend(dataset_fields)
            all_dataset_fields.append(dataset_fields)

        # Make it easier to slice into a specific dataset's fields
        start_index = 0
        index_range_dict = {}
        for name, fields in zip(dataset_names, all_dataset_fields):
            index_range_dict[name] = (start_index,
                                      start_index + len(fields))
            start_index += len(fields)

        return merged_fields, index_range_dict

    def _get_extra_attributes(self, name_feature_tuples, index_range_dict):
        """Combine attributes of 1+ hydrants into the merged field list
        used by all data sinks

        Input:
        - name_feature_tuples: list of (dataset_name, feature) tuples
        - index_range_dict: indexes within merged fields for each dataset,
          indexed by name (like {'dataset1': (0, 3), 'dataset2': (3, 5), ...})

        Output:
        - list of attributes merging those of the input features
        """
        n_attributes = max(end for (_, end) in index_range_dict.values())
        extra_attributes = [None] * n_attributes
        for dataset_name, hydrant in name_feature_tuples:
            fields_slice = slice(*index_range_dict[dataset_name])
            extra_attributes[fields_slice] = hydrant.attributes()
        return extra_attributes

    def _add_to_distinct(self, dataset_name, index, hydrant_point,
                        extra_attributes):
        """Add a feature to the distinct sink
        """
        attributes = [dataset_name, index] + extra_attributes

        feature = QgsFeature(self.distinct_fields)
        feature.setGeometry(hydrant_point)
        feature.setAttributes(attributes)
        self.distinct_sink.addFeature(feature, QgsFeatureSink.FastInsert)

    def _add_to_proposed(self, dataset_name, index, characteristic_distance,
                        hydrant_point, extra_attributes):
        """Add a feature to the proposed sink
        """
        attributes = [dataset_name, index,
                      str(characteristic_distance)] + extra_attributes

        feature = QgsFeature(self.proposed_fields)
        feature.setGeometry(hydrant_point)
        feature.setAttributes(attributes)
        self.proposed_sink.addFeature(feature, QgsFeatureSink.FastInsert)

    def _add_to_correspondence(self, dataset_name, index, characteristic_distance,
                        hydrant_line, extra_attributes):
        """Add a feature to the correspondence sink
        """
        attributes = [dataset_name, index,
                      str(characteristic_distance)] + extra_attributes

        feature = QgsFeature(self.correspondence_fields)
        feature.setGeometry(hydrant_line)
        feature.setAttributes(attributes)
        self.correspondence_sink.addFeature(feature, QgsFeatureSink.FastInsert)

    def _get_max_distance_for_component(self, hydrant_graph, component):
        """Construct a subgraph of the hydrant graph and get the longest
        distance between two corresponding hydrants.

        TODO: do we actually want max distance between any two hydrants in
        this correspondence? Probably...
        """
        subgraph = hydrant_graph.subgraph(component).copy()
        distances = nx.get_edge_attributes(subgraph, "distance").values()
        return max(distances)

    def _populate_sinks(self, hydrant_sources, hydrant_graph,
                       index_range_dict, feedback):
        """Take a hydrant graph and process it into the output layers we
        care about.
        """
        # Names of layers determine priority (alphabetical) – priority 0
        # will beat priority 1, etc.
        priority_dict = {hydrant_source.name(): i
                         for i, hydrant_source in enumerate(hydrant_sources)}

        sources_dict = {hydrant_source.name(): hydrant_source
                        for hydrant_source in hydrant_sources}

        # Look at graph's connected components and add each to either
        # distinct or a correspondence layer, and all to a proposed layer
        connected_components = list(nx.connected_components(hydrant_graph))
        total = 100.0 / len(connected_components)
        feedback.pushInfo(f"Processing {len(connected_components)} "
                          "connected components from hydrant graph")
        for counter, component in enumerate(connected_components):
            if feedback.isCanceled():
                break
            # Each component is just a set of nodes
            # Error checking: no component should have multiple nodes from
            # the same original data set.
            source_name_counts = Counter([name for name, _ in component])
            if max(source_name_counts.values()) > 1:
                feedback.pushInfo(
                        "WARNING! WARNING! A component has two hydrants "
                        "from the same dataset. This almost always means a "
                        "weird edge case with multiple datasets. Involved "
                        f"component: {component}")

            # Skip over any components that contain a confirmed negative

            # This doesn't work as is, because it's far too easy to have a
            # negative match with a real positive when that positive is in
            # a partial layer like a manual spot check.
            # if source_names.intersection(confirmed_negative_source_names):
            #     feedback.pushInfo(f"Skipping component {component} because "
            #                       "it contains a confirmed negative")
            #     continue

            if len(component) == 1:
                # This is a distinct (one-dataset) hydrant
                # With just one node, we're guaranteed to get it from pop
                dataset_name, index = component.pop()
                dataset = sources_dict[dataset_name]
                hydrant = list(dataset.getFeatures())[index]
                hydrant_point = self.hydrant_geometries[dataset_name][index]

                extra_attributes = self._get_extra_attributes(
                        [(dataset_name, hydrant)], index_range_dict)

                # Add to appropriate sinks
                self._add_to_distinct(dataset_name, index, hydrant_point,
                                      extra_attributes)
                self._add_to_proposed(dataset_name, index, 0,
                                      hydrant_point, extra_attributes)
            else:
                # This is a correspondence! Between multiple datasets!
                # Hooray! Also PANIIIIIIIC!!!!!
                # First, assemble attributes. Just hope the slicing works
                sorted_nodes = sorted(component)
                max_distance = self._get_max_distance_for_component(
                        hydrant_graph, component)

                source_datasets, source_indices = list(zip(*sorted_nodes))

                geom_points = []
                # proposed_geom will be type (priority, QgsPoint)
                proposed_geom = (float('inf'), QgsPoint(0, 0))
                for dataset_name, hydrant_index in sorted_nodes:
                    current_point = self.hydrant_geometries[dataset_name][hydrant_index]
                    geom_points.append(current_point)
                    # update proposed_geom if this point is higher priority
                    old_priority, _ = proposed_geom
                    current_priority = priority_dict[dataset_name]
                    if current_priority < old_priority:
                        proposed_geom = (current_priority, current_point)

                name_feature_tuples = [
                        (dataset_name,
                         list(sources_dict[dataset_name].getFeatures())[index])
                        for dataset_name, index in sorted_nodes]
                extra_attributes = self._get_extra_attributes(
                        name_feature_tuples, index_range_dict)

                # Now let's generate geometry. For now, just a line
                # connecting hydrants in no particular order – see what it
                # looks like
                geom_points = [QgsPoint(point.asPoint()) for point in geom_points]
                geom_points.append(geom_points[0])
                match_geometry = QgsGeometry.fromPolyline(geom_points)

                # Add to appropriate sinks
                self._add_to_correspondence(
                    ", ".join(source_datasets),
                    ", ".join([str(index) for index in source_indices]),
                    max_distance, match_geometry, extra_attributes
                    )

                _, proposed_point = proposed_geom
                self._add_to_proposed(
                    ", ".join(source_datasets),
                    ", ".join([str(index) for index in source_indices]),
                    max_distance, proposed_point, extra_attributes
                    )
            # Update progress bar
            feedback.setProgress(int(counter * total))

    def _configure_sinks(self, merged_fields, parameters, context):
        """Configure our data sinks

        Input:
        - merged_fields: list of fields from all input data sources
        - parameters + context passed directly from processAlgorithm

        No return value, just sets features of the instance.
        """
        # Now let's make specific fields for our sinks
        correspondence_fields = QgsFields()
        correspondence_fields.append(QgsField("Source datasets", QVariant.String))
        correspondence_fields.append(QgsField("Source indices", QVariant.String))
        correspondence_fields.append(QgsField("Max distance", QVariant.Double))
        for field in merged_fields:
            correspondence_fields.append(field)

        distinct_fields = QgsFields()
        distinct_fields.append(QgsField("Source data sets", QVariant.String))
        distinct_fields.append(QgsField("Source indices",
                                        QVariant.String))
        for field in merged_fields:
            distinct_fields.append(field)

        # Proposed hydrant locations: same fields as correspondence layer
        # (max distance will be zero for hydrants in only one dataset)
        self.correspondence_fields = correspondence_fields
        self.distinct_fields = distinct_fields
        self.proposed_fields = correspondence_fields

        # Create sink object
        (self.correspondence_sink,
         correspondence_dest_id) = self.parameterAsSink(
                parameters,
                self.CORRESPONDENCE,
                context,
                self.correspondence_fields,
                QgsWkbTypes.LineString,
                crs=self.DISTANCE_CRS)
        if self.correspondence_sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters,
                                                               self.CORRESPONDENCE))

        (self.distinct_sink,
         distinct_dest_id) = self.parameterAsSink(
                parameters,
                self.DISTINCT,
                context,
                self.distinct_fields,
                QgsWkbTypes.Point,
                crs=self.DISTANCE_CRS)
        if self.distinct_sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters,
                                                               self.DISTINCT))

        (self.proposed_sink,
         proposed_dest_id) = self.parameterAsSink(
                parameters,
                self.PROPOSED,
                context,
                self.proposed_fields,
                QgsWkbTypes.Point,
                crs=self.DISTANCE_CRS)
        if self.proposed_sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters,
                                                               self.PROPOSED))

        dest_ids = {self.CORRESPONDENCE: correspondence_dest_id,
                    self.DISTINCT: distinct_dest_id,
                    self.PROPOSED: proposed_dest_id}

        return dest_ids

    def processAlgorithm(self, parameters, context, feedback):  # pylint: disable=invalid-name
        """
        Here is where the processing itself takes place.
        """
        # Retrieve the feature source and sink. The 'dest_id' variable is used
        # to uniquely identify the feature sink, and must be included in the
        # dictionary returned by the processAlgorithm function.
        # All hydrant data
        hydrant_sources = self.parameterAsLayerList(
                parameters, self.HYDRANTS, context)
        if hydrant_sources is None:
            raise QgsProcessingException(
                    self.invalidSourceError(parameters, self.HYDRANTS)
                    )
        if len(hydrant_sources) < 2:
            raise QgsProcessingException(
                    "Invalid number of hydrant layers – got "
                    f"{len(hydrant_sources)} but expected 2+")

        # confirmed_negative_sources = self.parameterAsLayerList(
        #         parameters, self.CONFIRMED_NEGATIVES, context)
        # if confirmed_negative_sources is None:
        #     raise QgsProcessingException(
        #             self.invalidSourceError(parameters,
        #                                     self.CONFIRMED_NEGATIVES))
        # confirmed_negative_source_names = [
        #         source.name()
        #         for source in confirmed_negative_sources]

        # Okay, prepare our data sinks
        merged_fields, index_range_dict = self._generate_merged_fields(
                hydrant_sources)
        dest_ids = self._configure_sinks(merged_fields, parameters, context)

        # Transform geometries to one CRS
        # all_hydrant_sources = hydrant_sources + confirmed_negative_sources
        self._populate_geometries(hydrant_sources)

        # Construct hydrant graph!
        hydrant_graph = self._generate_hydrant_graph(hydrant_sources, feedback)

        # Populate sinks based on graph
        self._populate_sinks(hydrant_sources, hydrant_graph,
                             index_range_dict, feedback)

        return dest_ids

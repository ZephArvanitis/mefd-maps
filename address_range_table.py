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
        QgsProcessing,
        QgsProcessingParameterField,
        QgsProcessingParameterExpression,
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


class AddressInfo:
    """Container for info on addresses within a tile and sharing a street
    """

    def __init__(self):
        self.addresses = set()
        self.notes = []
        self.districts = set()
        self.custom_region = None
        self.supplementary_info = set()

    def __str__(self):
        return (f"AddressInfo({self.addresses}, {self.notes}, "
                f"{self.districts}, {self.custom_region}, "
                f"{self.supplementary_info})")

    def add_address(self, new_address):
        """Add address to container
        """
        self.addresses |= {new_address}

    def add_district(self, new_district):
        """Add district to container
        """
        self.districts |= {new_district}

    def add_supplementary_info(self, new_info):
        """Add extra info like "see Pioneer Trails supplementary map"
        """
        self.supplementary_info |= {new_info}

    def generate_simple_notes(self):
        """Generate EVEN/ODD ONLY and NO ADDRESSES notes
        """
        if len(self.addresses) == 0:
            self.notes = ["NO ADDRESSES"]
        else:
            # generate notes
            self.notes = []
            addresses_even = {address for address in self.addresses
                              if address % 2 == 0}
            addresses_odd = {address for address in self.addresses
                             if address % 2 == 1}
            n_even = len(addresses_even)
            n_odd = len(addresses_odd)
            if n_even == 0 and n_odd > 1:
                self.notes.append("ODD ONLY")
            if n_odd == 0 and n_even > 1:
                self.notes.append("EVEN ONLY")

    @property
    def notes_string(self):
        """Concatenate notes for the table
        """
        return ", ".join(self.notes)

    @property
    def districts_string(self):
        """Concatenate districts for the table
        """
        # Override district with supplementary info (e.g. "SHELTER BAY"
        # overrides "13")
        if self.supplementary_info:
            return " ".join(self.supplementary_info)

        return ", ".join(self.districts)

    @property
    def min(self):
        """Return minimum address
        """
        if self.addresses:
            return min(self.addresses)
        return None

    @property
    def max(self):
        """Return max address
        """
        if self.addresses:
            return max(self.addresses)
        return None


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
    NAME_FIELD = 'name_field'

    ADDRESS = 'ADDRESS'
    ADDRESS_FIELD = 'address_field'
    ADDRESS_STREET_FIELD = 'address_street_field'

    SHELTER_BAY = "SHELTER_BAY"
    SHELTER_BAY_FIELD = "shelter_bay_field"

    DISTRICTS = "DISTRICTS"
    DISTRICT_NAME_FIELD = "district_name_field"

    STREETS = "STREETS"
    STREET_FIELD = 'street_field'

    OUTPUT = 'OUTPUT'

    STREET_TO_ADDRESS_NAME_MAP = {"FIRST": "1ST",
                                  "SECOND": "2ND",
                                  "THIRD": "3RD",
                                  "FOURTH": "4TH",
                                  "FIFTH": "5TH",
                                  "SIXTH": "6TH",
                                  "&": "AND"}
    REVERSE_NAME_MAP = {"É": "E",
                        "-": " ",
                        "'": "",
                        r"\.": ""}

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

    def displayName(self):  # pylint: disable=invalid-name
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
        return self.tr("Given a polygon grid and a layer of address points, "
                       "generate a table showing address ranges per grid "
                       "tile")

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
                self.ADDRESS_STREET_FIELD,
                'Define full street name',
                '',
                self.ADDRESS)
            )

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

        # Shelter Bay
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.SHELTER_BAY,
                self.tr('Shelter Bay boundary'),
                [QgsProcessing.TypeVectorPolygon]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.SHELTER_BAY_FIELD,
                'Field specifying shelter-bay-ness',
                '',
                self.SHELTER_BAY))

        # Streets (needed for NO STRUCTURES rows in table)
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.STREETS,
                self.tr('Streets'),
                [QgsProcessing.TypeVectorLine]
            )
        )
        self.addParameter(
            QgsProcessingParameterExpression(
                self.STREET_FIELD,
                'Define full street name',
                '',
                self.STREETS)
            )

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

    def _get_address_inputs(self, parameters, context):
        """Get address-related inputs
        """
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
            self.ADDRESS_STREET_FIELD,
            context)
        street_name_expression = QgsExpression(street_name_exp_str)
        return (address_source, street_number_field, street_name_expression)

    def _get_shelter_bay(self, parameters, context):
        """Get shelter bay input
        """
        shelter_bay_source = self.parameterAsSource(
            parameters,
            self.SHELTER_BAY,
            context
        )
        if shelter_bay_source is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.SHELTER_BAY)
            )

        name_field = self.parameterAsString(
            parameters,
            self.SHELTER_BAY_FIELD,
            context)


        return shelter_bay_source, name_field

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

    def _get_street_inputs(self, parameters, context):
        """Get street-related inputs
        """
        streets_source = self.parameterAsSource(
            parameters,
            self.STREETS,
            context
        )
        if streets_source is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.STREETS)
            )

        street_name_exp_str = self.parameterAsString(
            parameters,
            self.STREET_FIELD,
            context)
        street_name_expression = QgsExpression(street_name_exp_str)
        return (streets_source, street_name_expression)

    def _configure_sink(self, parameters, context, crs):
        """Configure output layer
        """
        # Generate list of fields
        fields = QgsFields()
        fields.append(QgsField("Street name", QVariant.String))
        fields.append(QgsField("District(s)", QVariant.String))
        fields.append(QgsField("Start address", QVariant.String))
        fields.append(QgsField("End address", QVariant.String))
        fields.append(QgsField("Notes", QVariant.String))
        fields.append(QgsField("Map page", QVariant.String))

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

    def _intersection_join_old(self, input1, input2, context, feedback):
        """Run a spatial join with two inputs, storing result in memory
        """
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
        feedback.pushInfo(f"About to try a join with {input1} (type {type(input1)}) x {input2}")
        spatial_join_dict = {
            "INPUT": input1,
            "PREDICATE": [0,],  # 0 means intersection
            "JOIN": input2,
            "JOIN_FIELDS": [],  # no field join, only spatial
            "METHOD": 0,  # generates attribute for each matching feature
            "DISCARD_NONMATCHING": False,
            "PREFIX": "",
            "OUTPUT": "memory:"  # We'll store this in memory and steal it
            # back from the context after the algorithm completes
        }

        # Method from https://gis.stackexchange.com/a/426338/161588
        join_result = processing.run(
            "native:joinattributesbylocation",
            spatial_join_dict,
            is_child_algorithm=True,
            context=context, feedback=feedback)["OUTPUT"]
        join_layer = QgsProcessingContext.takeResultLayer(
            context, join_result)

        return join_layer

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
        - starting layer: the address or streets layer to join against
        - parameters: the processing algorithm parameters, from which the
          method will retrieve shelter bay, districts, and grid cells
        - context: the processing context, used for the actual joins
        - feedback: the feedback for the user, useful for printing messages
          or just to pass to the joins

        Returns:
        - a layer of the starting layer, joined with shelter bay,
          districts, and grid

        We're going to do three spatial joins:
        1. address layer -> shelter bay boundary layer
           - will add the `shelterbay` attribute – 1 if it's in shelter bay, NULL otherwise.
        2. address layer -> district boundaries
           - will add district_f attribute like 'MEFD' or '13'
        3. address layer -> grid cells
           - will add a whole bunch of attributes, but we only care about the
             grid ID one, which the user will choose from a dropdown.
        """
        # Spatial join #1: address layer -> shelter bay boundary
        address_with_shelter_bay_layer = self._intersection_join(
                 starting_layer, parameters[self.SHELTER_BAY],
                 context, feedback)

        # Spatial join #2: address layer -> district boundaries
        address_with_district_layer = self._intersection_join(
                address_with_shelter_bay_layer,
                parameters[self.DISTRICTS], context, feedback)

        # Spatial join #3: address layer -> grid cells
        address_with_grid_layer = self._intersection_join(
                 address_with_district_layer, parameters[self.GRID],
                 context, feedback)

        return address_with_grid_layer

    def apply_mapping(self, street_name, feedback):
        """Apply substitutions like FIRST -> 1ST in a street name

        Also populates self.address_to_street_name_map as we go
        """
        updated_street_name = street_name
        # First, do unambiguous substitutions
        for key, update in self.STREET_TO_ADDRESS_NAME_MAP.items():
            if re.search(key, updated_street_name) is not None:
                feedback.pushInfo(f"Updating {key} -> {update} in {updated_street_name}")
                updated_street_name = re.sub(key, update,
                                             updated_street_name)
        # Now do substitutions to make street names match address point
        # streets, but log those in a reverse dictionary for later
        # correction
        for key, update in self.REVERSE_NAME_MAP.items():
            if re.search(key, updated_street_name) is not None:
                feedback.pushInfo(f"Updating {key} -> {update} in {updated_street_name}")
                new_street_name = re.sub(key, update,
                                         updated_street_name)
                self.address_to_street_name_map[
                        new_street_name] = updated_street_name
                updated_street_name = new_street_name

        if updated_street_name != street_name:
            feedback.pushInfo(f"New street name {updated_street_name}")
            feedback.pushInfo(f"Street name mapping {self.address_to_street_name_map}")
        return updated_street_name

    def digest_address_points(self,
                              address_with_grid_layer, fields,
                              feedback, apply_map=False):
        """Process joined address points into dictionaries

        Inputs:
        - address_with_grid_features: joined address layer, which should
          have fields from shelter bay, districts, and grid
        - fields: dict mapping "map_id", "street_number", "shelter_bay",
          and "district" to the field names for each entry.
        - expression_context: context for evaluating full street names from
          fields on the address table.
        - feedback: for providing progress/updates to the user
        - apply_map: whether to use STREET_TO_ADDRESS_NAME_MAP and
          REVERSE_NAME_MAP when creating keys in the dictionaries. (Should
          be True when running on streets, False when running on address
          points)

        Returns a dict of dicts. Each dict is of form {(street_name,
        grid_tile): something}, where something is given by:
        - {"numbers": set of address numbers,
           "shelter_bay": True/False for whether addresses on the
           street/tile combo are in shelter bay,
           "districts": set of districts for the street/tile combo
          }

        """
        # Compute the number of steps to display within the progress bar and
        # get features from source
        total = (100.0 / address_with_grid_layer.featureCount()
                 if address_with_grid_layer.featureCount() else 0)
        feedback.pushInfo(f"Processing {address_with_grid_layer.featureCount()} "
                          "address points")

        address_with_grid_features = address_with_grid_layer.getFeatures()

        # Initialize expression context
        expression_context = QgsExpressionContext()
        expression_context.appendScopes(
            QgsExpressionContextUtils.globalProjectLayerScopes(
                address_with_grid_layer))

        # We're going to iterate over attributes and add 'em to a dict of
        # form:
        # {(street_name, grid_tile): AddressInfo}
        address_info_dict = defaultdict(AddressInfo)
        for i, feature in enumerate(address_with_grid_features):
            if feedback.isCanceled():
                break

            feature.setFields(address_with_grid_layer.fields(), False)
            map_id = feature[fields["map_id"]]
            # Set context and evaluate street name expression
            expression_context.setFeature(feature)
            street_name = fields["street_name_expression"].evaluate(expression_context)

            # skip streets outside the grid tiles
            if map_id is None:
                continue

            # Make street names match across streets vs address points
            if apply_map:
                new_street_name = self.apply_mapping(street_name,
                                                     feedback)
                if new_street_name != street_name:
                    feedback.pushInfo(f"street name {street_name} updated to {new_street_name}")
                    street_name = new_street_name

            index_tuple = (street_name, map_id)

            # Address
            if "street_number" in fields:
                address_info_dict[index_tuple].add_address(
                        feature[fields["street_number"]])

            # District
            district = feature[fields["district"]]
            if district:
                address_info_dict[index_tuple].add_district(district)
            # Supplementary info
            supplementary_info = feature[fields["shelter_bay"]]
            if supplementary_info:
                address_info_dict[index_tuple].add_supplementary_info(supplementary_info)

            feedback.setProgress(int(i * total))

        return address_info_dict

    def order_address_table(self, address_info_dict):
        """Apply our funky custom ordering and return a list of tuples like
        [((street, mapid), AddressInfo), ...]

        1. Street name (sort numbered streets by number rather than
            alphabetically)
        2. Start address
        3. Map page
        """
        items = address_info_dict.items()
        def custom_compare(item1, item2):
            (street1, grid1), info1 = item1
            (street2, grid2), info2 = item2

            # Last, if start address and street name match, use map page to
            # decide order
            if street1 == street2 and info1.min == info2.min:
                return -1 if grid1 < grid2 else (1 if grid1 > grid2 else 0)

            # Second from last: if street address matches, sort based on
            # min address number
            if street1 == street2:
                min1 = info1.min
                min2 = info2.min
                # Handle nones
                if min1 is None:
                    min1 = 100000000  # effectively infinity, sort later
                if min2 is None:
                    min2 = 100000000  # ditto
                return -1 if min1 < min2 else (1 if min1 > min2 else 0)

            # Finally, let's sort based on street number
            number_match1 = re.match(r"\d+", street1)
            number_match2 = re.match(r"\d+", street2)
            # If both are numbered streets, return based on the street
            # *values*
            if number_match1 is not None and number_match2 is not None:
                street_number1 = int(number_match1.group())
                street_number2 = int(number_match2.group())
                return street_number1 - street_number2
            # If zero or one of these is a number, use string comparison
            return -1 if street1 < street2 else (1 if street1 > street2 else 0)

        return sorted(items, key=functools.cmp_to_key(custom_compare))

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

        (address_source,
         street_number_field,
         address_street_name_expression) = self._get_address_inputs(
                parameters, context)

        _, shelter_bay_field = self._get_shelter_bay(
                parameters, context)

        _, district_field = self._get_district_inputs(
                parameters, context)

        _, street_name_expression = self._get_street_inputs(
                parameters, context)

        dest_ids = self._configure_sink(parameters, context, address_source.sourceCrs())

        # Send some information to the user
        grid_crs = grid_source.sourceCrs().authid()
        address_crs = address_source.sourceCrs().authid()
        feedback.pushInfo(
            f'Grid CRS is {grid_crs}, Address CRS is {address_crs}')

        address_with_grid_layer = self._perform_joins(parameters[self.ADDRESS],
                                                      parameters, context,
                                                      feedback)

        street_with_grid_layer = self._perform_joins(parameters[self.STREETS],
                                                     parameters, context,
                                                     feedback)

        address_info_dict = self.digest_address_points(
                address_with_grid_layer,
                {"map_id": name_field,
                 "street_number": street_number_field,
                 "shelter_bay": shelter_bay_field,
                 "district": district_field,
                 "street_name_expression": address_street_name_expression}, feedback)

        street_info_dict = self.digest_address_points(
                street_with_grid_layer,
                {"map_id": name_field,
                 "shelter_bay": shelter_bay_field,
                 "district": district_field,
                 "street_name_expression": street_name_expression},
                feedback, apply_map=True)

        address_streets = {street for street, _ in address_info_dict.keys()}
        street_streets = {street for street, _ in street_info_dict.keys()}
        feedback.pushInfo(f"address produces {len(address_streets)} streets")
        feedback.pushInfo(f"streets produce {len(street_streets)} streets")
        feedback.pushInfo(f"present in address but not streets: {address_streets - street_streets}")
        feedback.pushInfo(f"present in streets but not address: {street_streets - address_streets}")

        # Merge streets dicts into address point dicts, without overriding
        # existing values. Note that order of args to | matters!
        # >>> a = {('a', 'r6a'): {3, 4, 5}, ('a', 'r5a'): {1, 2}}
        # >>> b = {('a', 'r6a'): {}, ('a', 'r4a'): {}}
        # >>> a | b
        # {('a', 'r6a'): {}, ('a', 'r5a'): {1, 2}, ('a', 'r4a'): {}}
        # >>> b | a
        # {('a', 'r6a'): {3, 4, 5}, ('a', 'r4a'): {}, ('a', 'r5a'): {1, 2}}
        # Basically the second arg is the one whose values will "override"
        # those of the first
        address_info_dict = street_info_dict | address_info_dict

        address_info = self.order_address_table(address_info_dict)

        for (street_name, map_id), address_info in address_info:
            if street_name in self.address_to_street_name_map:
                table_street_name = self.address_to_street_name_map[street_name]
            else:
                table_street_name = street_name
            # Generate notes. In particular the ALL ADDRESSES entry
            # requires a global check of other grid tiles, so it has to
            # live here instead of AddressInfo
            address_info.generate_simple_notes()
            if len(address_info.addresses) > 0:
                all_grids_for_street = [
                        grid_tile
                        for street, grid_tile in address_info_dict.keys()
                        if street == street_name]
                if len(all_grids_for_street) == 1:
                    # intentionally override even/odd only, since all addresses
                    # is strictly more informative
                    address_info.notes = ["ALL ADDRESSES"]

            # Order of attributes matters! Make sure this matches order
            # defined above
            attributes = [table_street_name, address_info.districts_string,
                          address_info.min, address_info.max,
                          address_info.notes_string, map_id]
            feature = QgsFeature(self.output_fields)
            feature.setAttributes(attributes)
            self.sink.addFeature(feature, QgsFeatureSink.FastInsert)

        # Return the results of the algorithm.
        return dest_ids

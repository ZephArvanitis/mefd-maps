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
from collections import defaultdict
from typing import Dict, List, Optional, Set
import numpy as np
import random
from scipy.spatial import ConvexHull

from qgis.PyQt.QtCore import QCoreApplication, QVariant  # pylint: disable=import-error
from qgis.core import (  # pylint: disable=import-error
    QgsProcessing,
    QgsProcessingParameterField,
    QgsProcessingParameterExpression,
    QgsExpression,
    QgsExpressionContext,
    QgsExpressionContextUtils,
    QgsFeature,
    QgsFeatureRequest,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsMultiPoint,
    QgsPointXY,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeatureSourceDefinition,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterNumber,
    QgsProject,
    QgsWkbTypes,
)
from qgis import processing  # pylint: disable=import-error


class PointAddress:
    """Contain just a point and a label for it"""
    def __init__(self, point, label: str):
        self.point = point
        self.label = label

    def __repr__(self):
        return f"PointAddress({self.point}, {self.label})"


class PolygonAddress:
    """Contain a label within a polygon"""
    def __init__(self, polygon, label: str, scope: Optional[str] = None):
        self.polygon = polygon
        self.label = label
        self.scope = scope

    def __repr__(self):
        return f"PolygonAddress({self.polygon}, {self.label}, {self.scope})"


class Address:
    """Encapsulate a single address"""
    def __init__(self, address_number: int, full_street: str, town: str):
        self.address_number = address_number
        self.full_street = full_street
        self.town = town

    def __eq__(self, other):
        return (self.address_number == other.address_number and
                self.full_street == other.full_street and
                self.town == other.town)

    def __repr__(self):
        return (f"Address({self.address_number}, "
                f"{self.full_street}, {self.town})")

    def __str__(self):
        return f"{self.address_number} {self.full_street}, {self.town}"

    @property
    def street_address(self):
        return f"{self.address_number} {self.full_street}"

    def __hash__(self):
        return hash(repr(self))


class SkagitAddressPoint:
    """Contain info from a single skagit address point"""
    def __init__(self, point, address_number: int, full_street: str,
                 town: str,
                 building=None, unit=None,
                 parcel_id: Optional[str]=None):
        self.point = point
        self.address_number = address_number
        self.full_street = full_street
        self.town = town
        self.building = building
        self.unit = unit
        self.parcel_id = parcel_id

    @property
    def address(self) -> Address:
        """Extract just address"""
        return Address(self.address_number, self.full_street, self.town)

    @property
    def label(self) -> str:
        """Desired label"""
        if not self.building and not self.unit:
            return f"{self.address_number}"
        if not self.building:
            return f"{self.address_number} {self.unit}"
        if not self.unit:
            return f"{self.address_number} {self.building}"
        return f"{self.address_number} {self.building}-{self.unit}"

    @property
    def label_address_only(self) -> str:
        """If you're labeling and want *only* the address"""
        return f"{self.address_number}"

    def __repr__(self):
        return (f"SkagitAddressPoint({self.point}, {self.address_number}, "
                f"{self.full_street}, {self.town}, {self.building}, "
                f"{self.unit}, {self.parcel_id})")


class Parcel:
    """Describe a single parcel"""
    parcel_type: float
    width: float
    area: float
    global_id: str
    addresses: List[SkagitAddressPoint]

    def __init__(self, geometry, parcel_type: int, width: float,
                 area: float, global_id: str,
                 address_points: Optional[List[SkagitAddressPoint]] = None):
        self.geometry = geometry
        self.parcel_type = parcel_type
        self.width = width
        self.area = area
        self.global_id = global_id
        if address_points is None:
            address_points = []
        self.address_points = address_points

    def __repr__(self):
        return (f"Parcel({self.geometry}, {self.parcel_type}, "
                f"{self.width}, {self.area}, {self.global_id}, "
                f"{self.address_points})"
               )

    def __eq__(self, other):
        return (self.geometry == other.geometry and
                self.parcel_type == other.parcel_type and
                self.width == other.width and
                self.area == other.area and
                self.global_id == other.global_id)

    def __hash__(self):
        # Omit affiliated addresses from hash
        return hash(f"Parcel({self.geometry}, {self.parcel_type}, "
                f"{self.width}, {self.area}, {self.global_id})")


class OutputLayers:
    """Hold the four output layers, in python objects"""
    boring_addresses: List[PointAddress]
    dense_address_polygons: List[PolygonAddress]
    wrapper_polygons: List[PolygonAddress]
    sub_address_points: List[PointAddress]

    def __init__(self,
                 boring_addresses: Optional[List[PointAddress]] = None,
                 dense_address_polygons: Optional[List[PolygonAddress]] = None,
                 wrapper_polygons: Optional[List[PolygonAddress]] = None,
                 sub_address_points: Optional[List[PointAddress]] = None
                 ):
        if boring_addresses is None:
            boring_addresses = []
        if dense_address_polygons is None:
            dense_address_polygons = []
        if wrapper_polygons is None:
            wrapper_polygons = []
        if sub_address_points is None:
            sub_address_points = []

        self.boring_addresses = boring_addresses
        self.dense_address_polygons = dense_address_polygons
        self.wrapper_polygons = wrapper_polygons
        self.sub_address_points = sub_address_points

    def __str__(self):
        return("OutputLayers with "
               f"{len(self.boring_addresses)} boring addresses, "
               f"{len(self.dense_address_polygons)} dense address polygons, "
               f"{len(self.wrapper_polygons)} wrapper polygons, and "
               f"{len(self.sub_address_points)} sub address points.")


class AddressLabelProcessingAlgorithm(QgsProcessingAlgorithm):
    """
    This takes a layer of address points and a layer of parcels, and
    returns several layers with which to label addresses reasonably:
    - boring old points: just put an address on the point
    - polygons with an address: put the address within the polygon
    - wrapper polygons with an address: draw a line around the boundary and
      say "all the misc points in here are this address"
    - sub-points: points within the above polygons to label with individual
      sub-addresses

    All Processing algorithms should extend the QgsProcessingAlgorithm
    class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    # Parcels
    PARCEL = "PARCEL"
    PARCEL_TYPE_FIELD = "parcel_type_field"
    PARCEL_TYPE_VALUE = "parcel_type_value"  # 0 in skagit county

    # Address points
    ADDRESS = "ADDRESS"
    ADDRESS_FIELD = "address_field"
    ADDRESS_STREET_FIELD = "address_street_field"
    ADDRESS_BUILDING_FIELD = "address_building_field"
    ADDRESS_UNIT_FIELD = "address_unit_field"
    ADDRESS_TOWN_FIELD = "address_town_field"

    # Grid layer just for extent
    GRID = 'GRID'

    ADDRESS_POINTS = "OUTPUT_ADDRESS_POINTS"
    ADDRESS_POLYGONS = "OUTPUT_ADDRESS_POLYGONS"
    WRAPPER_POLYGONS = "WRAPPER_POLYGONS"
    ADDRESS_SUB_POINTS = "ADDRESS_SUB_POINTS"

    def __init__(self):
        self.address_points_sink = None
        self.address_polygons_sink = None
        self.wrapper_polygons_sink = None
        self.address_sub_points_sink = None
        self.output_fields = None
        self.parcel_type_value: float = None
        self.parcel_dict: Dict[str, Parcel] = {}
        self.address_dict: Dict[Address, List[SkagitAddressPoint]]
        self.output_layers: OutputLayers = None
        super().__init__()

    def tr(self, string):  # pylint: disable=invalid-name
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):  # pylint: disable=invalid-name
        """Initialize instance (used by QGIS processing toolbox)"""
        return AddressLabelProcessingAlgorithm()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "address_label"

    def displayName(self):  # pylint: disable=invalid-name
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr("Generate address label layers")

    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr("MEFD mapbook scripts")

    def groupId(self):  # pylint: disable=invalid-name
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "mefdmapbookscripts"

    def shortHelpString(self):  # pylint: disable=invalid-name
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it..
        """
        return self.tr(
            "Given address points and parcels, return several layers useful "
            "for labeling addresses"
        )

    def initAlgorithm(self, config=None):  # pylint: disable=invalid-name
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """
        # Grid
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.GRID, self.tr("Grid layer"), [QgsProcessing.TypeVectorPolygon]
            )
        )

        # Address points
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.ADDRESS, self.tr("Address layer"), [QgsProcessing.TypeVectorPoint]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.ADDRESS_FIELD, "Choose address number field", "", self.ADDRESS
            )
        )
        self.addParameter(
            QgsProcessingParameterExpression(
                self.ADDRESS_STREET_FIELD, "Define full street name", "", self.ADDRESS
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.ADDRESS_TOWN_FIELD, "Choose town field", "", self.ADDRESS
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.ADDRESS_BUILDING_FIELD, "Choose building field", "", self.ADDRESS
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.ADDRESS_UNIT_FIELD, "Choose unit field", "", self.ADDRESS
            )
        )

        # Parcels
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.PARCEL,
                self.tr("Parcels layer"),
                [QgsProcessing.TypeVectorPolygon],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.PARCEL_TYPE_FIELD,
                "Field giving the parcel type",
                "",
                self.PARCEL,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARCEL_TYPE_VALUE,
                "Parcel type of interest (0 for skagit count)",
                defaultValue=0
            )
        )

        # We add a feature sink in which to store our processed features (this
        # usually takes the form of a newly created vector layer when the
        # algorithm is run in QGIS).
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.ADDRESS_POINTS,
                self.tr("Boring addresses"))
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.ADDRESS_POLYGONS,
                self.tr("Dense address polygons"))
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.WRAPPER_POLYGONS,
                self.tr("Wrapper polygons"))
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.ADDRESS_SUB_POINTS,
                self.tr("Sub-addresses inside polygons"))
        )

    def _get_grid_inputs(self, parameters, context):
        """Get grid-related inputs"""
        grid_source = self.parameterAsSource(parameters, self.GRID, context)
        if grid_source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.GRID))

        return grid_source

    def _get_address_inputs(self, parameters, context):
        """Get address-related inputs"""
        # Address info
        address_source = self.parameterAsSource(parameters, self.ADDRESS, context)
        if address_source is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.ADDRESS)
            )

        street_number_field = self.parameterAsString(
            parameters, self.ADDRESS_FIELD, context
        )
        street_name_exp_str = self.parameterAsString(
            parameters, self.ADDRESS_STREET_FIELD, context
        )
        street_name_expression = QgsExpression(street_name_exp_str)

        building_field = self.parameterAsString(
            parameters, self.ADDRESS_BUILDING_FIELD, context
        )
        unit_field = self.parameterAsString(
            parameters, self.ADDRESS_UNIT_FIELD, context
        )
        town_field = self.parameterAsString(
            parameters, self.ADDRESS_TOWN_FIELD, context
        )
        return (address_source, street_number_field,
                street_name_expression, building_field, unit_field,
                town_field)

    def _get_parcel_inputs(self, parameters, context):
        """Get parcel-related inputs"""
        parcel_source = self.parameterAsSource(parameters, self.PARCEL, context)
        if parcel_source is None:
            raise QgsProcessingException(
                    self.invalidSourceError(parameters,
                                            self.PARCEL))
        parcel_type_field = self.parameterAsString(
            parameters, self.PARCEL_TYPE_FIELD, context
        )

        parcel_type_value = self.parameterAsString(
            parameters, self.PARCEL_TYPE_VALUE, context
        )

        return parcel_source, parcel_type_field, parcel_type_value

    def _configure_sinks(self, parameters, context, crs):
        """Configure output layers"""

        # Address points
        address_point_fields = QgsFields()
        address_point_fields.append(QgsField("Address", QVariant.String))

        (self.address_points_sink, address_points_dest_id) = self.parameterAsSink(
            parameters, self.ADDRESS_POINTS, context, address_point_fields,
            QgsWkbTypes.Point, crs=crs
        )
        if self.address_points_sink is None:
            raise QgsProcessingException(self.invalidSinkError(
                parameters, self.ADDRESS_POINTS))

        # Address polygons
        address_polygons_fields = QgsFields()
        address_polygons_fields.append(QgsField("Address", QVariant.String))

        (self.address_polygons_sink, address_polygons_dest_id) = self.parameterAsSink(
            parameters, self.ADDRESS_POLYGONS, context, address_polygons_fields,
            QgsWkbTypes.Polygon, crs=crs
        )
        if self.address_polygons_sink is None:
            raise QgsProcessingException(self.invalidSinkError(
                parameters, self.ADDRESS_POLYGONS))

        # Wrapper polygons
        wrapper_polygons_fields = QgsFields()
        wrapper_polygons_fields.append(QgsField("Full label", QVariant.String))
        wrapper_polygons_fields.append(QgsField("Scope", QVariant.String))

        (self.wrapper_polygons_sink, wrapper_polygons_dest_id) = self.parameterAsSink(
            parameters, self.WRAPPER_POLYGONS, context, wrapper_polygons_fields,
            QgsWkbTypes.Polygon, crs=crs
        )
        if self.wrapper_polygons_sink is None:
            raise QgsProcessingException(self.invalidSinkError(
                parameters, self.WRAPPER_POLYGONS))

        # Address sub-points
        address_sub_point_fields = QgsFields()
        address_sub_point_fields.append(QgsField("Label", QVariant.String))

        (self.address_sub_points_sink, address_sub_points_dest_id) = self.parameterAsSink(
            parameters, self.ADDRESS_SUB_POINTS, context, address_sub_point_fields,
            QgsWkbTypes.Point, crs=crs
        )
        if self.address_sub_points_sink is None:
            raise QgsProcessingException(self.invalidSinkError(
                parameters, self.ADDRESS_SUB_POINTS))

        # Return all this stuff
        self.output_fields = {
                self.ADDRESS_POINTS: address_point_fields,
                self.ADDRESS_POLYGONS: address_polygons_fields,
                self.WRAPPER_POLYGONS: wrapper_polygons_fields,
                self.ADDRESS_SUB_POINTS: address_sub_point_fields,
                }
        return {self.ADDRESS_POINTS: address_points_dest_id,
                self.ADDRESS_POLYGONS: address_polygons_dest_id,
                self.WRAPPER_POLYGONS: wrapper_polygons_dest_id,
                self.ADDRESS_SUB_POINTS: address_sub_points_dest_id,
                }

    def clip_layers(self, parameters, feedback):
        """Clip address + parcel layers to grid"""
        # Send some information to the user
        feedback.pushInfo("Clipping inputs")

        address_clip_layer = processing.run(
            "native:clip",
            {'INPUT':QgsProcessingFeatureSourceDefinition(
                 parameters[self.ADDRESS],
                 selectedFeaturesOnly=False,
                 featureLimit=-1,
                 flags=QgsProcessingFeatureSourceDefinition.FlagOverrideDefaultGeometryCheck,
                 geometryCheck=QgsFeatureRequest.GeometrySkipInvalid),
             'OVERLAY':parameters[self.GRID],
             'OUTPUT':'TEMPORARY_OUTPUT'})["OUTPUT"]

        parcel_clip_layer = processing.run(
            "native:clip",
            {'INPUT':QgsProcessingFeatureSourceDefinition(
                 parameters[self.PARCEL],
                 selectedFeaturesOnly=False,
                 featureLimit=-1,
                 flags=QgsProcessingFeatureSourceDefinition.FlagOverrideDefaultGeometryCheck,
                 geometryCheck=QgsFeatureRequest.GeometrySkipInvalid),
             'OVERLAY':parameters[self.GRID],
             'OUTPUT':'TEMPORARY_OUTPUT'})["OUTPUT"]

        return address_clip_layer, parcel_clip_layer

    def add_geom_fields(self, parcel_clip_layer, fields, feedback):
        """Add area and width to parcel layer"""
        feedback.pushInfo("Calculating area and width of parcels")
        parcel_clip_layer_with_area = processing.run(
            "native:refactorfields",
            {'INPUT': parcel_clip_layer,
             'FIELDS_MAPPING':[
                 {'expression': f'"{fields["parcel_type"]}"','length': 18,
                  'name': fields["parcel_type"],'precision': 8,
                  'sub_type': 0,'type': 6,'type_name': 'double precision'},
                 {'expression': f'"{fields["parcel_global_id"]}"','length': 38,
                  'name': fields["parcel_global_id"],'precision': 0,
                  'sub_type': 0,'type': 10,'type_name': 'text'},
                 {'expression': 'bounds_width($geometry)','length': 0,
                  'name': fields["width"],'precision': 0,
                  'sub_type': 0,'type': 6,'type_name': 'double precision'},
                 {'expression': '$area','length': 0,
                  'name': fields["area"],'precision': 0,
                  'sub_type': 0,'type': 6,'type_name': 'double precision'}],
             'OUTPUT':'TEMPORARY_OUTPUT'})["OUTPUT"]

        return parcel_clip_layer_with_area

    def get_parcels(self, parcel_layer, fields, feedback):
        """Convert parcels to python objects, make dict from global id -> parcel"""
        feedback.pushInfo("About to process parcel layer into python objects")
        parcel_dict = {}
        for feature in parcel_layer.getFeatures():
            feature.setFields(parcel_layer.fields(), False)

            geometry = feature.geometry()
            parcel_type = feature[fields["parcel_type"]]
            width = feature[fields["width"]]
            area = feature[fields["area"]]
            global_id = feature[fields["parcel_global_id"]]

            parcel = Parcel(geometry, parcel_type, width, area, global_id)
            assert global_id not in parcel_dict
            parcel_dict[global_id] = parcel

        return parcel_dict

    def get_address_points(self, address_with_parcel_layer, fields,
                           feedback):
        """Process layer points into a list of python objects

        Along the way, add address points to their parcels for
        back-reference.
        """
        expression_context = QgsExpressionContext()

        python_objects = []
        feedback.pushInfo("About to process address points into python objects")
        for feature in address_with_parcel_layer.getFeatures():
            feature.setFields(address_with_parcel_layer.fields(), False)
            address_number = feature[fields["street_number"]]
            building = feature[fields["building"]]
            unit = feature[fields["unit"]]
            town = feature[fields["town"]]
            parcel_id = feature[fields["parcel_global_id"]]

            # If unit or building is null, instead of showing up as None it
            # registers as a PyQt5.QtCore.QVariant with a value() of NULL?
            # Not sure why, but let's fix that now
            if hasattr(building, "value") and str(building) == "NULL":
                building = None
            if hasattr(unit, "value") and str(unit) == "NULL":
                unit = None

            # street name via expression
            expression_context.setFeature(feature)
            street_name = fields["street_name_expression"].evaluate(expression_context)

            # geometry
            geometry = feature.geometry()

            address_object = SkagitAddressPoint(geometry, address_number,
                                                street_name, town,
                                                building, unit, parcel_id)
            python_objects.append(address_object)
            if parcel_id is not None:
                self.parcel_dict[parcel_id].address_points.append(address_object)

        return python_objects

    def make_address_dict(
            self,
            address_points: List[SkagitAddressPoint]) -> Dict[
                    Address, Set[SkagitAddressPoint]]:
        """Collect address points by base address"""
        address_dict = defaultdict(set)
        for address_point in address_points:
            address_dict[address_point.address].add(address_point)

        return dict(address_dict)

    def add_as_point_or_parcel(self, geometry, label, parcel_id,
                               use_full_address=True):
        """Add a label as either a point or a parcel

        returns True if parcel, False if point
        """
        if parcel_id is not None:
            parcel = self.parcel_dict[parcel_id]
            # TODO: don't hardcode these values
            # Also maybe optimize them??
            is_small = parcel.width < 200 or parcel.area < 2000
            # Filter out road/water parcels
            is_plot = parcel.parcel_type == self.parcel_type_value
            if use_full_address:
                only_address_in_parcel = len(parcel.address_points) == 1
            else:
                only_address_in_parcel = len(set(
                    address_point.address
                    for address_point in parcel.address_points)) == 1

            if is_small and is_plot and only_address_in_parcel:
                # Small parcel -> put label inside parcel
                polygon = PolygonAddress(parcel.geometry,
                                         label)
                self.output_layers.dense_address_polygons.append(polygon)
                return True
        # If we reach here, we want to label on the point
        point_address = PointAddress(geometry, label)
        self.output_layers.boring_addresses.append(point_address)
        return False

    def get_output_layer_objects(self, address_points,
                                 feedback):
        """Actually do the algorithm. Dang it, now I have to figure that out"""
        feedback.pushInfo("Starting to process addresses + parcels")

        self.output_layers = OutputLayers()

        # Gather address points by base address
        self.address_dict = self.make_address_dict(address_points)

        for address, points in self.address_dict.items():
            # Just one point -> decide whether to label the point or the
            # polygon based on denseness of the area (as a proxy, use width
            # + area of parcel if it's the correct parcel type)
            if len(points) == 1:
                point = points.pop()
                geometry = point.point
                added_as_parcel = self.add_as_point_or_parcel(geometry, point.label,
                                                              point.parcel_id)
                continue
                #     # TODO: don't hardcode these values
                #     # Also maybe optimize them??
                #     is_small = parcel.width < 200 or parcel.area < 2000
                #     # Filter out road/water parcels
                #     is_plot = parcel.parcel_type == self.parcel_type_value
                #     only_address_in_parcel = len(parcel.address_points) == 1
                #     if is_small and is_plot and only_address_in_parcel:
                #         # Small parcel -> put label inside parcel
                #         polygon = PolygonAddress(parcel.geometry,
                #                                  point.label)
                #         self.output_layers.dense_address_polygons.append(polygon)
                #         continue
                # # If we reach here, we have just one address point for the
                # # address, and it's either a) not in a dense area or b) has
                # # multiple address points in the same parcel -> put the address
                # # label on the point
                # # TODO: can we do anything smarter with multiple address
                # # points in a single parcel?
                # point_address = PointAddress(point.point, point.label)
                # self.output_layers.boring_addresses.append(point_address)
                # continue

            # If we reach here, we have multiple points for this address,
            # which usually means multiple buildings/units
            n_points = len(points)
            units = set(point.unit for point in points
                        if point.unit is not None)
            buildings = set(point.building for point in points
                            if point.building is not None)
            # One not-uncommon case (about 20% on Fidalgo Island) is where
            # there are multiple points but they're not different
            # units/buildings. This might mean there are points per room or
            # floor, which we *really* don't care about for the mapbook
            if len(units) == 0 and len(buildings) == 0:
                # Use the weighted centroid of point locations. This is
                # subtly different from the centroid, but much easier to
                # calculate. And frankly if we have three points on one
                # side and one on the other, it kind of makes sense for the
                # actual label to be closer to the three points rather than
                # at the true center
                geom_points = [point.point.asMultiPoint() for point in points]
                x_coords = [geom[0].x() for geom in geom_points]
                y_coords = [geom[0].y() for geom in geom_points]
                new_point_coords = (sum(x_coords) / len(x_coords),
                                    sum(y_coords) / len(y_coords))
                new_point = QgsPointXY(*new_point_coords)
                point_geometry = QgsGeometry.fromPointXY(new_point)
                # Since there are no units or buildings, all labels should match
                label = points.pop().label
                # point_address = PointAddress(point_geometry, label)
                # output_layers.boring_addresses.append(point_address)
                # continue
                parcel_ids_for_points = [point.parcel_id
                                         for point in points
                                         if point.parcel_id is not None]
                if len(set(parcel_ids_for_points)) == 1:
                    parcel_id = parcel_ids_for_points[0]
                    added_as_parcel = self.add_as_point_or_parcel(
                            point_geometry, label, parcel_id,
                            use_full_address=False)
                    # feedback.pushInfo(f"multiple points matching address: Added {address} as {'parcel' if added_as_parcel else 'point'}")
                    continue
                # If we reach here, we label on the centroid
                point_address = PointAddress(point_geometry, label)
                self.output_layers.boring_addresses.append(point_address)
                # feedback.pushInfo(f"multiple points matching address: put {address} at centroid point")
                continue

            # Special case: there are two points with separate
            # building/unit information. If they're really close together,
            # just label midpoint with the address. If they're further
            # apart, label each individually.
            if len(points) == 2:
                geom_points = [point.point.asMultiPoint() for point in points]
                x_coords = [geom[0].x() for geom in geom_points]
                y_coords = [geom[0].y() for geom in geom_points]
                point1 = QgsPointXY(x_coords[0], y_coords[0])
                distance = point1.distance(x_coords[1], y_coords[1])
                # Get distance between the points (in crs units (grrr))
                if distance < 200:  # idk, total guess
                    # They're really close together: label midpoint
                    new_point_coords = (sum(x_coords) / len(x_coords),
                                        sum(y_coords) / len(y_coords))
                    new_point = QgsPointXY(*new_point_coords)
                    point_geometry = QgsGeometry.fromPointXY(new_point)
                    label = random.choice(list(points)).label_address_only

                    parcel_ids_for_points = [point.parcel_id
                                             for point in points
                                             if point.parcel_id is not None]
                    if len(set(parcel_ids_for_points)) == 1:
                        parcel_id = parcel_ids_for_points[0]
                        added_as_parcel = self.add_as_point_or_parcel(
                                point_geometry, label, parcel_id,
                                use_full_address=False)
                        # feedback.pushInfo(f"2 points within short distance: Added {address} as {'parcel' if added_as_parcel else 'point'}")
                        continue
                    # Multiple parcels: abdicate and label on centroid
                    point_address = PointAddress(point_geometry, label)
                    self.output_layers.boring_addresses.append(point_address)
                    # feedback.pushInfo(f"2 points within short distance with multiple parcels: put {address} at centroid point")
                    continue

                # Further apart: label both
                for point in points:
                    point_address = PointAddress(point.point, point.label)
                    self.output_layers.boring_addresses.append(point_address)
                # feedback.pushInfo(
                #         f"Labeling {address} on individual parts"
                #         f"distance {distance} is larger. Has "
                #         f"{len(points)} points, inc "
                #         f"{len(units)} units ({units}) and"
                #         f"{len(buildings)} buildings ({buildings})")
                continue


            # Now we know there are 3+ points and multiple
            # buildings/units. In general we are going to:
            # 1. Create a bounding geometry for the points.
            # 2. Based on the size of that bounding geometry and total
            #    number of points, decide whether to label just one point,
            #    label all the points individually, or actually draw the
            #    bounding hull and optionally label subpoints

            # Create bounding hull

            geom_points = [point.point.asMultiPoint() for point in points]
            coords = np.array([[geom[0].x(), geom[0].y()]
                               for geom in geom_points])
            hull = ConvexHull(points=coords)

            xspan, yspan = (np.max(hull.points, axis=0) -
                            np.min(hull.points, axis=0))

            feedback.pushInfo(f"Generated hull for {address} ({n_points}). It has area "
                              f"{hull.area:.0f},  xspan {xspan:.0f}, yspan {yspan:.0f}")
            # This is very much a first pass, I'm in a hurry to have a
            # draft sort of logic. But based on a cursory glance...
            if xspan >= 200 or yspan >= 200 or hull.area >= 500:
                # Outline wrapper hull and label within it
                geometry = QgsGeometry.fromPolygonXY(
                        [[QgsPointXY(*pt) for pt in
                          hull.points[hull.vertices]]])
                # Label large polygons with full address, small with just
                # the number
                if hull.area > 900:
                    label = address.street_address
                else:
                    label = address.address_number
                wrapper_polygon = PolygonAddress(geometry, label,
                                                 scope="address")
                self.output_layers.wrapper_polygons.append(wrapper_polygon)
                feedback.pushInfo(f"  Added as wrapper polygon woot {label}")
                continue

            # Merge the damn points and label on point/parcel like before
            x_coords = [geom[0].x() for geom in geom_points]
            y_coords = [geom[0].y() for geom in geom_points]
            new_point_coords = (sum(x_coords) / len(x_coords),
                                sum(y_coords) / len(y_coords))
            new_point = QgsPointXY(*new_point_coords)
            point_geometry = QgsGeometry.fromPointXY(new_point)
            parcel_ids_for_points = [point.parcel_id
                                     for point in points
                                     if point.parcel_id is not None]
            label = address.address_number
            if len(set(parcel_ids_for_points)) == 1:
                parcel_id = parcel_ids_for_points[0]
                added_as_parcel = self.add_as_point_or_parcel(
                        point_geometry, label, parcel_id,
                        use_full_address=False)
                feedback.pushInfo(f"  big-ass address thing: Added {address} as {'parcel' if added_as_parcel else 'point'}")
                continue
            # If we reach here, we label on the centroid
            point_address = PointAddress(point_geometry, label)
            self.output_layers.boring_addresses.append(point_address)
            feedback.pushInfo(f"  big-ass address thing: Added {address} as point b/c more than one parcel")
            continue


            # feedback.pushInfo(f"Skipping address {address} for now, as it has "
            #                   f"{n_points} points, inc {len(units)} "
            #                   f"units ({units}) and {len(buildings)} "
            #                   f"buildings ({buildings})")

        return self.output_layers

    def processAlgorithm(
        self, parameters, context, feedback
    ):  # pylint: disable=invalid-name
        """
        Here is where the processing itself takes place.
        """
        (
            address_source,
            street_number_field,
            address_street_name_expression,
            address_building_field,
            address_unit_field,
            address_town_field
        ) = self._get_address_inputs(parameters, context)

        (
            _,
            parcel_type_field,
            parcel_type_value
        ) = self._get_parcel_inputs(parameters, context)
        self.parcel_type_value = float(parcel_type_value)
        # TODO: accept as input rather than hardcoding
        global_id_field = "GlobalID"

        dest_ids = self._configure_sinks(parameters, context, address_source.sourceCrs())

        address_clip_layer, parcel_clip_layer = self.clip_layers(parameters, feedback)

        # Hard code these two fields
        area_field = "area"
        width_field = "width"
        fields = {
            "street_number": street_number_field,
            "street_name_expression": address_street_name_expression,
            "building": address_building_field,
            "unit": address_unit_field,
            "town": address_town_field,
            "parcel_type": parcel_type_field,
            "parcel_value": self.parcel_type_value,
            "parcel_global_id": global_id_field,
            "area": area_field,
            "width": width_field,
        }

        parcel_clip_layer_with_area = self.add_geom_fields(
                parcel_clip_layer, fields, feedback)

        feedback.pushInfo("About to create spatial indexes")
        processing.run("native:createspatialindex",
                       {'INPUT':address_clip_layer})
        processing.run("native:createspatialindex",
                       {'INPUT':parcel_clip_layer_with_area})

        feedback.pushInfo("About to do the join")

        # Have to save the layer or we can't specify we should skip invalid
        # geometries
        # Why there are invalid geometries is left as an exercise to the
        # reader; doing these steps via the GUI doesn't give any issues :(
        # TODO: fixed the invalid geometry thing, remove this new layer
        QgsProject.instance().addMapLayer(parcel_clip_layer_with_area)
        address_with_parcel_layer = processing.run(
            "native:joinattributesbylocation",
            {'INPUT':address_clip_layer,
             'PREDICATE':[0],
             'JOIN':QgsProcessingFeatureSourceDefinition(
                 parcel_clip_layer_with_area.source(),
                 selectedFeaturesOnly=False,
                 featureLimit=-1,
                 flags=QgsProcessingFeatureSourceDefinition.FlagOverrideDefaultGeometryCheck,
                 geometryCheck=QgsFeatureRequest.GeometrySkipInvalid),
             'JOIN_FIELDS':[],
             'METHOD':0,
             'DISCARD_NONMATCHING':False,
             'PREFIX':'',
             'OUTPUT':'TEMPORARY_OUTPUT'},
            context=context,
            feedback=feedback)["OUTPUT"]

        feedback.pushInfo(f"Joined layer: {address_with_parcel_layer}")

        self.parcel_dict = self.get_parcels(parcel_clip_layer_with_area,
                                            fields, feedback)

        address_points = self.get_address_points(address_with_parcel_layer,
                                                 fields, feedback)

        output_layer_objects = self.get_output_layer_objects(
                address_points, feedback)

        # Process these python objects back into QGIS layers
        for boring_address in output_layer_objects.boring_addresses:
            feature = QgsFeature(self.output_fields[self.ADDRESS_POINTS])
            feature.setAttributes([boring_address.label])
            feature.setGeometry(boring_address.point)
            self.address_points_sink.addFeature(feature,
                                                QgsFeatureSink.FastInsert)

        for dense_address_polygon in output_layer_objects.dense_address_polygons:
            feature = QgsFeature(self.output_fields[self.ADDRESS_POLYGONS])
            feature.setAttributes([dense_address_polygon.label])
            feature.setGeometry(dense_address_polygon.polygon)
            self.address_polygons_sink.addFeature(
                    feature, QgsFeatureSink.FastInsert)

        for wrapper_polygon in output_layer_objects.wrapper_polygons:
            feature = QgsFeature(self.output_fields[self.WRAPPER_POLYGONS])
            feature.setAttributes([wrapper_polygon.label,
                                   wrapper_polygon.scope])
            feature.setGeometry(wrapper_polygon.polygon)
            self.wrapper_polygons_sink.addFeature(
                    feature, QgsFeatureSink.FastInsert)

        for sub_address_point in output_layer_objects.sub_address_points:
            feature = QgsFeature(self.output_fields[self.ADDRESS_SUB_POINTS])
            feature.setAttributes([sub_address_point.label])
            feature.setGeometry(sub_address_point.point)
            self.address_sub_points_sink.addFeature(
                    feature, QgsFeatureSink.FastInsert)

        # First pass: just return address points
        # address_layer = address_source
        # for feature in address_layer.getFeatures():
        #     feature.setFields(address_layer.fields(), False)
        #     address_number = feature[street_number_field]
        #     address_geometry = feature.geometry()
        #     # Make sure order matches field definition above
        #     attributes = [
        #         address_number,
        #     ]
        #     feature = QgsFeature(self.output_fields[self.ADDRESS_POINTS])
        #     feature.setAttributes(attributes)
        #     feature.setGeometry(address_geometry)
        #     self.address_points_sink.addFeature(feature, QgsFeatureSink.FastInsert)

        return dest_ids

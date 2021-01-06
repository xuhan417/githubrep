#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import weakref

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

# note: need to change to path to behavior agent 
# CARLA default path: agents.navigation.behavior_agent
from path.to.behavior_agent import BehaviorAgent  

# ==============================================================================
# -- ADS regs APi and regulation look up ---------------------------------------
# ==============================================================================
from ADS_API import read_info, read_all_ADS_behavior, filter_moves_from_regs

# ------------------------- ADS Regs Workflow NOTE: ----------------------------
# 1. read_all_ADS_behavior: read all behavior from ADS server 
# input:      lat=lat of ego vehicle, 
#             lon= long of ego vehicle, 
#             jur_id_index=index of prefered jurisdiction
# output:     location range [minlat, maxlat, minlon, maxlon]
#             all behavior for current location from ADS server
# ------------------------------------------------------------------------------
# 2. filter_moves_from_regs: filter behaviors based on current situation  
# input:      ego_vector: behavior vector of ego vehicle
#             all_ads_situations: all regulations from ADS
# output:     all qulified behavior under current traffic situation 
# ------------------------------------------------------------------------------
# 3. Read info: transform string regulations into 1x4 behavior vector 
# input:                       ADS regulation form 
# output:                      transformed vector
# behavior vector form:        [action, direction, situation, signal] 
# ------------------------------ END of ADS Regs -------------------------------

from turn_on_red_cost_fn import determine_lane, turn_cost


# ==============================================================================
# -- helper functions ----------------------------------------------------------
# ==============================================================================
def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def arrange_yaw(yaw):
    """Method to re-arrange yaw within 0 to 360""" 
    if yaw<0: yaw = 360 + yaw 
    else: yaw = yaw  
    return yaw  

def is_turning_right(future_yaw, current_yaw):
    """determine if the vehicle is turning right """
    if future_yaw > current_yaw: 
        return True
    else: 
        return current_yaw-future_yaw <= 180

def distance_to_ego(NPC_loc, ego_transform):
    """ determine the distance between ego and current NPC """
    return math.sqrt((NPC_loc.x - ego_transform.location.x)**2 + (NPC_loc.y - ego_transform.location.y)
                     ** 2 + (NPC_loc.z - ego_transform.location.z)**2)

def transform_loc(x,y):
    '''
    function that transform simulation coordinates to lat lon coordinates
    #-------------------------------- location transform note ------------------------------
        1. Simulation Right turn on red leagal area: x~(-47.53, 27.46); y~(-84.63, -4.63)
        2. ADS Regs right turn on red leagal area: lat~(38.4, 41.98); lon~(-84.81, -80.51)
        3. Simulation Coordinates to lat/lon transform: from x->lat; y->lon
    # --------------------------------------------------------------------------------------
    '''
    lat = ((x-(-47.53-5))/(27.46+47.53+10)) *(41.98-38.4) + 38.4
    lon = ((y-(-84.63-5))/(84.63-4.63+10)) *(84.81-80.51) + (-84.81)
    return lat,lon

def ego_intention(isJunctiuon, right_lanemarking_type, waypoint_buffer, agent, ego_vehicle_wp, hud, debug=True):
    '''
    this function detemine the ego vehicle's intention and if it is leagal to right turn 
    on red traffic light 
    '''
    # read light state 
    light_state = str(agent.vehicle.get_traffic_light_state())
    # cast to lower case letter 
    if light_state[0] == 'R': light_state='red'
    elif light_state[0] == 'G': light_state='green'
    elif light_state[0] == 'Y': light_state='yellow'

    # intention not near junction (default)
    # if not isJunctiuon: ego_intention = ['proceed','','','']
    ego_intention = ['proceed','','','']

    # intention near junction but not at right lane 
    if isJunctiuon and (right_lanemarking_type != 'Solid'):
        ego_intention = ['proceed', 'at', 'intersection', light_state]

    # intention when near junction 
    if isJunctiuon and (right_lanemarking_type == 'Solid') and len(waypoint_buffer)>0:  # and (ego_direction == 'RoadOption.RIGHT' or ego_incoming_direction == 'RoadOption.RIGHT'):
        # find current target 
        end_waypoint = agent.end_waypoint
        # find yaw
        current_yaw = ego_vehicle_wp.transform.rotation.yaw
        future_yaw = end_waypoint.transform.rotation.yaw
        # rearrange the values within 0~180 and -180~0
        current_yaw = arrange_yaw(current_yaw)
        future_yaw = arrange_yaw(future_yaw)

        # print state 
        if debug:
            print('Current yaw: ')
            print(current_yaw)
            print('Future Yaw: ') 
            print(future_yaw)
            print('Is turning: ' + str(is_turning_right(future_yaw, current_yaw)))
            print('----------------------')

        # determine if ignore red light
        if is_turning_right(future_yaw, current_yaw): 
            # yaw greater means direction is to the right 
            # ok to ignore 
            if debug: print('!!! Right on RED !!!')
            hud.notification('Right Turn On Red !')
            agent.ignore_traffic_light = False
            # define agent intention vector 
            ego_intention = ['turn', 'right', 'intersection', 'red']
        
    return ego_intention

# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================
class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Set the seed if requested by user
        if args.seed is not None:
            random.seed(args.seed)

        # Choose mkz blueprint.
        blueprint = self.world.get_blueprint_library().find('vehicle.lincoln.mkz2017')
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = blueprint.get_attribute('color').recommended_values[0]
            blueprint.set_attribute('color', color)
        # Spawn the player.
        print("Spawning the player")
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        # spawn a vehicle in the world under agent control 
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            # define initiate spawn point
            default_spawn_points = self.map.get_spawn_points()
            spawn_point = carla.Transform(default_spawn_points[106].location)
            self.player = self.world.try_spawn_actor(blueprint, spawn_point) # player --> vehicle 
        # Set up the sensors.
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor,
            # self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================
class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================
class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 26.0f FPS' % self.server_fps,
            'Client:  % 26.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 30s' % 'CARLA Simulated Vehicle',
            'Map:     % 30s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 25.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 26.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 30s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 34s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 28.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            # ADS info
            '',
            'ADS Regs API info:',
            'Ego Lat-Lon: % 8s' % ('[%f, %f]' % (ego_lat, ego_lon)),
            'Applicable ADS regulations count:% 5d' %0, 
            'Is ego in right-turn-on-red zone: % 3s' %'False', 
            'ego intention vector:',
            '%s' %str(ego_vector),
            'Is ego-intention legal: % 15s' %str(is_legal),
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        # find ego vehicle yaw 
        ego_veh_yaw = transform.rotation.yaw

        for dist, vehicle in sorted(vehicles):
            # only record vehicles within 50 m
            if dist > 20.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            vehicle_yaw = vehicle.get_transform().rotation.yaw
            self._info_text.append('% 4dm %s %id' % (dist, vehicle_type, vehicle_yaw))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            # info_surface = pygame.Surface((220, self.dim[1]))
            info_surface = pygame.Surface((290, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================
class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================
class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================
class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================
class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================
class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud, gamma_correction):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(
                carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
                if blp.has_attribute('gamma'):
                    blp.set_attribute('gamma', str(gamma_correction))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================
def game_loop(args):
    """ Main loop for agent"""

    pygame.init()
    pygame.font.init()
    world = None
    tot_target_reached = 0
    num_min_waypoints = 21

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        # load world
        if args.town == 'Town04':
            world = client.load_world('Town04')
        else:
            world = client.load_world('Town05')
            start_loc = carla.Location(x=-87, y=-84.5, z=-0.09)
            start_rotation = carla.Rotation(yaw=0)
        # sychronous mode
        settings = world.get_settings()
        settings.sychronous_mode = True
        settings.fixed_delta_seconds = 0.1
        # get TM manager 
        traffic_manager = client.get_trafficmanager(args.tm_port)
        # pygame HUD 
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        hud = HUD(args.width+15, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)
        # initiate agent 
        agent = BehaviorAgent(world.player, behavior=args.behavior, ignore_traffic_light=False)
        # spaw ego vehicle 
        spawn_points = world.map.get_spawn_points()
        # initial target 
        if spawn_points[125].location != agent.vehicle.get_location():
            # agent.vehicle.set_rotation(start_rotation)
            destination = spawn_points[92].location
        else:
            destination = spawn_points[72].location

        agent.set_destination(agent.vehicle.get_location(), destination, clean=True)
        clock = pygame.time.Clock()

        #------------------------- check initial location and ADS regs --------------------------
        # initiate global variable for HUD display 
        global ego_lat
        global ego_lon
        global loc_range
        global all_ads_regulations
        global ego_vector
        global is_legal 
        global is_within_regulation_range
        cost_threshold = 10

        ego_vehicle_loc = agent.vehicle.get_location()
        ego_x,ego_y,ego_z = ego_vehicle_loc.x, ego_vehicle_loc.y,ego_vehicle_loc.z
        ego_lat,ego_lon = transform_loc(ego_x,ego_y)
        # read ADS behavior based on initial lat,lon
        loc_range, all_ads_regulations = read_all_ADS_behavior()
        # check range 
        is_within_regulation_range = ( (loc_range[0]<=ego_lat<=loc_range[1]) and\
                                           (loc_range[2]<=ego_lon<=loc_range[3]) )
        # read all ADS data and print result 
        ego_vector = ['proceed','','','']
        all_leagal_moves = filter_moves_from_regs(ego_vector, all_ads_regulations)
        is_legal = (ego_vector in all_leagal_moves)

        # SIM loop 
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events():
                return
            if not world.world.wait_for_tick(10.0):
                continue

            # current ego info
            agent.update_information(world)
            world.tick(clock)
            world.render(display)
            pygame.display.flip()
            ego_vehicle_loc = agent.vehicle.get_location()
            ego_vehicle_wp = agent._map.get_waypoint(ego_vehicle_loc)
            curr_road_ID =  ego_vehicle_wp.road_id
            curr_lane_ID = ego_vehicle_wp.lane_id
            isJunctiuon = agent.vehicle.is_at_traffic_light()
            vel_3d = agent.vehicle.get_velocity()
            vel = vel_3d.x**2 + vel_3d.y**2

            # more traffic info 
            right_lanemarking_type = str(ego_vehicle_wp.right_lane_marking.type)
            ego_incoming_direction = str(agent.incoming_direction)
            ego_direction = str(agent.direction)
            waypoint_queue = agent._local_planner.waypoints_queue
            waypoint_buffer = agent._local_planner._waypoint_buffer

            # check vehicle location 
            ego_x,ego_y,ego_z = ego_vehicle_loc.x, ego_vehicle_loc.y,ego_vehicle_loc.z
            ego_lat,ego_lon = transform_loc(ego_x,ego_y)
            is_within_regulation_range = ( (loc_range[0]<=ego_lat<=loc_range[1]) and\
                                           (loc_range[2]<=ego_lon<=loc_range[3]) )

            # ------------------------- routing and rerouting -------------------------------
            if len(agent.get_local_planner().waypoints_queue) <= 5 and args.loop:
                # agent.reroute(spawn_points)
                print("Target almost reached, setting new destination...")
                tot_target_reached += 1
                world.hud.notification("The target has been reached " +
                                       str(tot_target_reached) + " times.", seconds=4.0)

                # redefine new start point 
                new_start = agent._local_planner.waypoints_queue[-1][0].transform.location
                # start reroute by assigning new targets 
                if tot_target_reached == 1 or tot_target_reached%2 == 1: 
                    destination = spawn_points[238].location
                    print('Target has been reached ' + str(tot_target_reached) + ' times')
                    print("New destination: " + str(destination))
                    agent.set_destination(new_start, destination)
                else: 
                    destination = spawn_points[12].location
                    print('Target has been reached ' + str(tot_target_reached) + ' times')
                    print("New destination: " + str(destination))
                    agent.set_destination(new_start, destination)
                
            elif len(agent.get_local_planner().waypoints_queue) == 0 and not args.loop:
                print("Target reached, mission accomplished...")
                break

            # ------------------------- red light behavior -------------------------------
            if tot_target_reached <=1: 
                # stop at the first light 
                agent.ignore_traffic_light = False
                ego_vector = ego_intention(isJunctiuon, right_lanemarking_type, \
                                            waypoint_buffer, agent, ego_vehicle_wp, hud)
                # read all ADS legal behaviors  
                all_leagal_moves = filter_moves_from_regs(ego_vector, all_ads_regulations)
                # determin legality 
                if is_within_regulation_range and \
                    ego_vector[0:3]==['proceed', 'at', 'intersection']: is_legal=False
                elif 'red' in ego_vector: is_legal=False
                else: is_legal = (ego_vector in all_leagal_moves)

            else:
                # find ego intention 
                ego_vector = ego_intention(isJunctiuon, right_lanemarking_type, \
                                            waypoint_buffer, agent, ego_vehicle_wp, hud)
                # read all ADS legal behaviors 
                all_leagal_moves = filter_moves_from_regs(ego_vector, all_ads_regulations)
                
                # determin legalality  
                if is_within_regulation_range and \
                    ego_vector[0:3]==['proceed', 'at', 'intersection']: 
                    is_legal=False
                elif 'red' in ego_vector: 
                    is_legal=False
                else: 
                    is_legal = (ego_vector in all_leagal_moves)

            # ignore red light
            cost = turn_cost(agent)
            if is_legal and cost < cost_threshold: 
                agent.ignore_traffic_light = True

            # apply control for 1 step 
            speed_limit = world.player.get_speed_limit()
            agent.get_local_planner().set_speed(speed_limit)

            control = agent.run_step(debug=True)
            world.player.apply_control(control)

    finally:
        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument("-a", "--agent", type=str,
                           choices=["Behavior", "Roaming", "Basic"],
                           help="select which agent to run",
                           default="Behavior")
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    argparser.add_argument(
        '-t', '--town', type=str,
        choices=["Town04", "Town01"],
        help='Choose one of the town maps (Town04, Town01) ',
        default='Town01')
    # add args for traffic manager port (tm-port)
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()

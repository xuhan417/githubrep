"""Spawn NPCs into the simulation for freeway scenario"""

import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from carla import VehicleLightState as vls

import argparse
import logging
import random


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
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
        '-n', '--number-of-vehicles',
        metavar='N',
        default=7,
        type=int,
        help='number of vehicles (default: 10)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=0,
        type=int,
        help='number of walkers (default: 0)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='pedestrians filter (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Enanble')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enanble car lights')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False

    try:
        # read world of the behavioral agent 
        world = client.get_world()
        # initiate traffic manager for background traffic control
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
        # set synchronous mode 
        if args.sync:
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                world.apply_settings(settings)
            else:
                synchronous_master = False

        # read blueprints and spawn vehicles
        blueprints = world.get_blueprint_library().filter(args.filterv)
        blueprintsWalkers = world.get_blueprint_library().filter(args.filterw)
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        # vehicles for safe condition
        if args.safe:
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
        # read all spawn points
        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        # imports
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        
        # 1. next to 261, left lane
        loc263 = spawn_points[263].location
        rotation263 = spawn_points[263].rotation
        new_spwan_location_1 = carla.Location(loc263.x+40, \
                                              loc263.y,\
                                              loc263.z)
        new_spwan_1 = carla.Transform(new_spwan_location_1, rotation263) 

        # 2. next to 261, mid lane 
        loc262 = spawn_points[262].location
        rotation262 = spawn_points[262].rotation
        new_spwan_location_2 = carla.Location(loc262.x+23, \
                                              loc262.y,\
                                              loc262.z)
        new_spwan_2 = carla.Transform(new_spwan_location_2, rotation262) 

        # 3. next to 261, in curve 
        new_spwan_location_3 = carla.Location(loc263.x+40+20, \
                                              loc263.y+5,\
                                              loc263.z)
        new_spwan_3 = carla.Transform(new_spwan_location_3, rotation262) 

        # 4. next to 262, in curve mid lane
        new_spwan_location_4 = carla.Location(loc263.x+65, \
                                              loc263.y+10,\
                                              loc263.z)
        new_spwan_4 = carla.Transform(new_spwan_location_4, rotation262) 

        # 5. left lane
        start_loc = spawn_points[269].location
        start_rotation = spawn_points[269].rotation
        spwan_location = carla.Location(start_loc.x+58+32, \
                                        start_loc.y-3,\
                                        start_loc.z)
        spawn_point_mid = carla.Transform(spwan_location, start_rotation) 

        # 6, 7 further locations

        loc_179 = spawn_points[179].location
        loc_189 = spawn_points[189].location
        rot_179 = spawn_points[179].rotation

        new_spwan_location_6 = carla.Location(loc_179.x, \
                                              loc_179.y-125,\
                                              loc_179.z)
        new_spwan_location_7 = carla.Location(loc_189.x, \
                                              loc_189.y-125,\
                                              loc_189.z)

        new_spwan_6 = carla.Transform(new_spwan_location_6, rot_179) 
        new_spwan_7 = carla.Transform(new_spwan_location_7, rot_179) 

        # store all spawn points 
        selected_spawn_points = [spawn_points[193],spawn_points[263], \
                                new_spwan_1, new_spwan_3, spawn_point_mid,\
                                new_spwan_6, new_spwan_7]
        batch = []

        # spawn vehicles
        for n, transform in enumerate(selected_spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
        # log reponse 
        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)
                cur_actor = world.get_actor(response.actor_id)
                traffic_manager.auto_lane_change(cur_actor, False)

        # wait for a tick to ensure client receives updated info
        if not args.sync or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        # set background traffic speed
        traffic_manager.global_percentage_speed_difference(50.0)

        while True:
            if args.sync and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()

    # clear simulation when finished 
    finally:

        if args.sync and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')

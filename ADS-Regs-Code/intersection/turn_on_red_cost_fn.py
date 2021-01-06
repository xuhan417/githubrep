import math 
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
import numpy as np 
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

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.roaming_agent import RoamingAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
# from agents.navigation.AV_agent import AVAgent  # pylint: disable=import-error


def determine_lane(ego_yaw, veh_yaw):
	'''
	determine lanes of nearby vehicle's by orientation
	input: 
		ego-yaw: the yaw angle of ego veh 
		veh-yaw: the yaw angle of neaby veh
	output: 
		veh-lane: the lane assignment of the nearby veh 
		('target', 'opposite','other')
	'''
	
	# positive yaw --> clockwise from current angle 
	err = 15
	# target lane 
	target_lane_yaw = ego_yaw + 90
	neg_target_lane_yaw = target_lane_yaw - 360
	# opposite lane 
	opposite_lane_yaw = ego_yaw + 180
	neg_opposite_yaw = opposite_lane_yaw - 360
	# determine lane 
	if (target_lane_yaw-err  <= veh_yaw <= target_lane_yaw+err) or\
		(neg_target_lane_yaw-err  <= veh_yaw <= neg_target_lane_yaw+err):
		# target lane 
		veh_lane = 'target'
	elif (opposite_lane_yaw-err  <= veh_yaw <= opposite_lane_yaw+err) or\
		(neg_opposite_yaw-err  <= veh_yaw <= neg_opposite_yaw+err):
		# opposite lane 
		veh_lane = 'opposite'
	else: 
		# other 
		veh_lane = 'other'
	return veh_lane




def turn_cost(agent):
	# ego position 
	ego_vehicle = agent.vehicle
	ego_yaw = ego_vehicle.get_transform().rotation.yaw
	ego_vehicle_loc = ego_vehicle.get_location()
	ego_id = ego_vehicle.id 
	world = ego_vehicle.get_world()
	# speed limit 
	speed_limit = ego_vehicle.get_speed_limit()
	# read nearby vehicle 
	vehicle_list = world.get_actors().filter("*vehicle*")
	# helper function to find distance 
	def dist(v):
		return math.sqrt((v.get_location().x - ego_vehicle_loc.x)**2 + \
						 (v.get_location().y - ego_vehicle_loc.y)**2 + \
						 (v.get_location().z - ego_vehicle_loc.z)**2)
	# find opposite vehicle
	opposite_lane_vlist = [[(dist(v)),v] for v in vehicle_list \
						if dist(v) < 15 and v.id != ego_id and \
						determine_lane(ego_yaw, v.get_transform().rotation.yaw)=='opposite']
	# sort veh list based on distance 
	opposite_lane_vlist.sort()

	# find target lane vehicle 
	target_lane_vlist = [[dist(v),v] for v in vehicle_list \
						if dist(v) < 15 and v.id != ego_id and \
						determine_lane(ego_yaw, v.get_transform().rotation.yaw)=='target']
	# sort based on distance 
	target_lane_vlist.sort()

	# helper function to transfer velocity from 3D to 1D 
	def transfer_3d(velocity):
		return math.sqrt( (velocity.x)**2 + (velocity.y)**2 + (velocity.z)**2 )

	# target lane traffic
	if len(target_lane_vlist) > 1: 
		min_2_dist_tar = np.sum([target_lane_vlist[0][0], target_lane_vlist[1][0]])
		min_veh_speed_tar = np.mean([ transfer_3d(target_lane_vlist[0][1].get_velocity()), \
						  transfer_3d(target_lane_vlist[1][1].get_velocity()) ])
	else: 
		min_2_dist_tar = 0
		min_veh_speed_tar = 0

	# opposite lane traffic 
	if len(opposite_lane_vlist) > 1:
		min_distance_opposite = opposite_lane_vlist[0][0]
		min_veh_speed_opposite = transfer_3d(opposite_lane_vlist[0][1].get_velocity())
	else: 
		min_distance_opposite = 0
		min_veh_speed_opposite = 0

	# total cost 
	cost = 1/(1+min_2_dist_tar) + (min_veh_speed_tar/speed_limit) \
			+ 1/(1+min_distance_opposite) + (min_veh_speed_opposite/speed_limit)

	return cost
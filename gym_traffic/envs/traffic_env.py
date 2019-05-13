from gym import Env
from gym import error, spaces, utils
from gym.utils import seeding
from scipy.misc import imread
from gym import spaces
from string import Template
import os, sys
import numpy as np
import math
import time
from gym_traffic.envs.traffic_lights import TrafficLightTwoWay, TrafficLight

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import traci.constants as tc

class TrafficEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, lights, netfile, routefile, addfile, guifile="", lanes=[], exitloops=[],
                 tmpfile="tmp.rou.xml",
                 pngfile="tmp.png", mode="gui", simulation_end=3600, sleep_between_restart=1):
        # "--end", str(simulation_end),
        self.simulation_end = simulation_end
        self.sleep_between_restart = sleep_between_restart
        self.mode = mode
        self._seed()
        self.loops = [] #loops are defined in start_sumo
        self.exitloops = exitloops
        self.loop_variables = [tc.LAST_STEP_MEAN_SPEED, tc.LAST_STEP_TIME_SINCE_DETECTION, tc.LAST_STEP_VEHICLE_NUMBER]
        self.lanes = lanes
        self.detectors = []
        self.args = ["--net-file", netfile, "--route-files", tmpfile, "--additional-files", addfile, "-W"]
        if mode == "gui":
            binary = "sumo-gui"
            addon = ["-S", "-Q"]
        else:
            binary = "sumo"
            addon = ["--no-step-log"]
        with open(routefile) as f:
            self.route = f.read()
        self.tmpfile = tmpfile
        self.pngfile = pngfile
        self.sumo_cmd = [binary] + self.args + addon
        self.sumo_step = 0
        self.sumo_running = False
        self.viewer = None
        self.start_sumo(True)
        self.stop_sumo()

    def relative_path(self, *paths):
        os.path.join(os.path.dirname(__file__), *paths)

    def write_routes(self):
        self.route_info = self.route_sample()
        #with open(self.tmpfile, 'w') as f:
            #f.write(Template(self.route).substitute(self.route_info))

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def get_cars_in_lanes(self):
        obs = []
        for loop in self.loops:
            res = traci.inductionloop.getSubscriptionResults(loop)
            obs.append(res[traci.constants.LAST_STEP_VEHICLE_NUMBER])
        return obs
        """
        if self.sumo_running:
            for light_id in traci.trafficlight.getIDList():
                for lane_id in traci.trafficlight.getControlledLanes(light_id):
                    num_vehicles.append(traci.lane.getSubscriptionResults(lane_id)[traci.constants.LAST_STEP_VEHICLE_NUMBER])
        return num_vehicles
        """

    def get_light_actions(self):
        phases = []
        if self.sumo_running:
            for light_id in traci.trafficlight.getIDList():
                phases.append(traci.trafficlight.getPhase(light_id))
        return phases

    def start_sumo(self, dataCollection=False):
        if not self.sumo_running:
            self.write_routes()
            if not dataCollection:
                traci.start(self.sumo_cmd)
                self.screenshot()
            else:
                traci.start(["sumo"] + self.args + ["--no-step-log"])
            self.loops = [loopID for loopID in traci.inductionloop.getIDList()]
            for loopid in self.loops:
                traci.inductionloop.subscribe(loopid, self.loop_variables)
            self.detectors = [entryexitID for entryexitID in traci.multientryexit.getIDList()]
            self.lights = []
            for lightID in traci.trafficlight.getIDList():
                temp = traci.trafficlight.getCompleteRedYellowGreenDefinition(lightID)
                self.lights.append(TrafficLight(lightID, [phase._phaseDef for phase in temp[0]._phases]))
            self.action_space = spaces.MultiDiscrete([len(light.actions) for light in self.lights])
            trafficspace = spaces.Box(low=float('-inf'), high=float('inf'),
                                      shape=(len(self.loops) * len(self.loop_variables),))
            lightspaces = spaces.MultiDiscrete([len(light.actions) for light in self.lights])
            self.observation_space = spaces.Tuple([trafficspace, lightspaces])
            self.sumo_step = 0
            self.sumo_running = True


    def stop_sumo(self):
        if self.sumo_running:
            traci.close()
            self.sumo_running = False

    def _reward(self):
        # reward = 0.0
        # for lane in self.lanes:
        #    reward -= traci.lane.getWaitingTime(lane)
        # return reward
        reward = 0
        for d in self.detectors:
            speed = traci.multientryexit.getLastStepMeanSpeed(d)
            count = traci.multientryexit.getLastStepVehicleNumber(d)
            reward += speed * count
        # print("Speed: {}".format(traci.multientryexit.getLastStepMeanSpeed(self.detector)))
        # print("Count: {}".format(traci.multientryexit.getLastStepVehicleNumber(self.detector)))
        # reward = np.sqrt(speed)
        # print "Reward: {}".format(reward)
        # return speed
        # reward = 0.0
        # for loop in self.exitloops:
        #    reward += traci.inductionloop.getLastStepVehicleNumber(loop)
        return max(reward, 0)

    def step(self, action):
        #action = self.action_space(action)
        self.start_sumo()
        self.sumo_step += 1
        assert (len(action) == len(self.lights))
        for act, light in zip(action, self.lights):
            signal = light.act(act)
            traci.trafficlights.setRedYellowGreenState(light.id, signal)
        traci.simulationStep()
        observation = self._observation()
        reward = self._reward()
        done = self.sumo_step > self.simulation_end
        self.screenshot()
        if done:
            self.stop_sumo()
        return observation, reward, done, self.route_info

    def screenshot(self):
        if self.mode == "gui":
            traci.gui.screenshot("View #0", self.pngfile)

    def _observation(self):
        obs = []
        for loop in self.loops:
            res = traci.inductionloop.getSubscriptionResults(loop)
            for var in self.loop_variables:
                obs.append(res[var])
        trafficobs = np.array(obs)
        lightobs = [light.state for light in self.lights]
        return (obs, lightobs)

    def reset(self):
        self.stop_sumo()
        # sleep required on some systems
        if self.sleep_between_restart > 0:
            time.sleep(self.sleep_between_restart)
        self.start_sumo()
        observation = self._observation()
        return observation

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if self.mode == "gui":
            img = imread(self.pngfile, mode="RGB")
            if mode == 'rgb_array':
                return img
            elif mode == 'human':
                from gym.envs.classic_control import rendering
                if self.viewer is None:
                    self.viewer = rendering.SimpleImageViewer()
                self.viewer.imshow(img)
        else:
            raise NotImplementedError("Only rendering in GUI mode is supported. Please use Traffic-...-gui-v0.")

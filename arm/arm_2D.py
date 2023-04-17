import gym
import math 
import random
import pygame
import numpy as np
import time
from gym import utils
from gym import error, spaces
from gym.utils import seeding
from scipy.spatial.distance import euclidean

class Arm_2D(gym.Env):

    metadata = {'render.modes': ['human']}


    def __init__(self):
            self.set_window_size([600,600])
            self.set_link_properties([100,100])
            self.set_increment_rate(0.1)                    # +/- joint angle (0.1 rad) 
            self.target_pos = self.generate_random_pos()    # random target
            self.PnP_action = 0
            self.action = {0: "INC_J1",
                           1: "DEC_J1",
                           2: "INC_J2",
                           3: "DEC_J2"}
            
            self.theta_last = [-1000, -1000]                # record theta angle

            # state number:7, action number: 4
            # observation: [joint_1_theta, joint_2_theta, target_tip_dis_x, target_tip_dis_y, goal_tip_dis_x, goal_tip_dis_y, PnP_action]
            # self.observation_space = spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(7,), dtype=np.float32)
            self.observation_space = spaces.Box(np.float32(np.finfo(np.float32).min), np.float32(np.finfo(np.float32).max), shape=(7,))
            self.action_space = spaces.Discrete(len(self.action))

            self.current_error = -math.inf
            self.seed()
            self.viewer = None


    def set_window_size(self, window_size):
        self.window_size = window_size
        self.centre_window = [window_size[0]//2, window_size[1]//2]


    def set_increment_rate(self, rate):
        self.rate = rate


    def set_link_properties(self, links):
        self.links = links
        self.n_links = len(self.links)
        self.min_theta_1 = math.radians(0)              # convert degree to radian 
        self.max_theta_1 = math.radians(180)
        self.min_theta_2 = math.radians(-180)           # convert degree to radian 
        self.max_theta_2 = math.radians(180)
        # self.theta = self.generate_random_angle()       # random initial pose of the arm
        self.theta = self.arm_initial_angle()           # initial arm position 
        self.max_length = sum(self.links)


    def generate_random_angle(self):
        theta = np.zeros(self.n_links)
        theta[0] = random.uniform(self.min_theta_1, self.max_theta_1)   # random angle from (0-180)<->(0-pi)
        theta[1] = random.uniform(self.min_theta_2, self.max_theta_2)   # random angle from (-180-180)
        return theta
    

    def arm_initial_angle(self):
        theta = np.zeros(self.n_links)
        theta[0] = 1.57   # initial arm angle
        theta[1] = 0
        return theta

    # random position of the target 
    def generate_random_pos(self):
        theta = self.generate_random_angle()        # random angle that the arm can reach
        P = self.forward_kinematics(theta)          # calculate the position from angle
        pos = np.array([P[-1][0,3], P[-1][1,3]])    # get the x, y in the matrix
        while not (-100 < pos[0] < 100 and 100 < pos[1] < 170): # while pos is not within the assembly line area
            new_seed = self.seed()[0]                           # generate a new seed
            self.seed(new_seed)                                 # set the new seed for the env
            theta = self.generate_random_angle()                # random angle that the arm can reach
            P = self.forward_kinematics(theta)                  # calculate the position from angle
            pos = np.array([P[-1][0,3], P[-1][1,3]])            # get the x,y in the matrix
        return pos
    

    # forward kinematics calculation: return 
    def forward_kinematics(self, theta):
        P = []
        P.append(np.eye(4))
        for i in range(0, self.n_links):
            R = self.rotate_z(theta[i])
            T = self.translate(self.links[i], 0, 0)
            P.append(P[-1].dot(R).dot(T))
        return P


    def rotate_z(self, theta):
        rz = np.array([[np.cos(theta), - np.sin(theta), 0, 0],
                        [np.sin(theta), np.cos(theta), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        return rz


    def translate(self, dx, dy, dz):
        t = np.array([[1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])
        return t


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

    def inverse_theta(self, theta):
        new_theta = theta.copy()
        for i in range(theta.shape[0]):
            new_theta[i] = -1*theta[i]
        return new_theta


    def draw_arm(self, theta):
        LINK_COLOR = (0, 255, 255)
        JOINT_COLOR = (0, 0, 0)
        TIP_COLOR = (0, 0, 255)
        theta = self.inverse_theta(theta)
        P = self.forward_kinematics(theta)
        origin = np.eye(4)
        origin_to_base = self.translate(self.centre_window[0],self.centre_window[1],0)
        base = origin.dot(origin_to_base)
        F_prev = base.copy()
        for i in range(1, len(P)):
            F_next = base.dot(P[i])
            pygame.draw.line(self.screen, LINK_COLOR, (int(F_prev[0,3]), int(F_prev[1,3])), (int(F_next[0,3]), int(F_next[1,3])), 5)
            pygame.draw.circle(self.screen, JOINT_COLOR, (int(F_prev[0,3]), int(F_prev[1,3])), 10)
            F_prev = F_next.copy()
        pygame.draw.circle(self.screen, TIP_COLOR, (int(F_next[0,3]), int(F_next[1,3])), 8)


    def draw_target(self):
        TARGET_COLOR = (255,0,0)
        origin = np.eye(4)
        origin_to_base = self.translate(self.centre_window[0], self.centre_window[1], 0)
        base = origin.dot(origin_to_base)
        base_to_target = self.translate(self.target_pos[0], -self.target_pos[1], 0)
        target = base.dot(base_to_target)
        pygame.draw.circle(self.screen, TARGET_COLOR, (int(target[0,3]),int(target[1,3])), 20)


    def draw_target_on_tip(self):
        TARGET_COLOR = (255,0,0)
        pygame.draw.circle(self.screen, TARGET_COLOR, (int(self.tip_pos[0])+300, 300-int(self.tip_pos[1])), 20)


    def draw_assembly_line(self):
        al_color = (255, 255, 255)
        al_width = 300
        al_height = 100
        al_x = 150
        al_y = 100
        pygame.draw.rect(self.screen, al_color, (al_x, al_y, al_width, al_height))

    
    def draw_goal(self):
        goal_color = (0, 0, 0)
        goal_width = 100
        goal_height = 100
        goal_x = 450
        goal_y = 250
        pygame.draw.rect(self.screen, goal_color, (goal_x, goal_y, goal_width, goal_height))

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))


    def step(self, action):
        if self.action[action] == "INC_J1":
            self.theta[0] += self.rate
            # print("Action: ","J1+")
        elif self.action[action] == "DEC_J1":
            self.theta[0] -= self.rate
            # print("Action: ","J1-")
        elif self.action[action] == "INC_J2":
            self.theta[1] += self.rate 
            # print("Action: ","J2+")
        elif self.action[action] == "DEC_J2":
            self.theta[1] -= self.rate
            # print("Action: ","J2-")

        self.theta[0] = np.clip(self.theta[0], self.min_theta_1, self.max_theta_1)
        self.theta[1] = np.clip(self.theta[1], self.min_theta_2, self.max_theta_2)
        self.theta[0] = self.normalize_angle(self.theta[0])
        self.theta[1] = self.normalize_angle(self.theta[1])

        # Calc reward
        P = self.forward_kinematics(self.theta)
        self.tip_pos = [P[-1][0,3], P[-1][1,3]]
        
        target_tip_dis_x = self.target_pos[0] - self.tip_pos[0]
        target_tip_dis_y = self.target_pos[1] - self.tip_pos[1]

        goal_tip_dis_x = 200 - self.tip_pos[0]
        goal_tip_dis_y = 0 - self.tip_pos[1]

        dis_err_target = euclidean(self.target_pos, self.tip_pos)
        dis_err_goal = euclidean([200, 0], self.tip_pos)

        # The arm is looking for the target, PnP_action is 0 now
        if self.PnP_action == 0:
            goal_tip_dis_x = 0      # not return distance between tip and goal if not picking
            goal_tip_dis_y = 0

            reward = 0      # If approaching the target, give a small reward

            if dis_err_target >= self.current_error:    # If far from the target, give a penalty
                reward = -0.2

            close_enough_tar = 20
            if (dis_err_target > -close_enough_tar and dis_err_target < close_enough_tar):      # If reach to target, give large reward, set PnP_action to 1
                print("Pick target!!!")
                reward = 10
                self.PnP_action = 1
                time.sleep(0.1)

            if (abs(self.theta[0] - self.theta_last[0]) < 0.001 and abs(self.theta[1] - self.theta_last[1]) < 0.001):
                # print('stay!!!!!!!!!!!!!!!!')
                reward = -1

            self.current_error = dis_err_target         # Update current error

         # The arm is looking for the goal, PnP_action is 1 now
        elif self.PnP_action == 1:
            target_tip_dis_x = 0    # not return distance between tip and target if picking
            target_tip_dis_y = 0

            reward = 0     # If approaching the goal, give a small reward

            if dis_err_goal >= self.current_error:      # If far from the goal, give a penalty
                reward = -0.2

            close_enough_goal = 50 
            if (dis_err_goal > -close_enough_goal and dis_err_goal < close_enough_goal):        # If reach to goal, give large reward, set done flag
                print("Place target!!!")
                reward = 10
                done = True
                time.sleep(0.1)

            if (abs(self.theta[0] - self.theta_last[0]) < 0.001 and abs(self.theta[1] - self.theta_last[1]) < 0.001):
                # print('stay!!!!!!!!!!!!!!!!')
                reward = -1

            self.current_error = dis_err_goal           # Update current error

        self.theta_last[0] = self.theta[0]
        self.theta_last[1] = self.theta[1]
        
        self.current_score += reward    # Accumulative reward

        # print("Current score : ", self.current_score)

        if self.current_score <= -20 or self.current_score >= 20:     # Finish an epoch if accumulative reward is too small
            done = True
        else:
            done = False

        observation = np.hstack((self.theta, target_tip_dis_x, target_tip_dis_y, goal_tip_dis_x, goal_tip_dis_y, self.PnP_action))

        info = {
            'dis_err_target': dis_err_target,
            'dis_err_goal': dis_err_goal,
            'target_position': self.target_pos,
            'current_position': self.tip_pos,
            'theta': self.theta,
            'target_tip_dis_x': target_tip_dis_x,
            'target_tip_dis_y': target_tip_dis_y,
            'goal_tip_dis_x': goal_tip_dis_x,
            'goal_tip_dis_y': goal_tip_dis_y,
            'PnP_action': self.PnP_action
        }
        return observation, reward, done, info


    def reset(self):
        self.theta = self.arm_initial_angle()
        self.theta_last = [-1000, -1000]
        self.target_pos = self.generate_random_pos()
        self.current_score = 0
        P = self.forward_kinematics(self.theta)
        self.tip_pos = [P[-1][0,3], P[-1][1,3]]

        target_tip_dis_x = self.target_pos[0] - self.tip_pos[0]
        target_tip_dis_y = self.target_pos[1] - self.tip_pos[1]

        goal_tip_dis_x = 200 - self.tip_pos[0]
        goal_tip_dis_y = 0 - self.tip_pos[1]

        self.PnP_action = 0

        observation = np.hstack((self.theta, target_tip_dis_x, target_tip_dis_y, goal_tip_dis_x, goal_tip_dis_y, self.PnP_action))
        return observation


    def render(self, mode='human'):
        SCREEN_COLOR = (50, 168, 52)
        if self.viewer == None:
            pygame.init()
            pygame.display.set_caption("RobotArm-Env")
            self.screen = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
            self.viewer = 1
        self.screen.fill(SCREEN_COLOR)
        self.draw_assembly_line()
        self.draw_goal()

        if self.PnP_action == 0:
            self.draw_target()
        elif self.PnP_action == 1:
            self.draw_target_on_tip()

        self.draw_arm(self.theta)
        self.clock.tick(60)
        pygame.display.flip()


    def close(self):
        if self.viewer != None:
            self.viewer = None
            pygame.quit()

         

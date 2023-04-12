import gym
import math 
import random
import pygame
import numpy as np
from gym import utils
from gym import error, spaces
from gym.utils import seeding
from scipy.spatial.distance import euclidean

class Arm_2D(gym.Env):

    metadata = {'render.modes': ['human']}


    def __init__(self):
            self.set_window_size([600,600])
            self.set_link_properties([100,100])
            self.set_increment_rate(0.1)
            self.target_pos = self.generate_random_pos()
            self.PnP_action = 0
            self.action = {0: "PICK",
                           1: "PLACE",
                           2: "INC_J1",
                           3: "DEC_J1",
                           4: "INC_J2",
                           5: "DEC_J2"}
            
            # observation: [target_pos_x, target_pos_y, joint_1_theta, joint_2_theta, target_tip_dis_x, target_tip_dis_y, goal_tip_dis_x, goal_tip_dis_y, PnP_action]
            self.observation_space = spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(9,), dtype=np.float32)
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
        self.min_theta = math.radians(0)        # convert degree to radian 
        self.max_theta = math.radians(180)
        self.theta = self.generate_random_angle()   # random initial pose of the arm 
        self.max_length = sum(self.links)


    def generate_random_angle(self):
        theta = np.zeros(self.n_links)
        theta[0] = random.uniform(self.min_theta, self.max_theta)   # random angle from (0-180)<->(0-1.57)
        theta[1] = random.uniform(self.min_theta, self.max_theta)
        return theta
    

    # random position of the target 
    def generate_random_pos(self):
        theta = self.generate_random_angle()        # random angle that the arm can reach
        P = self.forward_kinematics(theta)          # calculate the position from angle
        pos = np.array([P[-1][0,3], P[-1][1,3]])    # get the x, y in the matrix
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
        LINK_COLOR = (255, 255, 255)
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
        pygame.draw.circle(self.screen, TARGET_COLOR, (int(target[0,3]),int(target[1,3])), 12)


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
        goal_x = 400
        goal_y = 250
        pygame.draw.rect(self.screen, goal_color, (goal_x, goal_y, goal_width, goal_height))

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))


    def step(self, action):
        if self.action[action] == "INC_J1":
            self.theta[0] += self.rate
            print("J1+")
        elif self.action[action] == "DEC_J1":
            self.theta[0] -= self.rate
            print("J1-")
        elif self.action[action] == "INC_J2":
            self.theta[1] += self.rate 
            print("J2+")
        elif self.action[action] == "DEC_J2":
            self.theta[1] -= self.rate
            print("J2-")
        elif self.action[action] == "PICK":
            self.PnP_action = 1
        elif self.action[action] == "PLACE":
            self.PnP_action = 0

        self.theta[0] = np.clip(self.theta[0], self.min_theta, self.max_theta)
        self.theta[1] = np.clip(self.theta[1], self.min_theta, self.max_theta)
        self.theta[0] = self.normalize_angle(self.theta[0])
        self.theta[1] = self.normalize_angle(self.theta[1])

        # Calc reward
        P = self.forward_kinematics(self.theta)
        tip_pos = [P[-1][0,3], P[-1][1,3]]
        distance_error = euclidean(self.target_pos, tip_pos)

        reward = 0
        if distance_error >= self.current_error:
            reward = -1
        epsilon = 10
        if (distance_error > -epsilon and distance_error < epsilon):
            reward = 1

        self.current_error = distance_error
        self.current_score += reward

        print(self.current_score)

        if self.current_score == -10 or self.current_score == 10:
            done = True
        else:
            done = False

        target_tip_dis_x = self.target_pos[0] - tip_pos[0]
        target_tip_dis_y = self.target_pos[1] - tip_pos[1]

        goal_tip_dis_x = 150 - tip_pos[0]
        goal_tip_dis_y = 0 - tip_pos[1]

        observation = np.hstack((self.target_pos, self.theta, target_tip_dis_x, target_tip_dis_y, goal_tip_dis_x, goal_tip_dis_y, self.PnP_action))
        info = {
            'distance_error': distance_error,
            'target_position': self.target_pos,
            'current_position': tip_pos,
            'target_tip_dis_x': target_tip_dis_x,
            'target_tip_dis_y': target_tip_dis_y,
            'goal_tip_dis_x': goal_tip_dis_x,
            'goal_tip_dis_y': goal_tip_dis_y,
            'PnP_action': self.PnP_action
        }
        return observation, reward, done, info


    def reset(self):
        self.target_pos = self.generate_random_pos()
        self.current_score = 0
        P = self.forward_kinematics(self.theta)
        tip_pos = [P[-1][0,3], P[-1][1,3]]

        target_tip_dis_x = self.target_pos[0] - tip_pos[0]
        target_tip_dis_y = self.target_pos[1] - tip_pos[1]

        goal_tip_dis_x = 150 - tip_pos[0]
        goal_tip_dis_y = 0 - tip_pos[1]

        observation = np.hstack((self.target_pos, self.theta, target_tip_dis_x, target_tip_dis_y, goal_tip_dis_x, goal_tip_dis_y, self.PnP_action))
        return observation


    def render(self, mode='human'):
        SCREEN_COLOR = (50, 168, 52)
        if self.viewer == None:
            pygame.init()
            pygame.display.set_caption("RobotArm-Env")
            self.screen = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
        self.screen.fill(SCREEN_COLOR)
        self.draw_assembly_line()
        self.draw_goal()
        self.draw_target()
        self.draw_arm(self.theta)
        self.clock.tick(60)
        pygame.display.flip()


    def close(self):
        if self.viewer != None:
            pygame.quit()

         
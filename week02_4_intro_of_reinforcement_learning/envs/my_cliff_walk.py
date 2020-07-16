import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from envs.maze_env import MazeEnv
from envs.maze_view_2d import Maze, MazeView2D
import pygame

class CliffWalkingEnv(MazeEnv):

    def __init__(self):

        self.viewer = None
        self.maze_view = CliffWalkView2D()
        self.maze_size = self.maze_view.maze_size

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2*len(self.maze_size))

        # observation is the x, y coordinate of the grid
        low = np.zeros(len(self.maze_size), dtype=int)
        high =  np.array(self.maze_size, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        self.observation_space = spaces.Box(low, high)

        # initial condition
        self.state = None
        self.steps_beyond_done = None

        # For MDP planning (added by jmlee)
        self.S = self.maze_size[0] * self.maze_size[1] + 1  # 25 + 1 (including absorbing state)
        print(self.maze_size)
        self.A = 4
        self.gamma = 0.95

        # Simulation related variables.
        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self._configure()
    '''
    def ob_to_state(self, ob):
        return int(ob[1] * self.maze_size[0] + ob[0])

    def state_to_ob(self, state):
        return np.array([state % self.maze_size[0], state // self.maze_size[0]], dtype=int)
    '''

    def ob_to_state(self, ob):
        return int(ob[0] * self.maze_size[1] + ob[1])

    def state_to_ob(self, state):
        return np.array([state // self.maze_size[1], state % self.maze_size[1]], dtype=int)


    def step(self, action):
        if np.any([np.array_equal(self.maze_view.robot, each) for each in self.maze_view.cliffs]):
            reward = -100
            done = True
        else:
            if isinstance(action, int) or isinstance(action, np.int64) or isinstance(action, np.int32):
                self.maze_view.move_robot(self.ACTION[action])
            else:
                self.maze_view.move_robot(action)
            reward = -1
            done = np.array_equal(self.maze_view.robot, self.maze_view.goal)

        self.state = self.maze_view.robot

        info = {'state': self.state}

        return self.ob_to_state(self.state), reward, done, info

    def reset(self):
        self.maze_view.reset_robot()
        self.state = np.array((0, 3), dtype=int)
        self.steps_beyond_done = None
        self.done = False
        return self.ob_to_state(self.state)

class CliffWalkView2D(MazeView2D):

    def __init__(self, screen_size=(601, 400)):

        # PyGame configurations
        pygame.init()
        pygame.display.set_caption("Cliff Walk")
        self.clock = pygame.time.Clock()
        self._game_over = False

        self._maze = Maze(maze_cells=np.ones((12, 4), dtype='int64')*15)
        self.maze_size = self._maze.maze_size
        # to show the right and bottom border
        self.screen = pygame.display.set_mode(screen_size)
        self._screen_size = tuple(map(sum, zip(screen_size, (-1, -1))))

        # Set the starting point
        self._entrance = np.array((0, 3), dtype=int)

        # Set the Goal
        self._goal = np.array((11, 3), dtype=int)

        # Set the Cliff
        self._cliffs = [np.array((i, 3), dtype=int) for i in range(1, 11)]

        # Create the Robot
        self._robot = self.entrance

        # Create a background
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.background.fill((255, 255, 255))

        # Create a layer for the maze
        self.maze_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
        self.maze_layer.fill((0, 0, 0, 0,))

        # added by jmlee (policy evaluation)
        self.is_policy_evaluation = False
        self.maze_layer2 = pygame.Surface(self.screen.get_size()).convert_alpha()
        self.maze_layer2.fill((0, 0, 0, 0,))
        pygame.font.init()
        self.font = pygame.font.SysFont(None, int(self.CELL_W * 0.25))
        self.Q = np.zeros((self.maze.MAZE_H * self.maze.MAZE_W + 1, 4))
        self.pi = np.zeros((self.maze.MAZE_H * self.maze.MAZE_W + 1, 4))

        # show the maze
        self._draw_maze()

        # show the portals
        self._draw_portals()

        # show the robot
        self._draw_robot()

        # show the entrance
        self._draw_entrance()

        # show the goal
        self._draw_goal()

    def reset_robot(self):

        self._draw_robot(transparency=0)
        self._robot = np.array((0, 3), dtype=int)
        self._draw_robot(transparency=255)

    def _draw_cliffs(self):
        colour = (0, 0, 0)
        for location in self.cliffs:
            if self.is_policy_evaluation:
                self._colour_cell(location, colour=colour, transparency=30)
            else:
                self._colour_cell(location, colour=colour, transparency=235)

    def _view_update(self, mode="human"):
        if not self._game_over:
            # update the robot's position
            self._draw_cliffs()
            self._draw_entrance()
            self._draw_goal()
            self._draw_portals()
            self._draw_robot()


            # update the screen
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.maze_layer, (0, 0))
            self.screen.blit(self.maze_layer2, (0, 0))
            if self.is_policy_evaluation:
                # Draw lines
                for x in range(self.maze.MAZE_W):
                    for y in range(self.maze.MAZE_H):
                        color = (200, 200, 200, 255)
                        CELL_W, CELL_H = self.CELL_W, self.CELL_H
                        pygame.draw.line(self.maze_layer2, color, (x * CELL_W, y * CELL_H), ((x + 1) * CELL_W, (y + 1) * CELL_H))
                        pygame.draw.line(self.maze_layer2, color, ((x + 1) * CELL_W, y * CELL_H), (x * CELL_W, (y + 1) * CELL_H))

                # Draw Q texts and policy
                for i in range(self.maze.MAZE_W * self.maze.MAZE_H):
                    y = i % self.maze.MAZE_H
                    x = i // self.maze.MAZE_H
                        
                    for j in range(4):
                        Q_str = "-100" if (y == 3 and x in [1, 2, 3, 4]) else "%.2f" % self.Q[i, j]
                        # N, S, E, W
                        center_x, center_y = (self.CELL_W * x + self.CELL_W * 0.5), (self.CELL_H * y + self.CELL_H * 0.5)
                        triangle_size = 10
                        if j == 0:
                            text_x, text_y = (self.CELL_W * x + self.CELL_W * 0.5), (self.CELL_H * y + self.CELL_H * 0.25)
                            if self.pi[i, j] > 0:
                                arrow_x, arrow_y = center_x, center_y + triangle_size - (self.CELL_H * 0.5) * self.pi[i, j]
                                pygame.draw.line(self.maze_layer2, (0, 0, 0, 128), (center_x, center_y), (arrow_x, arrow_y), 3)
                                pygame.draw.polygon(self.maze_layer2, (0, 0, 0, 128), (
                                    (arrow_x - triangle_size * 0.5, arrow_y),
                                    (arrow_x + triangle_size * 0.5, arrow_y),
                                    (arrow_x, arrow_y - triangle_size),
                                ))
                        elif j == 1:
                            text_x, text_y = (self.CELL_W * x + self.CELL_W * 0.5), (self.CELL_H * y + self.CELL_H * 0.75)
                            if self.pi[i, j] > 0:
                                arrow_x, arrow_y = center_x, center_y - triangle_size + (self.CELL_H * 0.5) * self.pi[i, j]
                                pygame.draw.line(self.maze_layer2, (0, 0, 0, 128), (center_x, center_y), (arrow_x, arrow_y),3)
                                pygame.draw.polygon(self.maze_layer2, (0, 0, 0, 128), (
                                    (arrow_x + triangle_size * 0.5, arrow_y),
                                    (arrow_x - triangle_size * 0.5, arrow_y),
                                    (arrow_x, arrow_y + triangle_size),
                                ))
                        elif j == 2:
                            text_x, text_y = (self.CELL_W * x + self.CELL_W * 0.75), (self.CELL_H * y + self.CELL_H * 0.5)
                            if self.pi[i, j] > 0:
                                arrow_x, arrow_y = center_x - triangle_size + (self.CELL_H * 0.5) * self.pi[i, j], center_y
                                pygame.draw.line(self.maze_layer2, (0, 0, 0, 128), (center_x, center_y), (arrow_x, arrow_y), 3)
                                pygame.draw.polygon(self.maze_layer2, (0, 0, 0, 128), (
                                    (arrow_x, arrow_y + triangle_size * 0.5),
                                    (arrow_x, arrow_y - triangle_size * 0.5),
                                    (arrow_x + triangle_size, arrow_y),
                                ))
                        else:
                            text_x, text_y = (self.CELL_W * x + self.CELL_W * 0.25), (self.CELL_H * y + self.CELL_H * 0.5)
                            if self.pi[i, j] > 0:
                                arrow_x, arrow_y = center_x + triangle_size - (self.CELL_H * 0.5) * self.pi[i, j], center_y
                                pygame.draw.line(self.maze_layer2, (0, 0, 0, 128), (center_x, center_y), (arrow_x, arrow_y), 3)
                                pygame.draw.polygon(self.maze_layer2, (0, 0, 0, 128), (
                                    (arrow_x, arrow_y - triangle_size * 0.5),
                                    (arrow_x, arrow_y + triangle_size * 0.5),
                                    (arrow_x - triangle_size, arrow_y),
                                ))
                        if (y == 3 and x in [1, 2, 3, 4]):
                            surface = self.font.render(Q_str, True, (160, 0, 0, 128))
                        else:
                            if self.Q[i, j] >= np.max(self.Q[i, :]) - 1e-6:
                                surface = self.font.render(Q_str, True, (0, 160, 0, 128))
                            else:
                                surface = self.font.render(Q_str, True, (160, 0, 0, 128))
                        text_rect = surface.get_rect(center=(text_x, text_y))
                        self.screen.blit(surface, text_rect)

            if mode == "human":
                pygame.display.flip()

            return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))



    @property
    def cliffs(self):
        return self._cliffs
import math
import time
from turtledemo.nim import SCREENHEIGHT, SCREENWIDTH

import pygame as pg
from pygame._sdl2.video import Window
import numpy as np
import json

pg.init()
global width, height
width = 800
height = 600
screen = pg.display.set_mode((width, height))
window = Window.from_display_module()
clock = pg.time.Clock()
running = True


class fluidsim():
    def __init__(self,start_dispersion,n,repulsion_speed,repulsion_strength,wall_repulsion_strength,partition_size,density_grid_size,fall_off,gravity,dampening,momentum_sway, air_density):
        self.repulsion = repulsion_speed*10
        self.frames = 0
        self.dampening = dampening
        self.partition_size = partition_size
        self.density_grid_size = density_grid_size
        self.fall_off = fall_off
        self.wall_repulsion_strength = wall_repulsion_strength
        self.repulsion_strength = repulsion_strength
        self.partition_grid_x = math.ceil(width/partition_size)
        self.partition_grid_y = math.ceil(height/partition_size)
        self.density_grid_x = math.ceil(width/density_grid_size)
        self.density_grid_y = math.ceil(height/density_grid_size)
        self.gravity_array = np.full((self.density_grid_y + 1, self.density_grid_x + 1), gravity/10)
        self.last_window_position = window.position
        self.barrier_grid = np.full((self.density_grid_y + 1, self.density_grid_x + 1), 0)
        self.momentum_sway = momentum_sway/10
        self.air_density = air_density

        for x in range(0, self.density_grid_x + 1):
            for y in range(0, self.density_grid_y + 1):
                self.barrier_grid[y][x] = 1 if x <= 1 or x >= self.density_grid_x - 1 or y <= 1 or y >= self.density_grid_y-1 else 0

        max_per_row = (width-200)/start_dispersion
        if n!=0:
            xcoords, ycoords = np.meshgrid(np.array(range(100, int(100+max_per_row*start_dispersion), start_dispersion)), np.array(range(100, int(100+(n//max_per_row)*start_dispersion), start_dispersion)))
            temp_array = np.zeros((len(xcoords), len(xcoords[0]),4))

            for y in range(len(xcoords)):
                for x in range(len(xcoords[0])):
                    temp_array[y][x] = np.array([xcoords[y][x], ycoords[y][x],0,0])
        else:
            temp_array = []
        self.partitioned_list = [[[] for _ in range(self.partition_grid_x)]for _ in range(self.partition_grid_y)]

        for row_n in range(len(self.partitioned_list)):
            for columb_n in range(len(self.partitioned_list[0])):
                    for row in temp_array:
                        for point in row:
                            if point[0]>=columb_n*self.partition_size and point[0]<(columb_n+1)*self.partition_size:
                                if point[1]>=row_n*self.partition_size and point[1]<(row_n+1)*self.partition_size:
                                    self.partitioned_list[row_n][columb_n].append(point)

    def density(self):
        density_array = np.zeros((self.density_grid_y +1, self.density_grid_x + 1))
        for y in range(self.density_grid_y + 1):
            for x in range(self.density_grid_x + 1):
                point = (x*self.density_grid_size, y*self.density_grid_size)
                y_partition_index = round(point[1]/self.partition_size)
                x_partition_index = round(point[0]/self.partition_size)
                ymin,ymax = max(0, y_partition_index-2),min(self.partition_grid_y, y_partition_index+1)
                xmin,xmax = max(0, x_partition_index-2),min(self.partition_grid_x, x_partition_index+1)

                if ymin != ymax:
                    list_y_range = self.partitioned_list[ymin:ymax]
                else:
                    list_y_range = self.partitioned_list[ymin]
                if xmin != xmax:
                    list_range = [row[xmin:xmax] for row in list_y_range]
                else:
                    list_range = list_y_range[::1][xmin]


                for row in list_range:
                    for box in row:
                        for particle in box:
                            to_add = self.repulsion_strength*2**(-((self.fall_off*(point[0]-particle[0]))**2 + (self.fall_off*(point[1]-particle[1]))**2))
                            if to_add >= 0.005:
                                density_array[y][x]+= to_add

                if density_array[y][x]>1:
                    density_array[y][x]= (density_array[y][x]-1)**0.9 +1

                density_array[self.barrier_grid==1] = self.wall_repulsion_strength
        density_array[density_array<0.15] = self.air_density
        self.density_array = density_array


    def velocity(self):
        self.velocity_array = np.gradient(self.density_array)
        self.velocity_array = (self.velocity_array[0] + self.gravity_array, self.velocity_array[1])



    def move(self):
        new_window_position = window.position
        movement_from_window = (self.last_window_position[0]-new_window_position[0], self.last_window_position[1]-new_window_position[1])
        self.last_window_position = new_window_position

        partitioned_list = [[[] for columb in range(self.partition_grid_x)]for row in range(self.partition_grid_y)]
        for row in self.partitioned_list:
            for columb in row:
                for point in columb:
                    if point[0]>width or point[1]>height:
                        continue
                    positionx_on_gradient = point[0]/self.density_grid_size
                    x_ratio = positionx_on_gradient%1
                    positiony_on_gradient = point[1]/self.density_grid_size
                    y_ratio = positiony_on_gradient%1
                    xf,xc = math.floor(positionx_on_gradient),math.ceil(positionx_on_gradient)
                    yf,yc = math.floor(positiony_on_gradient),math.ceil(positiony_on_gradient)
                    y_acceleration = self.repulsion*(self.velocity_array[0][yf][xf]*(1-x_ratio)*(1-y_ratio) + self.velocity_array[0][yf][xc]*(x_ratio)*(1-y_ratio) + self.velocity_array[0][yc][xf]*(1-x_ratio)*(y_ratio) + self.velocity_array[0][yc][xc]*(x_ratio)*(y_ratio))
                    x_acceleration = self.repulsion*(self.velocity_array[1][yf][xf]*(1-x_ratio)*(1-y_ratio) + self.velocity_array[1][yf][xc]*(x_ratio)*(1-y_ratio) + self.velocity_array[1][yc][xf]*(1-x_ratio)*(y_ratio) + self.velocity_array[1][yc][xc]*(x_ratio)*(y_ratio))
                    point[3] = (point[3] * self.dampening - y_acceleration) if y_acceleration<self.wall_repulsion_strength else point[3]*-0.4
                    point[2] = point[2] * self.dampening - x_acceleration

                    point[3] += movement_from_window[1]*self.momentum_sway
                    point[2] += movement_from_window[0]*self.momentum_sway

                    point[1] += point[3]
                    point[0] += point[2]
                    if point[1]<height and point[1] > 0 and point[0] > 0 and point[0] < width:
                        partitioned_list[int(point[1] // self.partition_size)][int(point[0] // self.partition_size)].append(
                            point)
        self.partitioned_list = partitioned_list



    def draw(self):
        for row in self.partitioned_list:
            for columb in row:
                for point in columb:
                    rect = pg.rect.Rect(point[0], point[1], 2, 2)
                    pg.draw.rect(screen, (200, 200, 200), rect)

    def drawDensity(self):
        for y in range(len(self.density_array)):
            for x in range(len(self.density_array[0])):
                rect = pg.rect.Rect(x*self.density_grid_size - self.density_grid_size/2, y*self.density_grid_size -self.density_grid_size/2 , self.density_grid_size, self.density_grid_size)
                pg.draw.rect(screen, (0,0,min(self.density_array[y][x],1)*255), rect)



    def drawVelocity(self,coord):
        maxvel = max(self.velocity_array[coord].reshape(-1))
        for y in range(len(self.velocity_array[0])):
            for x in range(len(self.velocity_array[0][0])):
                rect = pg.rect.Rect(x*self.partition_size - 5, y*self.partition_size -5 , 10, 10)
                if self.velocity_array[coord][y][x] > 0:
                    pg.draw.rect(screen, (abs(self.velocity_array[coord][y][x]*255/maxvel), 0, 0), rect)
                else:
                    pg.draw.rect(screen, (0,0,abs(self.velocity_array[coord][y][x] * 255 / maxvel)), rect)

    def add_particles(self,x,y,xvelocity,yvelocity):
        partitionx,partitiony = x/self.partition_size, y/self.partition_size
        self.partitioned_list[int(partitiony)][int(partitionx)].append([x,y,xvelocity,yvelocity])

    def inflow(self,x,y,xvelocity,yvelocity,frames_per_paritcle,max_particles):
        if frames_per_paritcle*max_particles < self.frames:
            return
        if self.frames%frames_per_paritcle == 0:
            self.add_particles(x,y,xvelocity,yvelocity)
        self.frames+=1

    def draw_Barrier(self):
        for y in range(len(self.barrier_grid)):
            for x in range(len(self.barrier_grid[0])):
                rect = pg.rect.Rect(x * self.density_grid_size - self.density_grid_size / 2,
                                    y * self.density_grid_size - self.density_grid_size / 2, self.density_grid_size,
                                    self.density_grid_size)
                pg.draw.rect(screen, (0, 0, min(self.barrier_grid[y][x], 1) * 255), rect)

    def read_barrier(self):
        with open('LayoutMap.txt', 'r') as filehandle:
            self.barrier_grid = np.array(json.load(filehandle))

    def write_barrier(self):
        with open('LayoutMap.txt', 'w') as filehandle:
            json.dump(self.barrier_grid.tolist(), filehandle)

    def edit_barrier(self):
        while not(pg.key.get_pressed()[pg.K_RETURN]):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
            screen.fill((0,200,0))
            self.density()
            self.draw_Barrier()
            if pg.mouse.get_pressed()[0]:
                x,y = pg.mouse.get_pos()
                x,y = round(x/self.density_grid_size), round(y/self.density_grid_size)
                self.barrier_grid[y][x] = 1 if not(pg.key.get_pressed()[pg.K_SPACE]) else 0
            pg.display.flip()
            clock.tick(60)






fluidsim = fluidsim(40,300,2,0.3,2,30,8, 0.07,-0.55,0.95,0.3, 0.0);


#fluidsim.read_barrier()

fluidsim.edit_barrier()
fluidsim.write_barrier()

while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False


    screen.fill((0,0,0))

    mark0 = time.perf_counter()
    fluidsim.inflow(20,40,1,0,30,300)
    mark1 = time.perf_counter()
    fluidsim.density()
    mark2 = time.perf_counter()
    fluidsim.velocity()
    mark3 = time.perf_counter()
    fluidsim.move()
    mark4 = time.perf_counter()
    #fluidsim.drawVelocity(1)
    #fluidsim.drawVelocity(0)
    fluidsim.drawDensity()
    mark5 = time.perf_counter()
    fluidsim.draw()
    mark6 = time.perf_counter()

    print(f"inflow: {(mark1 - mark0)/(mark6-mark0)*100:.2f}%")
    print(f"density: {(mark2 - mark1) / (mark6 - mark0) * 100:.2f}%")
    print(f"vel: {(mark3 - mark2) / (mark6 - mark0) * 100:.2f}%")
    print(f"move: {(mark4 - mark3) / (mark6 - mark0) * 100:.2f}%")
    print(f"draw density: {(mark5 - mark4) / (mark6 - mark0) * 100:.2f}%")
    print(f"draw balls: {(mark6 - mark5) / (mark6 - mark0) * 100:.2f}%")
    print(f"total: {mark6 - mark0}")
    print("\n")
    pg.display.flip()

    clock.tick(60)
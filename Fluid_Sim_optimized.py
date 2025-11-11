import math
import time
from turtledemo.sorting_animate import partition

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
        self.repulsion_strength = repulsion_strength
        self.partition_grid_x = math.ceil(width/partition_size)
        self.partition_grid_y = math.ceil(height/partition_size)
        self.density_grid_x = math.ceil(width/density_grid_size)
        self.density_grid_y = math.ceil(height/density_grid_size)
        self.grid_x = (np.arange(self.density_grid_x + 1) * self.density_grid_size).astype(float)
        self.grid_y = (np.arange(self.density_grid_y + 1) * self.density_grid_size).astype(float)
        self._k_exp = np.log(2.0) * (self.fall_off ** 2)
        self.gravity_array = np.full((self.density_grid_y + 1, self.density_grid_x + 1), gravity/10)
        self.last_window_position = window.position
        self.barrier_grid = np.full((self.density_grid_y + 1, self.density_grid_x + 1), 0)
        self.momentum_sway = momentum_sway/10
        self.air_density = air_density
        self.wall_repulsion_strength = wall_repulsion_strength

        self.density_coordinates_array = np.zeros(((self.density_grid_y + 1)*(self.density_grid_x + 1),2))
        self.density_coordinates_array = (np.array((
            np.arange((self.density_grid_y + 1)*(self.density_grid_x + 1))%(self.density_grid_x+1),
            np.arange((self.density_grid_y + 1)*(self.density_grid_x + 1))//(self.density_grid_x+1)))
                                          .transpose())*density_grid_size

        for x in range(0, self.density_grid_x + 1):
            for y in range(0, self.density_grid_y + 1):
                self.barrier_grid[y][x] = wall_repulsion_strength if x <= 1 or x >= self.density_grid_x - 1 or y <= 1 or y >= self.density_grid_y-1 else 0

        max_per_row = math.ceil((width-200)/start_dispersion)
        xcoords, ycoords = np.arange(n,dtype=float), np.arange(n,dtype=float)
        xcoords = (xcoords%max_per_row)*start_dispersion + 100
        ycoords = (ycoords//max_per_row)*start_dispersion + 100

        self.particle_array = np.array((xcoords, ycoords, np.zeros_like(xcoords), np.zeros_like(xcoords))).T
        self.particle_array = self.particle_array.reshape(-1, self.particle_array.shape[-1])

    def normal_distribution(self,array):
        return(self.repulsion_strength * 2 ** (-np.sum((self.fall_off*array)**2,axis=-1)))


    def density_optimized(self):
        """
        Build density grid by adding each particle's contribution only to nearby grid cells.
        Uses exp(-k * r^2) where k = ln(2) * fall_off**2 for speed and stability.
        """
        # Start with barrier grid as base (copy to float)
        density = self.barrier_grid.astype(np.float64).copy()

        if self.particle_array.size == 0:
            # no particles -> density is just barrier
            self.density_array = density
            return

        repulsion_strength = self.repulsion_strength
        k = self._k_exp
        density_grid_size = self.density_grid_size

        # radius in pixels beyond which contribution is effectively zero:
        # we'll use partition_size as the search radius; convert to grid-cell radius
        cell_radius = int(np.ceil(self.partition_size / density_grid_size))

        # particle positions
        x = self.particle_array[:, 0].astype(float)
        y = self.particle_array[:, 1].astype(float)

        # grid index of particle (floor)
        gx = np.floor(x / density_grid_size).astype(int)
        gy = np.floor(y / density_grid_size).astype(int)

        # clamp indices to valid range
        gx = np.clip(gx, 0, self.density_grid_x)
        gy = np.clip(gy, 0, self.density_grid_y)

        # For each particle, compute influence on the small subgrid around it
        for i in range(x.shape[0]):
            ix = gx[i]
            iy = gy[i]

            ix0 = max(ix - cell_radius, 0)
            ix1 = min(ix + cell_radius, self.density_grid_x)
            iy0 = max(iy - cell_radius, 0)
            iy1 = min(iy + cell_radius, self.density_grid_y)

            if ix1 < ix0 or iy1 < iy0:
                continue

            # grid cell coordinates (1D arrays) relative to the particle
            xs = self.grid_x[ix0:ix1 + 1] - x[i]  # shape (nx,)
            ys = self.grid_y[iy0:iy1 + 1] - y[i]  # shape (ny,)

            # compute r^2 as outer sum: r2[y, x] = ys[y]**2 + xs[x]**2
            r2 = ys[:, None] ** 2 + xs[None, :] ** 2

            # contribution using exp(-k * r2)
            contrib = repulsion_strength * np.exp(-k * r2)
            contrib = 2**(-((r2**0.5)/5))

            # Optionally mask very small contributions to save some adds
            contrib[r2 > 2*(self.partition_size**2)] = 0.0   # optional

            density[iy0:iy1 + 1, ix0:ix1 + 1] += contrib

        self.density_array = density

    def velocity(self):
        self.velocity_array = np.gradient(self.density_array)
        self.velocity_array = (self.velocity_array[0] + self.gravity_array, self.velocity_array[1])

    def move(self):
        new_window_position = window.position
        movement_from_window = (
            self.last_window_position[0] - new_window_position[0],
            self.last_window_position[1] - new_window_position[1],
        )
        self.last_window_position = new_window_position

        particles = self.particle_array
        vx = particles[:, 2]
        vy = particles[:, 3]
        x = particles[:, 0]
        y = particles[:, 1]

        gx = x / self.density_grid_size
        gy = y / self.density_grid_size

        xf = np.int32(np.floor(gx))
        yf = np.int32(np.floor(gy))

        xc = np.clip(xf + 1, 0, self.density_grid_x-1)
        yc = np.clip(yf + 1, 0, self.density_grid_y-1)

        x_ratio = gx - xf
        y_ratio = gy - yf

        vy_grid, vx_grid = self.velocity_array  # gradient order (y,x)
        y_acc = (
                vy_grid[yf, xf] * (1 - x_ratio) * (1 - y_ratio)
                + vy_grid[yf, xc] * (x_ratio) * (1 - y_ratio)
                + vy_grid[yc, xf] * (1 - x_ratio) * (y_ratio)
                + vy_grid[yc, xc] * (x_ratio) * (y_ratio)
        )
        x_acc = (
                vx_grid[yf, xf] * (1 - x_ratio) * (1 - y_ratio)
                + vx_grid[yf, xc] * (x_ratio) * (1 - y_ratio)
                + vx_grid[yc, xf] * (1 - x_ratio) * (y_ratio)
                + vx_grid[yc, xc] * (x_ratio) * (y_ratio)
        )

        speeds = np.sqrt(vx**2 + vy**2)
        mask = speeds < 20
        x_acc *= mask
        y_acc *= mask

        vx = vx * (1-self.dampening) - x_acc
        vy = vy * (1-self.dampening) - y_acc

        vx += movement_from_window[0] * self.momentum_sway
        vy += movement_from_window[1] * self.momentum_sway

        x = x + self.repulsion*vx
        y = y + self.repulsion*vy

        # --- Collision handling with walls ---
        # Convert particle positions to grid indices
        grid_x = np.clip((x / self.density_grid_size).astype(int), 0, self.density_grid_x)
        grid_y = np.clip((y / self.density_grid_size).astype(int), 0, self.density_grid_y)

        # Check barrier grid
        hits = self.barrier_grid[grid_y, grid_x] > 0

        if np.any(hits) and False:
            # Move particles back along velocity direction until outside wall
            # and damp their velocity to simulate inelastic collision
            vx[hits] *= -0.4  # reverse and dampen
            vy[hits] *= -0.4

            # push them slightly outward (based on which side is clear)
            for i in np.where(hits)[0]:
                gx, gy = grid_x[i], grid_y[i]
                if gx > 0 and self.barrier_grid[gy, gx - 1] == 0:
                    x[i] -= 1
                elif gx < self.density_grid_x and self.barrier_grid[gy, gx + 1] == 0:
                    x[i] += 1
                if gy > 0 and self.barrier_grid[gy - 1, gx] == 0:
                    y[i] -= 1
                elif gy < self.density_grid_y and self.barrier_grid[gy + 1, gx] == 0:
                    y[i] += 1

        particles[:, 0] = x
        particles[:, 1] = y
        particles[:, 2] = vx
        particles[:, 3] = vy

        inside = (
                (x >= 0) & (x < width) &
                (y >= 0) & (y < height)
        )


        self.particle_array = particles[inside]



    def draw(self):
        for point in self.particle_array:
            rect = pg.rect.Rect(point[0], point[1], 2, 2)
            pg.draw.rect(screen, (200, 200, 200), rect)

    def drawDensity(self):
        for y in range(len(self.density_array)):
            for x in range(len(self.density_array[0])):
                rect = pg.rect.Rect(x*self.density_grid_size - self.density_grid_size/2, y*self.density_grid_size -self.density_grid_size/2 , self.density_grid_size, self.density_grid_size)
                pg.draw.rect(screen, (0,0,min(self.density_array[y][x],1)**0.5*255), rect)

    def drawVelocity(self,coord):
        maxvel = max(abs(self.velocity_array[coord].reshape(-1)))
        for y in range(len(self.velocity_array[0])):
            for x in range(len(self.velocity_array[0][0])):
                rect = pg.rect.Rect(x*self.density_grid_size - self.density_grid_size/2, y*self.density_grid_size -self.density_grid_size/2 , self.density_grid_size, self.density_grid_size)
                if self.velocity_array[coord][y][x] > 0:
                    pg.draw.rect(screen, (abs(self.velocity_array[coord][y][x]*255/maxvel), 0, 0), rect)
                else:
                    pg.draw.rect(screen, (0,0,abs(self.velocity_array[coord][y][x] * 255 / maxvel)), rect)



    def add_particles(self,x,y,xvelocity,yvelocity):
        self.particle_array = np.append(self.particle_array,np.array([x,y,xvelocity,yvelocity])[np.newaxis],axis = 0)

    def inflow(self,x,y,number_of_inflow,x_spacing,y_spacing,xvelocity,yvelocity,frames_per_paritcle,max_particles):
        if frames_per_paritcle*max_particles/number_of_inflow < self.frames:
            return
        if self.frames%frames_per_paritcle == 0:
            for i in range(number_of_inflow):
                self.add_particles(x+i*x_spacing,y+i*y_spacing,xvelocity,yvelocity)
        self.frames+=1




    def draw_Barrier(self):
        for y in range(len(self.barrier_grid)):
            for x in range(len(self.barrier_grid[0])):
                if self.barrier_grid[y][x] == 0:
                    continue
                rect = pg.rect.Rect(x * self.density_grid_size - self.density_grid_size / 2,
                                    y * self.density_grid_size - self.density_grid_size / 2, self.density_grid_size,
                                    self.density_grid_size)
                pg.draw.rect(screen, (255, 255,255), rect)

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
            screen.fill((0,0,0))
            self.density_optimized()
            self.draw_Barrier()
            if pg.mouse.get_pressed()[0]:
                x,y = pg.mouse.get_pos()
                x,y = round(x/self.density_grid_size), round(y/self.density_grid_size)
                self.barrier_grid[y][x] = self.wall_repulsion_strength if not(pg.key.get_pressed()[pg.K_SPACE]) else 0
            pg.display.flip()
            clock.tick(60)







fluidsim = fluidsim(5,0,1,0.5,3,60,5, 0.2,-0.4,0.06,0.02, 0.0);

fluidsim.read_barrier()
fluidsim.edit_barrier()
fluidsim.write_barrier()


while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False


    screen.fill((0,0,0))

    mark0 = time.perf_counter()
    fluidsim.inflow(20, 40,2,0,50, 1, 0, 5, 4000)

    mark1 = time.perf_counter()
    fluidsim.density_optimized()
    mark2 = time.perf_counter()
    fluidsim.velocity()
    mark3 = time.perf_counter()
    fluidsim.move()
    mark4 = time.perf_counter()
    #fluidsim.drawVelocity(1)
    # fluidsim.drawVelocity(0)
    fluidsim.drawDensity()
    mark5 = time.perf_counter()
    fluidsim.draw()
    fluidsim.draw_Barrier()
    mark6 = time.perf_counter()

    '''
    print(f"inflow: {(mark1 - mark0) / (mark6 - mark0) * 100:.2f}%")
    print(f"density: {(mark2 - mark1) / (mark6 - mark0) * 100:.2f}%")
    print(f"vel: {(mark3 - mark2) / (mark6 - mark0) * 100:.2f}%")
    print(f"move: {(mark4 - mark3) / (mark6 - mark0) * 100:.2f}%")
    print(f"draw density: {(mark5 - mark4) / (mark6 - mark0) * 100:.2f}%")
    print(f"draw balls: {(mark6 - mark5) / (mark6 - mark0) * 100:.2f}%")
    print(f"total: {mark6-mark0}")
    print("\n")
    '''



    pg.display.flip()

    clock.tick(60)
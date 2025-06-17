import random
import numpy as np
import matplotlib.pyplot as plt
mu = 2.8
sigma = 0.5
immunity = 80

class Agent:

    def __init__(self, homex,homey, mask, pmask, pi):
        self.x = homex
        self.y = homey
        self.homex = homex
        self.homey = homey
        self.mask = mask
        self.state = "s"
        self.infected_days = 0
        self.pmask = pmask
        self.pi = pi
        self.recover_time = None
        self.immune_days = 0
        
        
    def move(self, radius, grid_size):
        newx = self.x + np.random.randint(-1, 2)
        newy = self.y + np.random.randint(-1, 2)
        if abs(newx - self.homex) > radius:
            newx = self.homex + np.sign(newx - self.homex) * radius

        if abs(newy - self.homey) > radius:
            newy = self.homey + np.sign(newy - self.homey) * radius
        
        newx = max(0, min(newx, grid_size - 1))
        newy = max(0, min(newy, grid_size - 1))
        
        self.x, self.y = newx, newy
        
    def update(self, agents):
        if self.state == "i":
            self.infected_days +=1
            if self.infected_days >= self.recover_time:
                self.state = "r"
                self.immune_days = 0
        elif self.state == "r":
            self.immune_days += 1
            if self.immune_days >= immunity:
                self.state = "s"
               
        else:
            psafe = 1
            for agent in agents:
                if agent.state == "i":
                
                    if abs(agent.x - self.x) <= 1 and abs(agent.y - self.y) <= 1:
                        if agent.mask:
                            psafe *= 1 - self.pi * self.pmask
                        else:
                            psafe *= 1 - self.pi
            if np.random.random() < 1 - psafe:
                self.state = "i"
                self.infected_days = 0
                self.recover_time = max(1, int(np.random.lognormal(mu, sigma)))
            
            

class Simulation:
    def __init__(self, grid_size, n_agents, initial_infected, 
                 pmask, pi, mask_ratio, radius):

        self.grid_size = grid_size
        self.n_agents = n_agents
        self.initial_infected = initial_infected
        self.pmask = pmask
        self.pi = pi
        self.mask_ratio = mask_ratio
        self.radius = radius
        self.agents = []
        self.time_step = 0
        self.history = []
        self.setup_agents()
        
    def setup_agents(self):
        home_clusters = min(15, self.n_agents // 20)  
        cluster_centers = [(np.random.randint(0, self.grid_size), 
                           np.random.randint(0, self.grid_size)) 
                          for _ in range(home_clusters)]
        for i in range(self.n_agents):
            
            cluster = random.choice(cluster_centers)
            
            homex = np.clip(cluster[0] + np.random.randint(-5, 6), 0, self.grid_size-1)
            homey = np.clip(cluster[1] + np.random.randint(-5, 6), 0, self.grid_size-1)
            
            
            mask = np.random.random() < self.mask_ratio
            
            agent = Agent(homex, homey, mask, self.pmask, self.pi)
            self.agents.append(agent)
            
        for agent in random.sample(self.agents, self.initial_infected):
            agent.state = "i"
            agent.infected_days = 0
            agent.recover_time = max(1, int(np.random.lognormal(mu, sigma)))
            
    def step(self):
       
        for agent in self.agents:
            agent.move(self.radius, self.grid_size)
        
        for agent in self.agents:
            agent.update(self.agents)
        
        stats = {
            'susceptible': sum(1 for x in self.agents if x.state == "s"),
            'infected': sum(1 for x in self.agents if x.state == "i"),
            #'immune': sum(1 for x in self.agents if x.state == "r"),
            'time_step': self.time_step
        }
        self.history.append(stats)
        self.time_step += 1  
        
    def run(self, steps):
        for i in range(steps):
            self.step()        
              
    def plot_results(self):
        time_steps = [record['time_step'] for record in self.history]
        susceptible = [record['susceptible'] for record in self.history]
        infected = [record['infected'] for record in self.history]
        #immune = [record['immune'] for record in self.history]
        
        plt.plot(time_steps, susceptible, label='susceptible', color='blue')
        plt.plot(time_steps, infected, label='infected', color='red')
        #plt.plot(time_steps, immune, label='immune', color='green')
        
        plt.title('covid simulation')
        plt.xlabel('time step')
        plt.ylabel('num agents')
        plt.legend()
        plt.grid(True)
        plt.show()
                    
sim = Simulation(grid_size = 50, n_agents = 1200 , initial_infected = 5, 
                 pmask = 0.8, pi = 0.05, mask_ratio = 0.5, radius = 5)
sim.run(600)
#sim.plot_results()

import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from IPython.display import HTML



def animate_simulation(sim, frame_skip=5):
    # Set larger embed limit
    plt.rcParams['animation.embed_limit'] = 50  # 50MB
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Color map for agent states
    cmap = ListedColormap(['blue', 'red', 'green'])
    
    # Set up the grid plot
    grid = np.zeros((sim.grid_size, sim.grid_size))
    ax1.imshow(grid, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Agent Positions')
    
    # Initialize empty scatter plot
    scat = ax1.scatter([], [], c=[], cmap=cmap, vmin=0, vmax=2, s=20)
    
    # Set up the time series plot
    ax2.set_xlim(0, len(sim.history))
    ax2.set_ylim(0, sim.n_agents)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Number of Agents')
    ax2.set_title('Epidemic Curve')
    ax2.grid(True)
    
    # Initialize lines for the time series
    line_s, = ax2.plot([], [], 'b-', label='Susceptible', lw=1)
    line_i, = ax2.plot([], [], 'r-', label='Infected', lw=1)
    line_r, = ax2.plot([], [], 'g-', label='Recovered', lw=1)
    ax2.legend()
    
    # Store the data for the time series
    time_data = []
    s_data = []
    i_data = []
    r_data = []
    
    # We need to store agent positions at each frame
    # Let's recreate the simulation steps to capture positions
    agent_positions = []
    agent_states = []
    
    # Create a copy of the simulation to replay
    temp_sim = Simulation(grid_size=sim.grid_size, 
                         n_agents=sim.n_agents,
                         initial_infected=sim.initial_infected,
                         pmask=sim.pmask,
                         pi=sim.pi,
                         mask_ratio=sim.mask_ratio,
                         radius=sim.radius)
    
    # Run the simulation and record positions at each step
    for i in range(len(sim.history)):
        # Record current positions and states
        positions = [(agent.x, agent.y) for agent in temp_sim.agents]
        states = [agent.state for agent in temp_sim.agents]
        agent_positions.append(positions)
        agent_states.append(states)
        
        # Step the simulation forward
        if i < len(sim.history) - 1:
            temp_sim.step()
    
    def init():
        scat.set_offsets(np.empty((0, 2)))
        line_s.set_data([], [])
        line_i.set_data([], [])
        line_r.set_data([], [])
        return scat, line_s, line_i, line_r
    
    def update(frame):
        # Calculate the actual frame number (skipping frames)
        actual_frame = frame * frame_skip
        if actual_frame >= len(sim.history):
            actual_frame = len(sim.history) - 1
        
        # Clear and redraw the grid
        ax1.clear()
        ax1.imshow(grid, cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f'Agent Positions (Step {actual_frame})')
        
        # Get current agent positions and states
        positions = agent_positions[actual_frame]
        states = agent_states[actual_frame]
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        color_codes = [0 if s == 's' else 1 if s == 'i' else 2 for s in states]
        
        # Update scatter plot
        scat = ax1.scatter(x_coords, y_coords, c=color_codes, 
                          cmap=cmap, vmin=0, vmax=2, s=20)
        
        # Update time series data
        time_data.append(actual_frame)
        s_data.append(sim.history[actual_frame]['susceptible'])
        i_data.append(sim.history[actual_frame]['infected'])
        r_data.append(sum(1 for s in states if s == 'r'))
        
        line_s.set_data(time_data, s_data)
        line_i.set_data(time_data, i_data)
        line_r.set_data(time_data, r_data)
        
        # Adjust axes limits
        ax2.set_xlim(0, len(sim.history))
        max_y = max(max(s_data), max(i_data), max(r_data))
        ax2.set_ylim(0, max(1, max_y * 1.1))
        
        return scat, line_s, line_i, line_r
    
    # Calculate number of frames to show
    n_frames = len(sim.history) // frame_skip
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=n_frames, 
        init_func=init, 
        blit=False, 
        interval=200,
        repeat=False
    )
    
    plt.tight_layout()
    return ani

# Create optimized animation (showing every 5th frame)
ani = animate_simulation(sim, frame_skip=5)

# To display in Jupyter Notebook
# from IPython.display import HTML
# HTML(ani.to_jshtml())

# To display in regular Python script
plt.show()
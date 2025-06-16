import matplotlib
matplotlib.use('TkAgg')
from pylab import *
import pycxsimulator

n = 50            # number of agents
r = 0.1           # infection radius
p_infect = 0.4    # probability of infection
move_range = 0.01 # movement per step

class agent:
    pass

def initialize():
    global agents
    agents = []
    for i in range(n):
        ag = agent()
        ag.type = 0  # 0 = Susceptible, 1 = Infected
        ag.x = random()
        ag.y = random()
        ag.house_x = ag.x
        ag.house_y = ag.y
        ag.house_radius = 0.05
        agents.append(ag)

    # Infect one random agent
    agents[randint(n)].type = 1

def observe():
    global agents
    cla()
    susceptible = [ag for ag in agents if ag.type == 0]
    infected = [ag for ag in agents if ag.type == 1]
    grid()
    plot([ag.x for ag in susceptible], [ag.y for ag in susceptible], 'bo', label='Susceptible')
    plot([ag.x for ag in infected], [ag.y for ag in infected], 'ro', label='Infected')
    axis('image')
    axis([0, 1, 0, 1])
    legend()

def update():
    global agents

    for ag in agents:
        dx = uniform(-move_range, move_range)
        dy = uniform(-move_range, move_range)

        if ag.type == 0:
            # Susceptible: move freely
            ag.x = max(0, min(1, ag.x + dx))
            ag.y = max(0, min(1, ag.y + dy))
        else:
            # Infected: move only inside house radius
            new_x = ag.x + dx
            new_y = ag.y + dy
            dist_sq = (new_x - ag.house_x)**2 + (new_y - ag.house_y)**2
            if dist_sq <= ag.house_radius**2:
                ag.x = new_x
                ag.y = new_y
            # else: stays in place (blocked by quarantine)

    # Infection spread
    for ag in agents:
        if ag.type == 0:  # Susceptible
            neighbors = [nb for nb in agents
                         if (ag.x - nb.x)**2 + (ag.y - nb.y)**2 < r**2 and nb != ag]
            for nb in neighbors:
                if nb.type == 1 and random() < p_infect:
                    ag.type = 1
                    break

pycxsimulator.GUI().start(func=[initialize, observe, update])


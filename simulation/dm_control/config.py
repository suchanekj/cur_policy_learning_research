from simulation.dm_control.simulation_api import SimulationAPI

HOF_ITERATIONS = 500
HOF_OUTPUT_PATH = 'hof.out'
NUM_STEPS = 50
INPUT_SIZE, = SimulationAPI().get_action_spec().shape
OBJECT_INITIAL_HEIGHT = 0.46163282 # hardcoded because the stuuuupid object keeps bouncing after spawning lol
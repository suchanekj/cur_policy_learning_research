from simulation.dm_control.simulation_api import SimulationAPI

HOF_ITERATIONS = 1000
HOF_POPULATIONS = 100
HOF_OUTPUT_PATH = 'hof2.out'
NUM_STEPS = 50
INPUT_SIZE, = SimulationAPI().get_action_spec().shape

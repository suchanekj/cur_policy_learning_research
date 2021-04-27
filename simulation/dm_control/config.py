from simulation.dm_control.simulation_api import SimulationAPI

HOF_ITERATIONS = 1  # number of matches that the children fight against each other for
HOF_POPULATIONS = 50  # number of children that fight each other every iteration
HOF_COUNT = 50  # number of winners in the final round to keep
HOF_OUTPUT_DIRECTORY = 'hall_of_fame_solutions/'
NUM_STEPS = 5  # number of steps that the robot can take
INPUT_SIZE, = SimulationAPI().get_action_spec().shape
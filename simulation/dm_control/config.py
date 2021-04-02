from simulation_api import SimulationAPI

HOF_ITERATIONS = 10  # number of matches that the children fight against each other for
HOF_POPULATIONS = 10  # number of children that fight each other every iteration
HOF_COUNT = 10  # number of winners in the final round to keep
HOF_OUTPUT_DIRECTORY = 'hall_of_fame_solutions/'
NUM_STEPS = 50  # number of steps that the robot can take
INPUT_SIZE, = SimulationAPI().get_action_spec().shape
import numpy as np

print([0]*4)
print(np.array([[0]*4, [0]*4])[0])

state = np.array([[0]*4, [0]*4])
range_around = 2
setpoint_temp_range = list(range(23-range_around, 23+range_around + 1))

comfort = [2 - abs(setpoint_temp_range[2] - t) if t in setpoint_temp_range else -(abs(min(setpoint_temp_range, key=lambda x: abs(x - t)) - t)) for t in state[0]]
print(comfort)
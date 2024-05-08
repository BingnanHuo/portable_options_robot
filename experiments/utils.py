from portable.utils.ale_utils import get_object_position, get_skull_position

def get_snake_x_left(position):
    if position[2] == 9:
        return 11
    if position[2] == 11:
        return 35
    if position[2] == 22:
        return 25

def get_snake_x_right(position):
    if position[2] == 9:
        return 60
    if position[2] == 11:
        return 118
    if position[2] == 22:
        return 25



def in_epsilon_square(current_position, final_position):
    epsilon = 2
    if current_position[0] <= (final_position[0] + epsilon) and \
        current_position[0] >= (final_position[0] - epsilon) and \
        current_position[1] <= (final_position[1] + epsilon) and \
        current_position[1] >= (final_position[1] - epsilon):
        return True
    return False   

def get_percent_completed_enemy(start_pos, final_pos, terminations, env):
    def manhatten(a,b):
        return sum(abs(val1-val2) for val1, val2 in zip((a[0], a[1]),(b[0],b[1])))

    if start_pos[2] != final_pos[2]:
        return 0

    info = env.get_current_info({})
    if info["dead"]:
        return 0

    room = start_pos[2]
    ground_y = start_pos[1]
    ram = env.unwrapped.ale.getRAM()
    true_distance = 0
    completed_distance = 0
    if final_pos[2] != room:
        return 0
    if final_pos[2] == 4 and final_pos[0] == 5:
        return 0
    if room in [2,3]:
        # skulls
        skull_x = get_skull_position(ram)
        end_pos = (skull_x-25, ground_y)
        if final_pos[0] < skull_x-25 and final_pos[1] <= ground_y:
            return 1
        else:
            true_distance = manhatten(start_pos, end_pos)
            completed_distance = manhatten(start_pos, final_pos)
    if room in [0,1,18]:
        # skulls
        skull_x = get_skull_position(ram)
        end_pos = (skull_x-6, ground_y)
        if final_pos[0] < skull_x-6 and final_pos[1] <= ground_y:
            return 1
        else:
            true_distance = manhatten(start_pos, end_pos)
            completed_distance = manhatten(start_pos, final_pos)
    elif room in [4,13,21]:
        # spiders
        spider_x, _ = get_object_position(ram)
        end_pos = (spider_x - 6, ground_y)
        if final_pos[0] < spider_x and final_pos[1] <= ground_y:
            return 1
        else:
            true_distance = manhatten(start_pos, end_pos)
            completed_distance = manhatten(start_pos, final_pos)
    elif room in [9,11,22]:
        # snakes
        end_pos = terminations
        if in_epsilon_square(final_pos, end_pos):
            return 1
        else:
            true_distance = manhatten(start_pos, end_pos)
            completed_distance = manhatten(start_pos, final_pos)
    else:
        return 0

    return completed_distance/(true_distance+1e-5)

def check_termination_correct_enemy_left(state, env):
    info = env.get_current_info({})
    if info["dead"]:
        return False
    
    position = info["position"]

    room = position[2]
    ram = env.unwrapped.ale.getRAM()
    if room in [2,3]:
        # dancing skulls
        skull_x = get_skull_position(ram)
        if position[0] < skull_x-25 and position[1] <= 235:
            return True
        else:
            return False
    if room in [1,5,18]:
        # rolling skulls
        skull_x = get_skull_position(ram)
        if room == 1:
            ground_y = 148
        elif room == 5:
            ground_y = 195
        elif room == 18:
            ground_y = 235
        if position[0] < skull_x-6 and position[1] <= ground_y:
            return True
        else:
            return False
    elif room in [4,13,21]:
        # spiders
        spider_x, _ = get_object_position(ram)
        ground_y = 235
        if position[0] < spider_x and position[1] <= ground_y:
            return True
        else:
            return False
    elif room in [9,11,22]:
        # snakes
        snake_x = get_snake_x_left(position)
        ground_y = 235
        if position[0] < snake_x and position[1] <= ground_y:
            return True
        else:
            return False
    else:
        return False

def check_termination_correct_enemy_right(state, env):
    info = env.get_current_info({})
    if info["dead"]:
        return False
    
    position = info["position"]
    room = position[2]
    ram = env.unwrapped.ale.getRAM()
    if room in [2,3]:
        # dancing skulls
        skull_x = get_skull_position(ram)
        if position[0] > skull_x+25 and position[1] <= 235:
            return True
        else:
            return False
    if room in [1,5,18]:
        # rolling skulls
        skull_x = get_skull_position(ram)
        if room == 1:
            ground_y = 148
        elif room == 5:
            ground_y = 195
        elif room == 18:
            ground_y = 235
        if position[0] > skull_x+6 and position[1] <= ground_y:
            return True
        else:
            return False
    elif room in [4,13,21]:
        # spiders
        spider_x, _ = get_object_position(ram)
        ground_y = 235
        if position[0] > spider_x and position[1] <= ground_y:
            return True
        else:
            return False
    elif room in [9,11,22]:
        # snakes
        snake_x = get_snake_x_right(position)
        ground_y = 235
        if position[0] > snake_x and position[1] <= ground_y:
            return True
        else:
            return False
    else:
        return False
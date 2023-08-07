# Solution: instead of calculating angles and whatnot,
# we reflect the room in all directions and consider
# the laser blasts to be linear.

def solution(dimensions, your_position, guard_position, distance):
    
    # define some variables to make things readable down below
    your_x, your_y = your_position
    guard_x, guard_y = guard_position
    width, height = dimensions
    x_min = your_x - distance
    x_max = your_x + distance
    y_min = your_y - distance
    y_max = your_y + distance

    # a few nested functions using aforementioned variables
    
    def distance_from_you(x, y):
        from math import sqrt
        return sqrt((x-your_x)**2 + (y-your_y)**2)

    def make_vector(x, y):
        if x == your_x and y == your_y:
            return (0, 0)
        from fractions import gcd
        x -= your_x
        y -= your_y
        d = abs(gcd(x, y))
        return (x // d, y // d)

    # this gets a set of coordinates from a starting point,
    # adding and subtracting alternating increments
    # (they alternate because of the reflection)
    # from a min to a max value
    def get_coords(start, inc1, inc2, min_value, max_value):
        coords = [start]
        this_v = start
        while True:
            this_v -= inc1
            if this_v < min_value:
                break
            coords.append(this_v)
            this_v -= inc2
            if this_v < min_value:
                break
            coords.append(this_v)

        this_v = start
        while True:
            this_v += inc2
            if this_v > max_value:
                break
            coords.append(this_v)
            this_v += inc1
            if this_v > max_value:
                break
            coords.append(this_v)

        return coords

    # getting the YOU coordinates:
    # first we get the increments
    # (aka distance from each wall)
    inc_left = 2 * your_x
    inc_right = 2 * (width - your_x)
    inc_up = 2 * (height - your_y)
    inc_down = 2 * your_y

    # we get all possible Xs and Ys
    your_X = get_coords(your_x, inc_left, inc_right, x_min, x_max)
    your_Y = get_coords(your_y, inc_down, inc_up, y_min, y_max)
    
    # then we calculate the vectors from each position
    # and store in the dict only the least distant results
    vectors = dict()
    for x in your_X:
        for y in your_Y:
            d = distance_from_you(x, y)
            if d <= distance:
                vec = make_vector(x, y)
                if vec not in vectors or d < vectors[vec][0]:
                    vectors[vec] = (d, 'y')
                
    # same thing for the guard
    inc_left = 2 * guard_x
    inc_right = 2 * (width - guard_x)
    inc_up = 2 * (height - guard_y)
    inc_down = 2 * guard_y

    guard_X = get_coords(guard_x, inc_left, inc_right, x_min, x_max)
    guard_Y = get_coords(guard_y, inc_down, inc_up, y_min, y_max)

    for x in guard_X:
        for y in guard_Y:
            d = distance_from_you(x, y)
            if d <= distance:
                vec = make_vector(x, y)
                if vec not in vectors or d < vectors[vec][0]:
                    vectors[vec] = (d, 'g')

    # here, all entries in the dict
    # that have 'g' in the tuple represent
    # clear shots to the guard

    # now we just count 'em
    k = len([value for value in vectors.values() if value[1] == 'g'])
    return k



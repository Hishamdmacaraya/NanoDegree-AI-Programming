def reward_function(params):
    # Example of rewarding the agent to stay inside the two borders of the track
    
    # Read input parameters
    all_wheels_on_track = params['all_wheels_on_track']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    speed = params['speed']
    steering = abs(params['steering_angle']) # Only need the absolute steering angle
    progress = params['progress']
    
    # Give a very low reward by default
    reward = 1e-3
    
    #Reward if the car completes the track
    if progress == 100:
        reward = reward + 2
   
    # Give a high reward if no wheels go off the track and
    # the agent is somewhere in between the track borders
    if all_wheels_on_track and (0.5*track_width - distance_from_center) >= 0.05:
        reward = 1.0

    # Steering penality threshold, change the number based on your action space setting
    ABS_STEERING_THRESHOLD = 15

    # Penalize reward if the agent is steering too much
    if steering > ABS_STEERING_THRESHOLD:
        reward *= 0.8
    
    SPEED_THRESHOLD = 1
    
    if speed < SPEED_THRESHOLD:

        # Penalize if the car goes too slow

        reward = reward + 0.5

    else:
        # High reward if the car stays on track and goes fast
        reward = reward + 1.0
    

    # Always return a float value
    return float(reward)

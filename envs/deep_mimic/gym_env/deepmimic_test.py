import gym
import envs
import time

#env = gym.make('HumanoidDeepMimicBackflipBulletEnv-v1')
env = gym.make('HumanoidDeepMimicWalkBulletEnv-v1')
env.reset()
env.render(mode='human')
print("------------------------------------")
print("env=",env)
print(dir(env))
print(dir(env.unwrapped))
dt = 1./240.
logId = env.unwrapped._internal_env._pybullet_client.startStateLogging(env.unwrapped._internal_env._pybullet_client.STATE_LOGGING_PROFILE_TIMINGS, "perf.json")
for i in range (100):
  env.unwrapped._internal_env._pybullet_client.submitProfileTiming("loop")
  
  env.reset()
  env.unwrapped._internal_env._pybullet_client.submitProfileTiming()
env.unwrapped._internal_env._pybullet_client.stopStateLogging(logId)

action = env.unwrapped.action_space.sample()
while (1):
  time.sleep(dt)
  #keys = env.unwrapped._internal_env._pybullet_client.getKeyboardEvents()
  #if keys:
  #  env.reset()
  #action=[0]*36
  action = env.unwrapped.action_space.sample()
  state, reward, done, truncated, info = env.step(action)
  img = env.render(mode='rgb_array')

  if done:
    env.reset()
    #action = env.unwrapped.action_space.sample()
  #print("reward=",reward)
    
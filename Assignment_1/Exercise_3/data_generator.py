import numpy as np
import random

def generate_losangles(data_size, quadrant):
  points = []
  for _ in range(data_size):
    if quadrant == 0:
      x = random.uniform(0,1)
      y = random.uniform(0,1-x)
      points.append([x,y,quadrant])

    if quadrant == 1:
      x = random.uniform(-1,0)
      y = random.uniform(0,x+1)
      points.append([x,y,quadrant])

    if quadrant == 2:
      x = random.uniform(0,1)
      y = random.uniform(-1+x, 0)
      points.append([x,y,quadrant])

    if quadrant == 3:
      x = random.uniform(-1,0)
      y = random.uniform(-1-x, 0)
      points.append([x,y,quadrant])

  return points

def generate_edges(data_size, quadrant):
  points = []

  for _ in range(data_size):

    if quadrant == 0:
      x = random.uniform(0,1)
      y = random.uniform(1-x, np.sqrt(1 - x**2))
      points.append([x,y,quadrant+4])

    if quadrant == 1:
      x = random.uniform(-1,0)
      y = random.uniform(1+x, np.sqrt(1 - x**2))
      points.append([x,y,quadrant+4])    

    if quadrant == 2:
      x = random.uniform(-1,0)
      y = random.uniform(-np.sqrt(1 - x**2), -1 - x)
      points.append([x,y,quadrant+4])        

    if quadrant == 3:
      x = random.uniform(0,1)
      y = random.uniform(-np.sqrt(1 - x**2), -1 + x)
      points.append([x,y,quadrant+4])    

  return points 
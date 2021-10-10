import cv2
import random
import math
import numpy as np
import torch

def get_cluster_points(num_points_per_cluster, x, y, points, cluster_radius):
	for c in range(num_points_per_cluster // 2):
		x_range = (x - cluster_radius, x + cluster_radius)
		y_range = (y - cluster_radius, y + cluster_radius)

		c_1_x = random.randint(x_range[0], x_range[1])
		c_1_y = random.randint(y_range[0], y_range[1])

		c_2_x = 2 * x - c_1_x
		c_2_y = 2 * y - c_1_y

		points.append((c_1_x, c_1_y))
		points.append((c_2_x, c_2_y))
		return points

def get_point(x1, y1, x2, y2):
    #express coordinates of the point (x2, y2) with respect to point (x1, y1)
    dx = x2 - x1
    dy = y2 - y1

    alpha = 60./180*math.pi
    #rotate the displacement vector and add the result back to the original point
    xp = x1 + math.cos( alpha)*dx + math.sin(alpha)*dy
    yp = y1 + math.sin(-alpha)*dx + math.cos(alpha)*dy

    return (int(xp), int(yp))

def get_point_square(x1, y1, x2, y2):
	direction = random.randint(0, 1)
	if direction == 0:
		slope_y = (y2 - y1) 
		slope_x = -(x2 - x1)
	else:
		slope_y = -(y2 - y1) 
		slope_x = (x2 - x1)

	x3 = x1 + slope_y
	x4 = x2 + slope_y

	y3 = y1 + slope_x
	y4 = y2 + slope_x 

	return int(x3), int(y3), int(x4), int(y4)

def get_point_rectangle(x1, y1, x2, y2):
	direction = random.randint(0, 1)
	if direction == 0:
		slope_y = (y2 - y1) 
		slope_x = -(x2 - x1)
	else:
		slope_y = -(y2 - y1) 
		slope_x = (x2 - x1)

	length = random.uniform(0, 1) + 0.5

	x3 = x1 + length * slope_y
	x4 = x2 + length * slope_y

	y3 = y1 + length * slope_x
	y4 = y2 + length * slope_x 

	return int(x3), int(y3), int(x4), int(y4)

def make_square(img_size = (64, 64), num_points_per_cluster = 8, cluster_radius = 1):
	is_square = False
	while not is_square:
		point_1_x = random.randint(0 + cluster_radius, img_size[0] - cluster_radius)
		point_1_y = random.randint(0 + cluster_radius, img_size[1] - cluster_radius)

		point_2_x = random.randint(0 + cluster_radius, img_size[0] - cluster_radius)
		point_2_y = random.randint(0 + cluster_radius, img_size[1] - cluster_radius)

		point_3_x, point_3_y, point_4_x, point_4_y = get_point_square(point_1_x, point_1_y, point_2_x, point_2_y)

		if point_3_x + cluster_radius > img_size[0] or point_3_y + cluster_radius > img_size[1] or point_3_x - cluster_radius < 0 or point_3_y - cluster_radius < 0:
				continue

		if point_4_x + cluster_radius > img_size[0] or point_4_y + cluster_radius > img_size[1] or point_4_x - cluster_radius < 0 or point_4_y - cluster_radius < 0:
				continue

		points = []

		points = get_cluster_points(num_points_per_cluster, point_1_x, point_1_y, points, cluster_radius)
		points = get_cluster_points(num_points_per_cluster, point_2_x, point_2_y, points, cluster_radius)
		points = get_cluster_points(num_points_per_cluster, point_3_x, point_3_y, points, cluster_radius)
		points = get_cluster_points(num_points_per_cluster, point_4_x, point_4_y, points, cluster_radius)

		image = np.zeros((img_size[0], img_size[1], 1))

		for p in points:
			image = cv2.circle(image, p, radius=2, color=255, thickness=-1)
		
		is_square = True
		return image

def make_rectangle(img_size = (64, 64), num_points_per_cluster = 8, cluster_radius = 1):
	is_rectangle = False
	while not is_rectangle:
		point_1_x = random.randint(0 + cluster_radius, img_size[0] - cluster_radius)
		point_1_y = random.randint(0 + cluster_radius, img_size[1] - cluster_radius)

		point_2_x = random.randint(0 + cluster_radius, img_size[0] - cluster_radius)
		point_2_y = random.randint(0 + cluster_radius, img_size[1] - cluster_radius)

		point_3_x, point_3_y, point_4_x, point_4_y = get_point_rectangle(point_1_x, point_1_y, point_2_x, point_2_y)

		if point_3_x + cluster_radius > img_size[0] or point_3_y + cluster_radius > img_size[1] or point_3_x - cluster_radius < 0 or point_3_y - cluster_radius < 0:
				continue

		if point_4_x + cluster_radius > img_size[0] or point_4_y + cluster_radius > img_size[1] or point_4_x - cluster_radius < 0 or point_4_y - cluster_radius < 0:
				continue

		points = []

		points = get_cluster_points(num_points_per_cluster, point_1_x, point_1_y, points, cluster_radius)
		points = get_cluster_points(num_points_per_cluster, point_2_x, point_2_y, points, cluster_radius)
		points = get_cluster_points(num_points_per_cluster, point_3_x, point_3_y, points, cluster_radius)
		points = get_cluster_points(num_points_per_cluster, point_4_x, point_4_y, points, cluster_radius)

		image = np.zeros((img_size[0], img_size[1], 1))

		for p in points:
			image = cv2.circle(image, p, radius=2, color=255, thickness=-1)

		is_rectangle = True
		return image

def make_right_angle_triangle(img_size = (64, 64), num_points_per_cluster = 8, cluster_radius = 1):
	is_rectangle = False
	while not is_rectangle:
		point_1_x = random.randint(0 + cluster_radius, img_size[0] - cluster_radius)
		point_1_y = random.randint(0 + cluster_radius, img_size[1] - cluster_radius)

		point_2_x = random.randint(0 + cluster_radius, img_size[0] - cluster_radius)
		point_2_y = random.randint(0 + cluster_radius, img_size[1] - cluster_radius)

		point_3_x, point_3_y, point_4_x, point_4_y = get_point_rectangle(point_1_x, point_1_y, point_2_x, point_2_y)

		if point_3_x + cluster_radius > img_size[0] or point_3_y + cluster_radius > img_size[1] or point_3_x - cluster_radius < 0 or point_3_y - cluster_radius < 0:
				continue

		if point_4_x + cluster_radius > img_size[0] or point_4_y + cluster_radius > img_size[1] or point_4_x - cluster_radius < 0 or point_4_y - cluster_radius < 0:
				continue

		points = []

		points = get_cluster_points(num_points_per_cluster, point_1_x, point_1_y, points, cluster_radius)
		points = get_cluster_points(num_points_per_cluster, point_2_x, point_2_y, points, cluster_radius)
		points = get_cluster_points(num_points_per_cluster, point_3_x, point_3_y, points, cluster_radius)

		image = np.zeros((img_size[0], img_size[1], 1))

		for p in points:
			image = cv2.circle(image, p, radius=2, color=255, thickness=-1)
		
		is_rectangle = True
		return image

def make_equilateral_triangle(img_size = (64, 64), make_equilateral = True, num_points_per_cluster = 8, cluster_radius = 1):
	if make_equilateral:
		is_equilateral = False
		while not is_equilateral:
			point_1_x = random.randint(0 + cluster_radius, img_size[0] - cluster_radius)
			point_1_y = random.randint(0 + cluster_radius, img_size[1] - cluster_radius)

			point_2_x = random.randint(0 + cluster_radius, img_size[0] - cluster_radius)
			point_2_y = random.randint(0 + cluster_radius, img_size[1] - cluster_radius)

			point_3_x, point_3_y = get_point(point_1_x, point_1_y, point_2_x, point_2_y)

			if point_3_x + cluster_radius > img_size[0] or point_3_y + cluster_radius > img_size[1] or point_3_x - cluster_radius < 0 or point_3_y - cluster_radius < 0:
				continue

			points = []
			for c in range(num_points_per_cluster // 2):
				x_range = (point_1_x - cluster_radius, point_1_x + cluster_radius)
				y_range = (point_1_y - cluster_radius, point_1_y + cluster_radius)

				c_1_x = random.randint(x_range[0], x_range[1])
				c_1_y = random.randint(y_range[0], y_range[1])

				c_2_x = 2 * point_1_x - c_1_x
				c_2_y = 2 * point_1_y - c_1_y

				points.append((c_1_x, c_1_y))
				points.append((c_2_x, c_2_y))

			for c in range(num_points_per_cluster // 2):
				x_range = (point_2_x - cluster_radius, point_2_x + cluster_radius)
				y_range = (point_2_y - cluster_radius, point_2_y + cluster_radius)

				c_1_x = random.randint(x_range[0], x_range[1])
				c_1_y = random.randint(y_range[0], y_range[1])

				c_2_x = 2 * point_2_x - c_1_x
				c_2_y = 2 * point_2_y - c_1_y

				points.append((c_1_x, c_1_y))
				points.append((c_2_x, c_2_y))

			for c in range(num_points_per_cluster // 2):
				x_range = (point_3_x - cluster_radius, point_3_x + cluster_radius)
				y_range = (point_3_y - cluster_radius, point_3_y + cluster_radius)

				c_1_x = random.randint(x_range[0], x_range[1])
				c_1_y = random.randint(y_range[0], y_range[1])

				c_2_x = 2 * point_3_x - c_1_x
				c_2_y = 2 * point_3_y - c_1_y

				points.append((c_1_x, c_1_y))
				points.append((c_2_x, c_2_y))

			image = np.zeros((img_size[0], img_size[1], 1))

			for p in points:
				image = cv2.circle(image, p, radius=2, color=255, thickness=-1)
			is_equilateral = True
			return image
	else:
		is_equilateral = False
		while not is_equilateral:
			point_1_x = random.randint(0 + cluster_radius, img_size[0] - cluster_radius)
			point_1_y = random.randint(0 + cluster_radius, img_size[1] - cluster_radius)

			point_2_x = random.randint(0 + cluster_radius, img_size[0] - cluster_radius)
			point_2_y = random.randint(0 + cluster_radius, img_size[1] - cluster_radius)

			point_3_x = random.randint(0 + cluster_radius, img_size[0] - cluster_radius)
			point_3_y = random.randint(0 + cluster_radius, img_size[1] - cluster_radius)

			if point_3_x + cluster_radius > img_size[0] or point_3_y + cluster_radius > img_size[1] or point_3_x - cluster_radius < 0 or point_3_y - cluster_radius < 0:
				continue

			points = []
			for c in range(num_points_per_cluster // 2):
				x_range = (point_1_x - cluster_radius, point_1_x + cluster_radius)
				y_range = (point_1_y - cluster_radius, point_1_y + cluster_radius)

				c_1_x = random.randint(x_range[0], x_range[1])
				c_1_y = random.randint(y_range[0], y_range[1])

				c_2_x = 2 * point_1_x - c_1_x
				c_2_y = 2 * point_1_y - c_1_y

				points.append((c_1_x, c_1_y))
				points.append((c_2_x, c_2_y))

			for c in range(num_points_per_cluster // 2):
				x_range = (point_2_x - cluster_radius, point_2_x + cluster_radius)
				y_range = (point_2_y - cluster_radius, point_2_y + cluster_radius)

				c_1_x = random.randint(x_range[0], x_range[1])
				c_1_y = random.randint(y_range[0], y_range[1])

				c_2_x = 2 * point_2_x - c_1_x
				c_2_y = 2 * point_2_y - c_1_y

				points.append((c_1_x, c_1_y))
				points.append((c_2_x, c_2_y))

			for c in range(num_points_per_cluster // 2):
				x_range = (point_3_x - cluster_radius, point_3_x + cluster_radius)
				y_range = (point_3_y - cluster_radius, point_3_y + cluster_radius)

				c_1_x = random.randint(x_range[0], x_range[1])
				c_1_y = random.randint(y_range[0], y_range[1])

				c_2_x = 2 * point_3_x - c_1_x
				c_2_y = 2 * point_3_y - c_1_y

				points.append((c_1_x, c_1_y))
				points.append((c_2_x, c_2_y))

			image = np.zeros((img_size[0], img_size[1], 1))

			for p in points:
				image = cv2.circle(image, p, radius=2, color=255, thickness=-1)
			is_equilateral = True
			return image

class TriangleDataset(torch.utils.data.Dataset):
	def __init__(self, num_examples = 60000):
		self.num_examples = num_examples

	def __len__(self):
		return self.num_examples

	def __getitem__(self, i):
		n = random.randint(0, 1)
		if n == 0:
			image = make_equilateral_triangle(make_equilateral = True)
		elif n == 1:
			image = make_equilateral_triangle(make_equilateral = False)

		image = torch.from_numpy(image)
		image = image.permute(2, 0, 1)
		return image.float(), torch.tensor(n)#.cuda()
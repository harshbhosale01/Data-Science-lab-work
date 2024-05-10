import numpy as np
import matplotlib.pyplot as plt

def generate_points(num_points):
    # Generate random points within a square [-1, 1] x [-1, 1]
    points = np.random.rand(num_points, 2) * 2 - 1
    return points

def is_inside_circle(x, y):
    # Check if a point (x, y) is inside the circle (unit circle centered at the origin)
    return x**2 + y**2 <= 1

def is_inside_diamond(x, y):
    # Check if a point (x, y) is inside the diamond
    return (abs(x) + abs(y)) <= 1

def monte_carlo_simulation(num_points):
    points = generate_points(num_points)

    circle_points = points[np.sum(points**2, axis=1) <= 1]
    diamond_points = points[(np.abs(points[:, 0]) + np.abs(points[:, 1])) <= 1]

    circle_area = np.sum(np.sum(circle_points**2, axis=1) <= 1) / num_points * 4
    diamond_area = np.sum((np.abs(diamond_points[:, 0]) + np.abs(diamond_points[:, 1])) <= 1) / num_points * 4
    
    # Plot the points
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], c='blue', alpha=0.5, label='Generated Points')
    plt.scatter(circle_points[:, 0], circle_points[:, 1], c='green', alpha=0.5, label='Inside Circle')
    plt.scatter(diamond_points[:, 0], diamond_points[:, 1], c='red', alpha=0.5, label='Inside Diamond')
    
    plt.title('Monte Carlo Simulation for Circle and Diamond')
    plt.legend()
    plt.show()

    return circle_area, diamond_area

# Number of points in the simulation
num_points = 100000

# Run the Monte Carlo simulation
circle_area, diamond_area = monte_carlo_simulation(num_points)

# Calculate theoretical areas
theoretical_circle_area = np.pi * 1**2  # Area of a circle with radius 1
theoretical_diamond_area = 2 * 1**2  # Area of a diamond with side length 2

# Calculate percentage errors
circle_error = np.abs((circle_area - theoretical_circle_area) / theoretical_circle_area) * 100
diamond_error = np.abs((diamond_area - theoretical_diamond_area) / theoretical_diamond_area) * 100

print(f"Estimated Area of Circle: {circle_area}")
print(f"Theoretical Area of Circle: {theoretical_circle_area}")
print(f"Percentage Error in Circle: {circle_error:.2f}%")

print(f"\nEstimated Area of Diamond: {diamond_area}")
print(f"Theoretical Area of Diamond: {theoretical_diamond_area}")
print(f"Percentage Error in Diamond: {diamond_error:.2f}%")


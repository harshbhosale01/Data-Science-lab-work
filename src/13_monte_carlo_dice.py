import numpy as np
import matplotlib.pyplot as plt

def roll_dice():
    outcome = np.random.randint(1, 7)  # Simulate rolling a fair six-sided dice
    return outcome

# Perform Monte Carlo simulation
num_simulations = 10000
num_each_number = {i: 0 for i in range(1, 7)}
for _ in range(num_simulations):
    outcome = roll_dice()
    num_each_number[outcome] += 1

# Calculate probabilities
dice_probabilities = {number: count / num_simulations for number, count in num_each_number.items()}

# Print and plot probabilities
print("Probability of getting each number on the dice:", dice_probabilities)

# Plotting
numbers = list(dice_probabilities.keys())
probabilities = list(dice_probabilities.values())

plt.bar(numbers, probabilities, color='orange')
plt.xlabel('Number on the Dice')
plt.ylabel('Probability')
plt.title('Probability of Getting Each Number on the Dice')
plt.xticks(range(1, 7))
plt.show()


import numpy as np
import matplotlib.pyplot as plt

def draw_card():
    colors = ['red', 'blue', 'green', 'yellow']
    probabilities = [0.25, 0.3, 0.2, 0.25]
    
    card = np.random.choice(colors, p=probabilities)
    return card

# Perform Monte Carlo simulation
num_simulations = 10000
card_counts = {'red': 0, 'blue': 0, 'green': 0, 'yellow': 0}
for _ in range(num_simulations):
    card = draw_card()
    card_counts[card] += 1

# Calculate probabilities
card_probabilities = {color: count / num_simulations for color, count in card_counts.items()}

# Print and plot probabilities
print("Probability of drawing each color card:", card_probabilities)

# Plotting
colors = list(card_probabilities.keys())
probabilities = list(card_probabilities.values())

plt.bar(colors, probabilities, color=colors)
plt.xlabel('Colors')
plt.ylabel('Probability')
plt.title('Probability of Drawing Each Color Card')
plt.show()
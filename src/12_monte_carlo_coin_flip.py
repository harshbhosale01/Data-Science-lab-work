import numpy as np
import matplotlib.pyplot as plt

def coin_flip():
    outcome = np.random.randint(0, 2)  # 0 for heads, 1 for tails
    return outcome

# Perform Monte Carlo simulation
num_simulations = 10000
num_heads = 0
num_tails = 0
for _ in range(num_simulations):
    outcome = coin_flip()
    if outcome == 0:
        num_heads += 1
    else:
        num_tails += 1

# Calculate probabilities
prob_heads = num_heads / num_simulations
prob_tails = num_tails / num_simulations

# Print and plot probabilities
print("Probability of getting heads:", prob_heads)
print("Probability of getting tails:", prob_tails)

# Plotting
labels = ['Heads', 'Tails']
probabilities = [prob_heads, prob_tails]

plt.bar(labels, probabilities, color=['blue', 'green'])
plt.xlabel('Outcomes')
plt.ylabel('Probability')
plt.title('Probability of Getting Heads or Tails in a Coin Flip')
plt.show()


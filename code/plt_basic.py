import matplotlib.pyplot as plt

# Using vscode, we can use:
# "right click" -> "Run in Interactive Window" Then the graph will show in a Jupyter notebook manner.

# Sample data
x = [1, 2, 3, 4, 5]
y = [10, 15, 13, 18, 16]

# Create a plot
plt.plot(x, y)

# Add labels and title
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Simple Plot')

# Display the plot
plt.show()


# Sample data
x = [1, 2, 3, 4, 5]
y = [10, 15, 13, 18, 16]

# Create a figure and a set of subplots (1 row, 2 columns)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot 1: Line plot
axs[0].plot(x, y, 'r--', linewidth=2, label='Line Plot')
axs[0].set_xlabel('X-axis label')
axs[0].set_ylabel('Y-axis label')
axs[0].set_title('Line Plot')
axs[0].legend()

# Plot 2: Scatter plot
axs[1].scatter(x, y, color='b', marker='o', s=100, label='Scatter Plot')
axs[1].set_xlabel('X-axis label')
axs[1].set_ylabel('Y-axis label')
axs[1].set_title('Scatter Plot')
axs[1].legend()

# Display the plots
plt.tight_layout()
plt.show()



x = [1, 2, 3, 4, 5]
y1 = [10, 15, 13, 18, 16]
y2 = [5, 8, 7, 12, 10]

plt.plot(x, y1, label='Line 1')
plt.plot(x, y2, label='Line 2')

plt.legend(loc='upper left', fontsize='large', title='Legend Title', title_fontsize='large', shadow=True, fancybox=True)

plt.show()



# Sample data
x = [1, 2, 3, 4, 5]
y1 = [10, 15, 13, 18, 16]
y2 = [5, 8, 7, 12, 10]

# Plot the data
plt.plot(x, y1, label='Line 1')
plt.plot(x, y2, label='Line 2')

# Add a legend
plt.legend()

# Show the plot
plt.show()


# Create a new figure with a single Axes
fig, ax = plt.subplots()

# Add a plot to the Axes
ax.plot([1, 2, 3, 4], [1, 4, 9, 16])


# Set the axis labels and title
ax.set_xlabel('X-axis label')
ax.set_ylabel('Y-axis label')
ax.set_title('Title of the Plot')

# Set the axis limits
ax.set_xlim(0, 5)
ax.set_ylim(0, 20)

# Customize ticks
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_yticks([0, 5, 10, 15, 20])

# Show grid
ax.grid(True)

fig.show()

# Create a figure with 2x2 grid of Axes
fig, axs = plt.subplots(nrows=2, ncols=2)

# Access each Axes using indexing
ax1 = axs[0, 0]
ax2 = axs[0, 1]
ax3 = axs[1, 0]
ax4 = axs[1, 1]

# Share x-axis and y-axis
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, sharey=True)

# Create a new figure
fig = plt.figure(figsize=(8, 4))

# Create subplots
ax1 = fig.add_subplot(121)  # 1 row, 2 columns, 1st subplot
ax2 = fig.add_subplot(122)  # 1 row, 2 columns, 2nd subplot

# Plot on the subplots
ax1.plot([1, 2, 3], [4, 5, 6], label='Line 1')
ax2.plot([3, 2, 1], [6, 5, 4], label='Line 2')

# Add legends to the subplots
ax1.legend()
ax2.legend()

# Show the figure
plt.show()

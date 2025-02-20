import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, LinearSegmentedColormap
import numpy as np


class ColorTransition:
    def __init__(self, *colors):
        """
        Initialize with a variable number of color names.
        """
        self.colors = colors

    def generate_gradient(self, num_steps):
        """
            Generate a list of distinct colors transitioning evenly through the specified colors.

            :param num_steps: The number of colors to generate in the gradient.
            :return: A list of unique RGB tuples (0-1 range) for the color transitions.
            """
        if len(self.colors) < 2:
            raise ValueError("At least two colors are required for a gradient.")

        # Convert named colors to RGB format
        rgb_colors = [to_rgb(color) for color in self.colors]

        # Create a colormap using the given colors
        cmap = LinearSegmentedColormap.from_list("custom_cmap", rgb_colors)

        # Generate evenly spaced color steps
        gradient_colors = [tuple(cmap(i / (num_steps - 1))[:3]) for i in range(num_steps)]

        # Remove duplicate colors (if any are calculated due to input constraints)
        unique_colors = []
        seen = set()
        for color in gradient_colors:
            if color not in seen:
                seen.add(color)
                unique_colors.append(color)

        # Check if the number of unique colors satisfies the required steps
        if len(unique_colors) < num_steps:
            print(
                f"Info: Fewer unique colors than requested ({len(unique_colors)} < {num_steps}). "
                f"Evenly spreading the colors to fill the steps."
            )
            indices = np.linspace(0, len(unique_colors) - 1, num_steps)
            unique_colors = [unique_colors[int(round(i))] for i in indices]

        return unique_colors


# Example usage
if __name__ == "__main__":
    # Create the ColorTransition object with distinct colors
    transition = ColorTransition("red", "orange", "yellow")

    n = 25
    # Generate a set of distinct colors
    gradient = transition.generate_gradient(n)

    # Output the distinct gradient
    print("Generated Colors:", gradient)

    # Visualize the gradient
    gradient_array = np.linspace(0, 1, n).reshape(1, -1)
    gradient_image = np.zeros((n, n, 3))
    for idx, color in enumerate(gradient):
        gradient_image[:, idx, :] = color

    plt.imshow(gradient_image, aspect='auto')
    plt.axis("off")
    plt.show()

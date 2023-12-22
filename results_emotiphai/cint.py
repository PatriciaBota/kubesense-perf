import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(x, y, ax, n_std=1.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)



import numpy as np
import pandas as pd
data=pd.read_excel('PIC.xlsx')
x = np.array(data['ann_record'])
y = np.array(data['max eda'])


# Now X and y are in the format you specified
fig, ax = plt.subplots()

# Plot the data points
ax.scatter(x, y, label='Data Points')

# Draw the confidence ellipse
confidence_ellipse(x, y, ax, n_std=2, edgecolor='red', label='Confidence Ellipse')

# Set labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.yaxis.set_major_locator(plt.MaxNLocator(10))  # Adjust the number of ticks as needed


# Show the plot
plt.show()
plt.savefig('cintmean.png')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse_params(x, y, n_std=3.0):
    """
    Calculate parameters of the covariance confidence ellipse of x and y.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    center_x, center_y, width, height, angle_deg
        Parameters of the confidence ellipse.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Using a special case to obtain the eigenvalues of this two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    # Calculating the standard deviation of x from the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # Calculating the standard deviation of y...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    angle_rad = np.arctan2(cov[1, 0], cov[0, 0])

    return mean_x, mean_y, scale_x * 2, scale_y * 2, np.degrees(angle_rad)

# Example usage:
# Assuming 'x' and 'y' are your data arrays
center_x, center_y, width, height, angle_deg = confidence_ellipse_params(x, y)

print("Ellipse Parameters:")
print(f"Center (X, Y): ({center_x}, {center_y})")
print(f"Width: {width}")
print(f"Height: {height}")
print(f"Rotation Angle: {angle_deg}")

# Now you can use these parameters as needed, for example, to plot the ellipse

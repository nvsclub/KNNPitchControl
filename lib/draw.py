import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Wedge, Arc

def pitch(bg_color = '#FFFFFF', line_color = '#000000', dpi = 144):
    dev_neutralizer = 3
    horizontal_scalling = 10.5/6.8

    # Background cleanup
    plt.rcParams['figure.figsize'] = (10.5,6.8)
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['figure.facecolor'] = bg_color
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.scatter(50, 50, s=1000000, marker='s', color=bg_color)

    # Set plotting limit
    plt.xlim([-5, 105])
    plt.ylim([-5, 105])

    # Outside lines
    plt.axvline(0, ymin=0.0455, ymax=0.9545, linewidth=3, color=line_color)
    plt.axvline(100, ymin=0.0455, ymax=0.9545, linewidth=3, color=line_color)
    plt.axhline(0, xmin=0.0455, xmax=0.9545, linewidth=3, color=line_color)
    plt.axhline(100, xmin=0.0455, xmax=0.9545, linewidth=3, color=line_color)

    # Midfield line
    plt.axvline(50, ymin=0.0455, ymax=0.9545, linewidth=1, color=line_color)

    # Goals
    plt.axvline(0, ymin=0.4511, ymax=0.5489, linewidth=5, color=line_color)
    plt.axvline(100, ymin=0.4511, ymax=0.5489, linewidth=5, color=line_color)
    plt.axvline(-1, ymin=0.4511, ymax=0.5489, linewidth=5, color=line_color)
    plt.axvline(101, ymin=0.4511, ymax=0.5489, linewidth=5, color=line_color)

    # Small Box
    ## (Width-SmallboxWidth)/2/ScaleTo100, (Margin+(Width-SmallboxWidth)/2/ScaleTo100)/(100+Margins)
    ## (68-7.32-11)/2/0.68, (5+((68-7.32-11)/2/.68))/110
    ## (5+5.5/1.05)/110, 5.25/1.05
    plt.axvline(5.24, ymin=0.3775, ymax=0.6225, linewidth=1, color=line_color)
    plt.axvline(94.76, ymin=0.3775, ymax=0.6225, linewidth=1, color=line_color)

    plt.axhline(36.53, xmin=0.0455, xmax=0.0931, linewidth=1, color=line_color)
    plt.axhline(63.47, xmin=0.0455, xmax=0.0931, linewidth=1, color=line_color)

    plt.axhline(36.53, xmin=0.9069, xmax=0.9545, linewidth=1, color=line_color)
    plt.axhline(63.47, xmin=0.9069, xmax=0.9545, linewidth=1, color=line_color)

    # Big Box
    plt.axvline(15.72, ymin=0.2306, ymax=0.7694, linewidth=1, color=line_color)
    plt.axhline(20.37, xmin=0.0455, xmax=0.1883, linewidth=1, color=line_color)
    plt.axhline(79.63, xmin=0.0455, xmax=0.1883, linewidth=1, color=line_color)

    plt.axvline(84.28, ymin=0.2306, ymax=0.7694, linewidth=1, color=line_color)
    plt.axhline(20.37, xmin=0.8117, xmax=0.9545, linewidth=1, color=line_color)
    plt.axhline(79.63, xmin=0.8117, xmax=0.9545, linewidth=1, color=line_color);

    # Penalty and starting spots and arcs
    plt.scatter([10.4762, 89.5238, 50], [50,50,50], s=1, color=line_color)
    e1 = Arc((10.4762,50), 17.5, 27, theta1=-64, theta2=64, fill=False, color=line_color)
    e2 = Arc((89.5238,50), 17.5, 27, theta1=116, theta2=244, fill=False, color=line_color)
    e3 = Arc((50,50), 17.5, 27, fill=False, color=line_color)
    plt.gcf().gca().add_artist(e1)
    plt.gcf().gca().add_artist(e2)
    plt.gcf().gca().add_artist(e3)
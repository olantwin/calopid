"""Define useful plot styles and utilities."""

import matplotlib.pyplot as plt

# import scienceplots  # noqa: F401

#:plt.style.use(["science", "notebook"])
plt.rcParams["font.size"] = 14
plt.rcParams["axes.formatter.limits"] = -5, 4
plt.rcParams["figure.figsize"] = 6, 4
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def watermark(text="preliminary"):
    """Watermark plot."""
    ax = plt.gca()
    plt.text(
        0.8,
        1.02,
        "AdvSND",
        fontweight="bold",
        fontfamily="sans-serif",
        fontsize=16,
        transform=ax.transAxes,
        usetex=False,
    )
    plt.text(
        0.0,
        1.02,
        text,
        fontfamily="sans-serif",
        fontsize=16,
        transform=ax.transAxes,
        usetex=False,
    )


def plot_event(hitmaps, event=0):
    """Plot CNN image."""
    plt.imshow(hitmaps[event], aspect=0.05)
    plt.xlabel(r"$2 \times \text{Station} + \text{Plane}$")
    plt.ylabel(r"Global strip index")
    watermark()
    plt.savefig(f"plots/hitmap_{event}.pdf")
    plt.savefig(f"plots/hitmap_{event}.png")

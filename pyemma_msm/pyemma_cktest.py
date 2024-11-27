import pyemma
import matplotlib.pyplot as plt

# Script performs pyemma's chapman kolmogorov test on an estimated msm.


def main():
    model = pyemma.load("model.file")
    cktest_result = model.cktest(10)
    print(cktest_result)
    pyemma.plots.plot_cktest(cktest_result)
    plt.savefig("cktest.png", dpi=300)


if __name__ == "__main__":
    main()

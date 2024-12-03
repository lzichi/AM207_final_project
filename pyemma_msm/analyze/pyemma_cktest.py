import pyemma
import matplotlib.pyplot as plt

"""Script performs pyemma's chapman kolmogorov test on an estimated msm,
using 2 PCCA sets to reduce the complexity."""


def main():
    # Load an estimated MSM
    model = pyemma.load("model.file")
    # Compute the CKtest where model is estimated at larger lag times
    num_states = 2
    cktest_result = model.cktest(num_states, mlags=[1, 3, 6, 9, 10])

    # Save results
    print(cktest_result)
    pyemma.plots.plot_cktest(cktest_result)
    plt.savefig("cktest.png", dpi=300)
    pyemma.save(cktest_result)


if __name__ == "__main__":
    main()

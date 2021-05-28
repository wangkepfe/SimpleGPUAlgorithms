import numpy as np
import matplotlib.pyplot as plt
import diskSampler
import poissonSampler
import diskSampler
import halton
import goldenRatio

an = np.linspace(0, 2 * np.pi, 100)
fig, axs = plt.subplots(3, 7)

# ---------------------------------------- Concentric disk sampling 16 ----------------------------------------

diskSamples = diskSampler.get_concentric_disk_samples(squareLength = 4)

axs[0, 0].plot(5 *np.cos(an), 5 *np.sin(an))
axs[0, 0].set_aspect('equal', 'box')
axs[0, 0].set_title('Concentric disk sampling\nsample count = 16', fontsize=10)

for point in diskSamples:
    axs[0, 0].scatter(point[0], point[1])

# ---------------------------------------- Concentric disk sampling 12 ----------------------------------------

diskSamples = diskSampler.get_concentric_disk_samples(squareLength = 5, skipping = 1)

axs[1, 0].plot(5 *np.cos(an), 5 *np.sin(an))
axs[1, 0].set_aspect('equal', 'box')
axs[1, 0].set_title('Concentric disk sampling\nsample count = ' + str(len(diskSamples)), fontsize=10)

for point in diskSamples:
    axs[1, 0].scatter(point[0], point[1])

# ---------------------------------------- Concentric disk sampling 9 ----------------------------------------

diskSamples = diskSampler.get_concentric_disk_samples(squareLength = 3)

axs[2, 0].plot(5 *np.cos(an), 5 *np.sin(an))
axs[2, 0].set_aspect('equal', 'box')
axs[2, 0].set_title('Concentric disk sampling\nsample count = 9', fontsize=10)

for point in diskSamples:
    axs[2, 0].scatter(point[0], point[1])

# ---------------------------------------- Concentric disk sampling + noise 16 ----------------------------------------

diskSamples = diskSampler.get_concentric_disk_samples(squareLength = 4)
seq = halton.halton_sequence(16, 2)
seq = np.transpose(seq)
seq -= 0.5
seq *= 2 * 0.3
diskSamples += seq

axs[0, 1].plot(5 *np.cos(an), 5 *np.sin(an))
axs[0, 1].set_aspect('equal', 'box')
axs[0, 1].set_title('Concentric disk sampling + noise\nsample count = 16', fontsize=10)

for point in diskSamples:
    axs[0, 1].scatter(point[0], point[1])

# ---------------------------------------- Concentric disk sampling + noise 12 ----------------------------------------

diskSamples = diskSampler.get_concentric_disk_samples(squareLength = 5, skipping = 1)
seq = halton.halton_sequence(12, 2)
seq = np.transpose(seq)
seq -= 0.5
seq *= 2 * 0.3
diskSamples += seq

axs[1, 1].plot(5 *np.cos(an), 5 *np.sin(an))
axs[1, 1].set_aspect('equal', 'box')
axs[1, 1].set_title('Concentric disk sampling + noise\nsample count = ' + str(len(diskSamples)), fontsize=10)

for point in diskSamples:
    axs[1, 1].scatter(point[0], point[1])

# ---------------------------------------- Concentric disk sampling + noise 9 ----------------------------------------

diskSamples = diskSampler.get_concentric_disk_samples(squareLength = 3)
seq = halton.halton_sequence(9, 2)
seq = np.transpose(seq)
seq -= 0.5
seq *= 2 * 0.3
diskSamples += seq

axs[2, 1].plot(5 *np.cos(an), 5 *np.sin(an))
axs[2, 1].set_aspect('equal', 'box')
axs[2, 1].set_title('Concentric disk sampling + noise\nsample count = 9', fontsize=10)

for point in diskSamples:
    axs[2, 1].scatter(point[0], point[1])

# ---------------------------------------- Poisson disk sampling 16 ----------------------------------------
np.random.seed(25)
radius = 4.5
poisson = poissonSampler.PoissonSampler(k = 50, r = 1.7, width = radius * 2, height = radius * 2, radius = radius)
poissonsamples = poisson.poissonSample(16)
poissonsamples = np.asarray(poissonsamples)
poissonsamples -= radius

axs[0, 2].plot(5 *np.cos(an), 5 *np.sin(an))
axs[0, 2].set_aspect('equal', 'box')
axs[0, 2].set_title('Poisson disk sampling\nsample count = ' + str(len(poissonsamples)), fontsize=10)

for point in poissonsamples:
    axs[0, 2].scatter(point[0], point[1])

# ---------------------------------------- Poisson disk sampling 12 ----------------------------------------
np.random.seed(29)
radius = 4.5
poisson = poissonSampler.PoissonSampler(k = 50, r = 2.0, width = radius * 2, height = radius * 2, radius = radius)
poissonsamples = poisson.poissonSample(12)
poissonsamples = np.asarray(poissonsamples)
poissonsamples -= radius

axs[1, 2].plot(5 *np.cos(an), 5 *np.sin(an))
axs[1, 2].set_aspect('equal', 'box')
axs[1, 2].set_title('Poisson disk sampling\nsample count = ' + str(len(poissonsamples)), fontsize=10)

for point in poissonsamples:
    axs[1, 2].scatter(point[0], point[1])

# ---------------------------------------- Poisson disk sampling 9 ----------------------------------------
np.random.seed(31)
radius = 4.5
poisson = poissonSampler.PoissonSampler(k = 50, r = 2.3, width = radius * 2, height = radius * 2, radius = radius)
poissonsamples = poisson.poissonSample(9)
poissonsamples = np.asarray(poissonsamples)
poissonsamples -= radius

axs[2, 2].plot(5 *np.cos(an), 5 *np.sin(an))
axs[2, 2].set_aspect('equal', 'box')
axs[2, 2].set_title('Poisson disk sampling\nsample count = ' + str(len(poissonsamples)), fontsize=10)

for point in poissonsamples:
    axs[2, 2].scatter(point[0], point[1])

# ---------------------------------------- halton concentric disk sampling 16 ----------------------------------------
haltonSamples = diskSampler.get_halton_concentric_samples(16)
haltonSamples *= 4.5

axs[0, 3].plot(5 *np.cos(an), 5 *np.sin(an))
axs[0, 3].set_aspect('equal', 'box')
axs[0, 3].set_title('Halton concentric disk sampling\nsample count = ' + str(len(haltonSamples)), fontsize=10)

for point in haltonSamples:
    axs[0, 3].scatter(point[0], point[1])

# ---------------------------------------- halton concentric disk sampling 12 ----------------------------------------
haltonSamples = diskSampler.get_halton_concentric_samples(12)
haltonSamples *= 4.5
axs[1, 3].plot(5 *np.cos(an), 5 *np.sin(an))
axs[1, 3].set_aspect('equal', 'box')
axs[1, 3].set_title('Halton concentric disk sampling\nsample count = ' + str(len(haltonSamples)), fontsize=10)
for point in haltonSamples:
    axs[1, 3].scatter(point[0], point[1])

# ---------------------------------------- halton concentric disk sampling 9 ----------------------------------------
haltonSamples = diskSampler.get_halton_concentric_samples(9)
haltonSamples *= 4.5
axs[2, 3].plot(5 *np.cos(an), 5 *np.sin(an))
axs[2, 3].set_aspect('equal', 'box')
axs[2, 3].set_title('Halton concentric disk sampling\nsample count = ' + str(len(haltonSamples)), fontsize=10)
for point in haltonSamples:
    axs[2, 3].scatter(point[0], point[1])

# ---------------------------------------- sobol concentric disk sampling 16 ----------------------------------------
haltonSamples = diskSampler.get_sobol_concentric_samples(16)
haltonSamples *= 4.5
axs[0, 4].plot(5 *np.cos(an), 5 *np.sin(an))
axs[0, 4].set_aspect('equal', 'box')
axs[0, 4].set_title('Sobol concentric disk sampling\nsample count = ' + str(len(haltonSamples)), fontsize=10)
for point in haltonSamples:
    axs[0, 4].scatter(point[0], point[1])

# ---------------------------------------- sobol concentric disk sampling 12 ----------------------------------------
haltonSamples = diskSampler.get_sobol_concentric_samples(12)
haltonSamples *= 4.5
axs[1, 4].plot(5 *np.cos(an), 5 *np.sin(an))
axs[1, 4].set_aspect('equal', 'box')
axs[1, 4].set_title('Sobol concentric disk sampling\nsample count = ' + str(len(haltonSamples)), fontsize=10)
for point in haltonSamples:
    axs[1, 4].scatter(point[0], point[1])

# ---------------------------------------- sobol concentric disk sampling 9 ----------------------------------------
haltonSamples = diskSampler.get_sobol_concentric_samples(9)
haltonSamples *= 4.5
axs[2, 4].plot(5 *np.cos(an), 5 *np.sin(an))
axs[2, 4].set_aspect('equal', 'box')
axs[2, 4].set_title('Sobol concentric disk sampling\nsample count = ' + str(len(haltonSamples)), fontsize=10)
for point in haltonSamples:
    axs[2, 4].scatter(point[0], point[1])

# ---------------------------------------- golden ratio rejection disk sampling 16 ----------------------------------------
haltonSamples = goldenRatio.golden_rejection_disk(0.5, 2, 16)
haltonSamples *= 4.5
axs[0, 5].plot(5 *np.cos(an), 5 *np.sin(an))
axs[0, 5].set_aspect('equal', 'box')
axs[0, 5].set_title('Golden ratio rejection disk sampling\nsample count = ' + str(len(haltonSamples)), fontsize=10)
for point in haltonSamples:
    axs[0, 5].scatter(point[0], point[1])

# ---------------------------------------- golden ratio rejection  disk sampling 12 ----------------------------------------
haltonSamples = goldenRatio.golden_rejection_disk(0.6, 2, 12)
haltonSamples *= 4.5
axs[1, 5].plot(5 *np.cos(an), 5 *np.sin(an))
axs[1, 5].set_aspect('equal', 'box')
axs[1, 5].set_title('Golden ratio rejection disk sampling\nsample count = ' + str(len(haltonSamples)), fontsize=10)
for point in haltonSamples:
    axs[1, 5].scatter(point[0], point[1])

# ---------------------------------------- golden ratio rejectiondisk sampling 9 ----------------------------------------
haltonSamples = goldenRatio.golden_rejection_disk(0.4, 2, 9)
haltonSamples *= 4.5
axs[2, 5].plot(5 *np.cos(an), 5 *np.sin(an))
axs[2, 5].set_aspect('equal', 'box')
axs[2, 5].set_title('Golden ratio rejection disk sampling\nsample count = ' + str(len(haltonSamples)), fontsize=10)
for point in haltonSamples:
    axs[2, 5].scatter(point[0], point[1])

# ---------------------------------------- golden ratio concentric disk sampling 16 ----------------------------------------
haltonSamples = diskSampler.get_golden_concentric_samples(0.1, 16)
haltonSamples *= 4.5
axs[0, 6].plot(5 *np.cos(an), 5 *np.sin(an))
axs[0, 6].set_aspect('equal', 'box')
axs[0, 6].set_title('Golden ratio concentric disk sampling\nsample count = ' + str(len(haltonSamples)), fontsize=10)
for point in haltonSamples:
    axs[0, 6].scatter(point[0], point[1])

# ---------------------------------------- golden ratio concentric  disk sampling 12 ----------------------------------------
haltonSamples = diskSampler.get_golden_concentric_samples(0.1, 12)
haltonSamples *= 4.5
axs[1, 6].plot(5 *np.cos(an), 5 *np.sin(an))
axs[1, 6].set_aspect('equal', 'box')
axs[1, 6].set_title('Golden ratio concentric disk sampling\nsample count = ' + str(len(haltonSamples)), fontsize=10)
for point in haltonSamples:
    axs[1, 6].scatter(point[0], point[1])

# ---------------------------------------- golden ratio concentric disk sampling 9 ----------------------------------------
haltonSamples = diskSampler.get_golden_concentric_samples(0.1, 9)
haltonSamples *= 4.5
axs[2, 6].plot(5 *np.cos(an), 5 *np.sin(an))
axs[2, 6].set_aspect('equal', 'box')
axs[2, 6].set_title('Golden ratio concentric disk sampling\nsample count = ' + str(len(haltonSamples)), fontsize=10)
for point in haltonSamples:
    axs[2, 6].scatter(point[0], point[1])

fig.tight_layout()

plt.show()
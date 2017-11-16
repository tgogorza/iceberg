import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def get_data(data_path):
    with open(data_path, 'rb') as f:
        data = json.load(f)
    print 'Loaded {} items'.format(len(data))
    return pd.DataFrame(data)


df = get_data('data/train.json')
df.inc_angle = pd.to_numeric(df['inc_angle'], errors='coerce')

ships = df[df.is_iceberg == 0]
icebergs = df[df.is_iceberg == 1]

ship_bands_1 = np.stack([np.array(band).reshape(75, 75) for band in ships['band_1']], axis=0)
ship_bands_2 = np.stack([np.array(band).reshape(75, 75) for band in ships['band_2']], axis=0)
ice_bands_1 = np.stack([np.array(band).reshape(75, 75) for band in icebergs['band_1']], axis=0)
ice_bands_2 = np.stack([np.array(band).reshape(75, 75) for band in icebergs['band_2']], axis=0)
ship_bands = np.stack([ship_bands_1, ship_bands_2], axis=3)
ice_bands = np.stack([ice_bands_1, ice_bands_2], axis=3)

# df.band_1 = df.band_1.apply(lambda x: np.array(x).reshape(75, 75))
# df.band_2 = df.band_2.apply(lambda x: np.array(x).reshape(75, 75))

# df.bands = df.apply(lambda row: np.stack([row.band_1[0], row.band_2[0]], axis=3), axis=1)

# bands_1 = df.band_1
# bands_2 = df.band_2
# np.stack([df.band_1.head(1)[0], df.band_2.head(1)[0]], axis=3)

# Ships vs Icebergs
# Col 0 -> Ship band 1
# Col 1 -> Ship band 2
# Col 2 -> Iceberg band 1
# Col 3 -> Iceberg band 2
fig, axs = plt.subplots(10, 6)
for i_plot, row in enumerate(axs):
    row[0].imshow(ship_bands_1[i_plot])
    row[1].imshow(ship_bands_2[i_plot])
    row[2].imshow(ship_bands_1[i_plot] + ship_bands_2[i_plot])
    row[3].imshow(ice_bands_1[i_plot])
    row[4].imshow(ice_bands_2[i_plot])
    row[5].imshow(ice_bands_1[i_plot] + ice_bands_2[i_plot])

ship_bands_plus = ship_bands_1 + ship_bands_2
ice_bands_plus = ice_bands_1 + ice_bands_2
ship_plus_flat = ship_bands_plus.reshape(ship_bands_plus.shape[0], -1)
ice_plus_flat = ice_bands_plus.reshape(ice_bands_plus.shape[0], -1)

fig, plts = plt.subplots(1, 2)
for i_plot, row in enumerate(plts):
    # row[0].hist(ship_bands_1[0, :], bins=100)
    # row[1].hist(ship_plus_flat[0, :], bins=100)
    row[0].hist(ship_plus_flat[i_plot, :], bins=100)
    # row[3].hist(ship_plus_flat[0, :], bins=100)
    # row[4].hist(ship_plus_flat[0, :], bins=100)
    row[1].hist(ice_plus_flat[i_plot, :], bins=100)



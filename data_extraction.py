import pandas as pd
import numpy as np


def infer_value_linearly(x: int, x1: int, x2: int, v1: float, v2: float) -> float:
    a = (v2 - v1) / (x2 - x1)
    b = (x2 * v1 - x1 * v2) / (x2 - x1)

    return a * x + b


fractional_cover = pd.read_csv("datasets/fractional_cover.csv")
spectra_forest_floor = pd.read_csv("datasets/spectra_forest_floor.csv")
complete_fc = pd.DataFrame(columns=["plot_ID", "location", "vasc", "nonvasc", "lichen", "intactlitt", "decomplitt"])

for i, sample in fractional_cover.iterrows():
    for loc in range(1, 16):
        corresponding_spectra = spectra_forest_floor[
                   (spectra_forest_floor["plot_ID"] == sample["plot_ID"]) &
                   (spectra_forest_floor["location"] == loc)
               ]

        # Verify one-to-one correspondence between fractional cover and spectra forest floor samples
        assert corresponding_spectra.index[0] == i * 15 + loc - 1, "The two datasets do not match."

        # Skip empty lines
        if np.isnan(corresponding_spectra.iloc[0]["wl350"]):
            spectra_forest_floor = spectra_forest_floor.drop(index=corresponding_spectra.index[0])
            continue

        if loc <= 5:
            point1 = 1
            point2 = 2
            loc1 = 1
            loc2 = 5

        elif loc <= 9:
            point1 = 2
            point2 = 3
            loc1 = 5
            loc2 = 9

        else:
            point1 = 3
            point2 = 4
            loc1 = 9
            loc2 = 13

        complete_fc.loc[len(complete_fc)] = {
            "plot_ID": sample["plot_ID"],
            "location": loc,
            "vasc": infer_value_linearly(loc, loc1, loc2, sample[f"vasc_q{point1}"], sample[f"vasc_q{point2}"]),
            "nonvasc": infer_value_linearly(loc, loc1, loc2, sample[f"nonvasc_q{point1}"], sample[f"nonvasc_q{point2}"]),
            "lichen": infer_value_linearly(loc, loc1, loc2, sample[f"lichen_q{point1}"], sample[f"lichen_q{point2}"]),
            "intactlitt": infer_value_linearly(loc, loc1, loc2, sample[f"intactlitt_q{point1}"], sample[f"intactlitt_q{point2}"]),
            "decomplitt": infer_value_linearly(loc, loc1, loc2, sample[f"decomplitt_q{point1}"], sample[f"decomplitt_q{point2}"]),
        }

complete_fc.to_csv("datasets/complete_fc_loc.csv", sep=",", index=False)
complete_fc_noloc = complete_fc.drop(columns=["plot_ID", "location"])
complete_fc_noloc.to_csv("datasets/complete_fc.csv", sep=",", index=False)
sff_noloc = spectra_forest_floor.drop(columns=["plot_ID", "location"])
sff_noloc.to_csv("datasets/sff.csv", sep=",", index=False)

import lib
from collections import OrderedDict
cuttoff_length = 24
runs = lib.get_runs([f"scan_trafo_{cuttoff_length}"])
runs = lib.common.group(runs, ['transformer.variant', "scan.length_cutoff"])

variants = OrderedDict()
variants["Uni. Trafo"] = ["universal_noscale"]
variants["Comp. Trafo 1"] = ["compositional_1r_16"]
variants["Comp. Trafo 2"] = ["compositional_2r_16"]
variants["Comp. Trafo 4"] = ["compositional_4r_16"]
variants["Comp. Trafo 8"] = ["compositional_8r_16"]


lengths = [cuttoff_length]

best = [0.58, 0.54, 0.69, 0.82, 0.88]

stats = lib.common.calc_stat(runs, lambda name: name.endswith("val/accuracy/total"), tracker=lib.MedianTracker)

ourtab = []
for i, (v, vlist) in enumerate(variants.items()):
    ourtab.append([])
    for l in lengths:
        all_stats = [stats.get(f"transformer.variant_{vn}/scan.length_cutoff_{l}") for vn in vlist]
        all_stats = [a for a in all_stats if a is not None]
        assert all([len(a) == 1 for a in all_stats])
        all_stats = [list(a.values())[0].get() for a in all_stats]
        ourtab[-1].append(max(all_stats))

for l in ourtab:
    for i, v in enumerate(l):
        best[i] = max(best[i], v)

for i, (v, vn) in enumerate(variants.items()):
    pstr = []
    for j, val in enumerate(ourtab[i]):
        pstr.append(("\\bf" if best[j] - val < 0.02 else "") + f"{val:.2f}")

    print(f"{' & ' if i>0 else ''}\\texttt{{{v}}}\\xspace & {' & '.join(pstr)} \\\\")

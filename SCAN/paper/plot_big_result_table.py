import lib
from collections import OrderedDict

data = OrderedDict()

data[f"SCAN (length cutoff=22)"] = lib.get_runs([f"scan_trafo_22"], filters = {"config.scan.length_cutoff.value": 22}), "val", None, "$0.00^{[1]}$"
data[f"SCAN (length cutoff=24)"] = lib.get_runs([f"scan_trafo_24"], filters = {"config.scan.length_cutoff.value": 24}), "val", None, "$0.00^{[1]}$"
data[f"SCAN (length cutoff=25)"] = lib.get_runs([f"scan_trafo_25"], filters = {"config.scan.length_cutoff.value": 25}), "val", None, "$0.00^{[1]}$"
data[f"SCAN (length cutoff=26)"] = lib.get_runs([f"scan_trafo_26"], filters = {"config.scan.length_cutoff.value": 26}), "val", None, "$0.00^{[1]}$"
data[f"SCAN (length cutoff=27)"] = lib.get_runs([f"scan_trafo_27"], filters = {"config.scan.length_cutoff.value": 27}), "val", None, "$0.00^{[1]}$"
data[f"SCAN (length cutoff=28)"] = lib.get_runs([f"scan_trafo_28"], filters = {"config.scan.length_cutoff.value": 28}), "val", None, "$0.00^{[1]}$"
data[f"SCAN (length cutoff=29)"] = lib.get_runs([f"scan_trafo_29"], filters = {"config.scan.length_cutoff.value": 29}), "val", None, "$0.00^{[1]}$"
data[f"SCAN (length cutoff=30)"] = lib.get_runs([f"scan_trafo_30"], filters = {"config.scan.length_cutoff.value": 30}), "val", None, "$0.00^{[1]}$"
data["a"] = None, None, None, None


columns = OrderedDict()
columns["Uni. Trafo"] = ["universal_noscale"]
columns["Comp. Trafo 1"] = ["compositional_1r_16"]
columns["Comp. Trafo 2"] = ["compositional_2r_16"]
columns["Comp. Trafo 4"] = ["compositional_4r_16"]
columns["Comp. Trafo 8"] = ["compositional_8r_16"]


def average_accuracy(runs, split_name, step) -> float:
    st = lib.StatTracker()
    runs = list(runs)
    # print(runs)
    it = max([r.summary["iteration"] for r in runs])
    for r in runs:
        if f"validation/{split_name}/accuracy/total" not in r.summary:
            continue

        if step is None:
            st.add(r.summary[f"validation/{split_name}/accuracy/total"])
            assert r.summary["iteration"] == it, f"Inconsistend final iteration for run {r.id}: {r.summary['iteration']} instead of {it}"
        else:
            hist = r.history(keys=[f"validation/{split_name}/accuracy/total", "iteration"], pandas=False)
            for h in hist:
                if h["iteration"] == step:
                    st.add(h[f"validation/{split_name}/accuracy/total"])
                    break
            else:
                assert False, f"Step {step} not found."
    return st.get()


def format_results(runs, split_name, step) -> str:
    run_group = lib.common.group(runs, ['transformer.variant'])

    cols = []
    for clist in columns.values():
        found = []
        for c in clist:
            full_name = f"transformer.variant_{c}"
            if full_name in run_group:
                found.append(average_accuracy(run_group[full_name], split_name, step))

        cols.append(max(found, key=lambda x: x.mean) if found else None)
    maxval = max(c.mean for c in cols if c is not None)
    cols = [(("\\bf{" if c.mean == maxval else "") + f"{c.mean:.2f} $\\pm$ {c.std:.2f}" +
             ("}" if c.mean == maxval else "")) if c is not None else "-" for c in cols]
    return " & ".join(cols)


print(" & " + " & ".join(columns.keys()) + " & Reported\\\\")
print("\\midrule")
for dname, (runs, splitname, at_step, best_other) in data.items():
    print(dname, runs, splitname, at_step, best_other)
    if runs is None:
        print("\\midrule")
    else:
        print(f"{dname} & {format_results(runs, splitname, at_step)} & {best_other} \\\\")

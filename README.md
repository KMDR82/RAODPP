# RAOD++ — iso-latency and iso-energy comparison of input resolution and model capacity

Code, measurements and manuscript source for:

> **Iso-Latency and Iso-Energy Comparison of Input Resolution and Model Capacity for
> Real-Time UAV Object Detection.** Ahmet Akkaya, Bandırma Onyedi Eylül University.
> Submitted to *Image and Vision Computing*, special issue VSI: VP-NAV.

The question is where a fixed inference budget is better spent in a real-time UAV
detector: on a larger input, or on a larger model. The answer here rests on cost measured
on the device at batch size one rather than inferred from parameter counts or FLOPs.

## What is in this repository

```
scripts/     training, measurement, analysis and figure code
notebooks/   dataset preparation and the exploratory work that preceded the grid
output/      runs.csv, the per-run record every number is computed from
output/figdata/   the per-figure and per-table CSVs that the paper's captions cite
fig/         the figures as they appear in the manuscript
tex/         manuscript source and bibliography
```

Every number in the paper comes from `output/runs.csv` by way of a script in `scripts/`.
No value is typed into the manuscript by hand, and `scripts/check_repo.py` checks that
every path a caption cites actually exists here.

## Reproducing the analysis

The analysis runs on the recorded measurements and needs no GPU. From a clone with
`output/runs.csv` in place:

```bash
pip install -r requirements.txt

python scripts/iso_analysis.py --dataset visdrone --cost latency
python scripts/iso_analysis.py --dataset visdrone --cost energy
python scripts/iso_analysis.py --dataset visdrone --cost energy_marginal
python scripts/holm_table.py            # Tables 4 and 5, with the Holm correction
python scripts/dominance_table.py       # Table 6, the interpolation-free comparison
python scripts/per_seed_table.py        # Appendix A
python scripts/make_figures.py          # every figure
python scripts/make_graphical_abstract.py
python scripts/verify_figures.py        # figures against the manuscript's numbers
```

`bash scripts/run_analysis.sh` does all of the above in one pass and writes a log to
`output/logs/`.

## Reproducing the experiments

Training needs the datasets and a GPU. The grid was run on a single NVIDIA RTX 4500 Ada
(24 GB) under WSL2.

```bash
bash scripts/run_training.sh visdrone
python scripts/measure_cost_only.py --dataset visdrone
bash scripts/run_training.sh uavdt
python scripts/measure_cost_only.py --dataset uavdt
```

`run_training.sh` refuses to start if another training process is already running, passes
`--workers 0`, and checks afterwards that no condition came out with zero variance across
seeds. `train_resolution.py` is resumable, so an interrupted run can be restarted.

`--workers 0` is not optional. With the dataloader at eight workers the seed does not reach
the training loop on one of the datasets: three nominally independent runs produce
bit-identical trajectories and a standard deviation of exactly zero. `output/PROJECT_LOG.md`
records how this was found.

## Datasets

Neither dataset is redistributed here.

- **VisDrone2019-DET**: obtain from the VisDrone challenge release. The paper uses the
  official train and validation splits, 6,471 and 548 images.
- **UAVDT**: obtain from the authors' release. The paper uses the detection subset only
  (`M####` sequences); the `S####` tracking sequences are excluded, and
  `scripts/convert_uavdt.py` logs which subset each file came from.

## Environment

Python 3.14, PyTorch 2.13.0+cu130, Ultralytics 8.4.118, CUDA 13.0. Exact versions are in
`requirements.txt`. Inference uses the library defaults recorded in each run's `args.yaml`:
confidence 0.001 for validation, IoU 0.7, at most 300 detections per image, no
class-agnostic NMS and no test-time augmentation.

## How cost is measured

Latency is the mean over 1,000 timed inferences after 100 warm-up passes with explicit
device synchronisation, at batch size one, and the value reported is the median of three
such passes on an otherwise idle GPU. Energy is that latency multiplied by the mean board
power sampled through NVML during the same passes. Board power includes the device's idle
draw, which on this hardware is a third to two thirds of the total, so the paper reports
energy both ways and `measure_cost_only.py` samples the idle floor before every run.

## Provenance

`output/PROJECT_LOG.md` is an append-only record of the study, including the defects found
along the way and what each one cost: a confounded capacity arm that inverted the headline
result until it was rebuilt, a dataloader setting that removed seed variance, a
multiplicity correction that was computed once by hand and never recomputed, and several
figures and captions that went stale when the arms changed. The protocol amendments record
what was fixed before the results were seen and what was fixed after.

## Licence

See `LICENSE`. Code is released under the MIT licence; the measurement files under
`output/` and the figures under `fig/` are released under CC BY 4.0.

## Citation

A BibTeX entry will be added here once the paper has a DOI.

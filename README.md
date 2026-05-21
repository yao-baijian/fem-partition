# FEM

`fem-partition` is a python library for solving graph partition problems using `FEM` framework. The current build-in problem types are:
* normal graph balance minimum cut
* hypergraph balance minimum cut

## Installation

1. One can use conda to install the package with the following commands:
    ```bash
    conda env create -f environment.yml
    ```
    this will create an environment named `fem` with all the dependencies except for the pytorch, then activate the environment with `conda activate fem`.

2. Then `pytorch` have to be installed manually with 
    ```bash
    pip3 install torch torchvision torchaudio
    ```
    see the [pytorch website](https://pytorch.org/) for more details.

## Recent Work / How to run the modified tests

This workspace has been updated to support FEM-based coarse initial partitions
and tool-native refinement when possible. Key test drivers updated:

- `tests/test_bmincut.py` — runs several partitioning modes (direct FEM,
    METIS, KaHyPar, KaFFPa-like). FEM now produces q-way coarse initial
    partitions and the refiners are invoked explicitly. For KaFFPa (kahip
    wrapper) a local replacement `simple_kaffpa` is used because the installed
    wrapper does not accept an initial `part=` parameter.
- `tests/test_hyper_bmincut.py` — runs hypergraph PUBO and coarsening flows.
    When KaHyPar is available, it will be used to refine the FEM-projected
    partition; otherwise a greedy local-refinement is applied.

Run tests (example):

```powershell
python -u tests/test_bmincut.py
python -u tests/test_hyper_bmincut.py
```

Files of interest:

- `tests/utils.py` — contains `simple_kaffpa(...)` (a small KL/FM-like local
    refiner) and `call_pymetis_with_part(...)` which attempts to pass an
    initial partition to `pymetis.part_graph` when supported by the wrapper.
- `FEM/initial_partition.py` — QUBO/Ising builders and FEM integration. Use
    `FEM.from_couplings('bmincut', q=k)` to request a q-way solver.

If you'd like, I can:
- Expand `simple_kaffpa` into a full FM implementation (bucket queue),
- Add unit tests for the new helpers,
- Or revert to calling native KaFFPa if you provide a compatible kaffpa wrapper.

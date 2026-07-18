# Linear Drift Dry Run

This sandbox mirrors the input construction in `experiments/run_single_case.py`
without modifying production code.

For `--sensor P_IM --fault linear_drift --severity 0.05`, the feature order is:

`[Torque, p_0, T_IM, P_IM, EGR_Rate, ECU_VTG_Pos]`.

The healthy matrix has shape `(300, 6)`. Only its P_IM column is noisy:

`P_IM_healthy(t) = 2.8 + Normal(0, 0.02 * 2.8)`.

Every other healthy channel remains exactly constant. The faulty matrix begins as
a copy of the healthy matrix, then its P_IM column is fully replaced. Therefore
the final faulty matrix is clean in every channel:

`P_IM_faulty(t) = 2.8 + 2.8 * 0.05 * t / 299`.

At the endpoints, P_IM is `2.8` and `2.94`. The healthy P_IM samples vary
randomly near `2.8`; they are not reused by the drift injector.

The two ANN calls receive separate `(300, 6)` matrices. Exact MF_IA predictions
cannot be inferred from the six scalars alone because they depend on the saved
scalers and learned ANN weights. Run the sandbox with `--evaluate-ann` in the
project's trained-model environment to print those actual values.

The correct single-case residual is:

`faulty_prediction(t) - healthy_prediction(t)`.

Because the faulty P_IM is clean while healthy P_IM contains independent random
measurement variation, subtraction does not cancel the healthy variation. For
a local ANN sensitivity `dF/dP_IM = g` and healthy noise `epsilon(t)`, the
residual has the approximate form:

`g * 0.14 * t / 299 - g * epsilon(t)`.

The second term is high-frequency. It is the expected source of rapid changes
in Figure B, even when the drift term itself is smooth.

`src/experiment/runner.py` is not identical to the single-case path: it stores
`faulty_prediction - healthy_ground_truth` as `residual` and
`faulty_prediction - healthy_prediction` as `prediction_error`. The single-case
runner uses the requested definitions.

Run:

```bat
python simulation_sandbox\linear_drift_trace.py --severity 0.05 --seed 42
python simulation_sandbox\linear_drift_trace.py --severity 0.05 --seed 42 --evaluate-ann
```

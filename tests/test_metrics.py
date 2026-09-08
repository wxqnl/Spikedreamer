import pytest
import torch
from spikedreamer.metrics import OperationCounter, energy_proxy


def test_binary_activity_does_not_discount_dense_gpu_macs():
    linear = torch.nn.Linear(3, 2, bias=False)
    with OperationCounter(linear) as counter:
        linear(torch.tensor([[1., 0., 1.], [0., 1., 0.]]))
    ledger = counter.result()
    assert ledger["totals"]["dense_macs"] == 12
    assert ledger["totals"]["linear_binary_synops_proxy"] == 6
    # Arbitrary test coefficients, not hardware-energy evidence.
    estimate = energy_proxy(ledger, 4, 1, 2, "unit test only")
    assert estimate["estimated_arithmetic_joules"] == pytest.approx(6e-12)


def test_fused_gru_projection_macs_are_counted():
    cell = torch.nn.GRUCell(3, 4)
    with OperationCounter(cell) as counter:
        cell(torch.ones(2, 3), torch.zeros(2, 4))
    total = counter.result()["totals"]
    assert total["dense_macs"] == 2 * 3 * 4 * (3 + 4)
    assert total["analog_input_dense_macs"] == total["dense_macs"]

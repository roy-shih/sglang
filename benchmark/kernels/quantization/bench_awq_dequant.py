import argparse
import random
import itertools

import torch
import triton
from sglang.srt.layers.quantization.awq_dequant import awq_dequantize_triton
from vllm.model_executor.layers.quantization.awq_triton import awq_dequantize_triton as awq_dequantize_triton_vllm
import vllm._C

device = "cuda"

def reverse_awq_order(t: torch.Tensor):
    bits = 4
    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    reverse_order_tensor = torch.arange(
        t.shape[-1],
        dtype=torch.int32,
        device=t.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    t = t[:, reverse_order_tensor] & 0xF
    return t


# qweights - [R     , C // 8], int32
# scales   - [R // G, C     ], float16
# zeros    - [R // G, C // 8], int32
@torch.compile(backend="inductor")
def awq_dequantize_torch(qweight: torch.Tensor, scales: torch.Tensor,
                         qzeros: torch.Tensor,
                         group_size: int) -> torch.Tensor:

    if group_size == -1:
        group_size = qweight.shape[0]

    bits = 4
    shifts = torch.arange(0, 32, bits, device=qzeros.device)

    iweights = torch.bitwise_right_shift(qweight[:, :, None],
                                         shifts[None, None, :]).to(torch.int8)

    iweights = iweights.view(iweights.shape[0], -1)

    zeros = torch.bitwise_right_shift(qzeros[:, :, None],
                                      shifts[None, None, :]).to(torch.int8)
    zeros = zeros.view(qzeros.shape[0], -1)
    zeros = reverse_awq_order(zeros)

    iweights = reverse_awq_order(iweights)

    iweights = torch.bitwise_and(iweights, (2**bits) - 1)
    zeros = torch.bitwise_and(zeros, (2**bits) - 1)

    scales = scales.repeat_interleave(group_size, dim=0)
    zeros = zeros.repeat_interleave(group_size, dim=0)
    return (iweights - zeros) * scales


def _test_accuracy_once(qweight_row, qweight_col, group_size):
    qweight_dtype = torch.int32
    scales_rows = qweight_row // group_size
    scales_cols = qweight_col * 8
    scales_dtype = torch.float16
    zeros_rows = scales_rows
    zeros_cols = qweight_col
    zeros_dtype = torch.int32

    random.seed(0)

    qweight = torch.randint(0,
                            torch.iinfo(torch.int32).max,
                            (qweight_row, qweight_col),
                            dtype=qweight_dtype,
                            device=device)
    scales = torch.rand(scales_rows,
                        scales_cols,
                        dtype=scales_dtype,
                        device=device)
    zeros = torch.randint(0,
                          torch.iinfo(torch.int32).max,
                          (zeros_rows, zeros_cols),
                          dtype=zeros_dtype,
                          device=device)

    iweights_triton = awq_dequantize_triton(qweight, scales, zeros)
    
    assert (not torch.any(torch.isinf(iweights_triton))
            and not torch.any(torch.isnan(iweights_triton)))
    
    iweights_triton_vllm = awq_dequantize_triton_vllm(qweight, scales, zeros)

    assert (not torch.any(torch.isinf(iweights_triton_vllm))
            and not torch.any(torch.isnan(iweights_triton_vllm)))

    iweights_torch = awq_dequantize_torch(qweight, scales, zeros, group_size)

    assert (not torch.any(torch.isinf(iweights_torch))
            and not torch.any(torch.isnan(iweights_torch)))
    
    # iweights_cuda = torch.ops._C.awq_dequantize(qweight, scales, zeros, 0, 0, 0)

    torch.testing.assert_close(iweights_triton_vllm, iweights_torch)
    torch.testing.assert_close(iweights_triton, iweights_torch)

    # torch.testing.assert_close(iweights_cuda, iweights_torch)


def test_accuracy():
    # DeepSeek-V2-Lite, DeepSeek-V2/DeepSeek-V3
    qweight_rows = [512]
    qweight_cols = [512, 4096]
    group_sizes = [64]

    for qweight_row in qweight_rows:
        for qweight_col in qweight_cols:
            for group_size in group_sizes:
                _test_accuracy_once(qweight_row, qweight_col, group_size)


qweight_rows = [512]
qweight_cols = [512, 4096]
group_sizes = [64]
configs = list(itertools.product(qweight_rows, qweight_cols, group_sizes))
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["qweight_row", "qweight_col", "group_size"],
        x_vals=[list(_) for _ in configs],
        x_log=False,
        line_arg="provider",
        line_vals=["triton from vllm", "triton", "torch.compile"],
        line_names=["triton from vllm", "triton", "torch.compile"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="ms",
        plot_name="awq dequant",
        args={},
    )
)
def benchmark(qweight_row, qweight_col, group_size, provider):
    qweight_dtype = torch.int32
    scales_rows = qweight_row // group_size
    scales_cols = qweight_col * 8
    scales_dtype = torch.float16
    zeros_rows = scales_rows
    zeros_cols = qweight_col
    zeros_dtype = torch.int32

    random.seed(0)

    qweight = torch.randint(0,
                            torch.iinfo(torch.int32).max,
                            (qweight_row, qweight_col),
                            dtype=qweight_dtype,
                            device=device)
    scales = torch.rand(scales_rows,
                        scales_cols,
                        dtype=scales_dtype,
                        device=device)
    zeros = torch.randint(0,
                          torch.iinfo(torch.int32).max,
                          (zeros_rows, zeros_cols),
                          dtype=zeros_dtype,
                          device=device)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton from vllm":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: awq_dequantize_triton_vllm(qweight, scales, zeros),
            quantiles=quantiles,
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: awq_dequantize_triton(qweight, scales, zeros),
            quantiles=quantiles,
        )
    # if provider == "cuda from vllm":
    #     ms, min_ms, max_ms = triton.testing.do_bench(
    #         lambda: torch.ops._C.awq_dequantize(qweight, scales, zeros, 0, 0, 0),
    #         quantiles=quantiles,
    #     )
    if provider == "torch.compile":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: awq_dequantize_torch(qweight, scales, zeros, group_size),
            quantiles=quantiles,
        )

    return round(1000*ms, 2), round(1000*min_ms, 2), round(1000*max_ms, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./bench_awq_dequant_res",
        help="Path to save awq dequant benchmark results",
    )
    args = parser.parse_args()

    test_accuracy()

    benchmark.run(print_data=True, show_plots=True, save_path=args.save_path)

import torch
from torch import nn
import torch.nn.functional as F
import tilelang as tl
from tilelang.autotuner import *
import tilelang.language as T
from tilelang.profiler import do_bench

num_split = 4


@tl.jit(out_idx=[5], pass_configs={tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True})
def _mha_infer(batch, heads, seqlen_q, seqlen_kv, dim, is_causal, block_M, block_N):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape_q = [batch, seqlen_q, heads, dim]
    shape_kv = [batch, seqlen_kv, heads, dim]
    part_shape = [batch, seqlen_q, heads, num_split, dim]
    dtype = "float16"
    accum_dtype = "float"

    @T.macro
    def MMA0(
        K: T.Tensor(shape_kv, dtype),  # type: ignore
        Q_shared: T.SharedBuffer([block_M, dim], dtype),  # type: ignore
        K_shared: T.SharedBuffer([block_N, dim], dtype),  # type: ignore
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),  # type: ignore
        k: T.int32,
        mid: T.int32,
        hid: T.int32,
        bid: T.int32,
        sid: T.int32,
    ):
        T.copy(
            K[bid, (seqlen_kv // num_split) * sid + k * block_N:(seqlen_kv // num_split) * sid +
              (k + 1) * block_N, hid, :], K_shared)
        # TODO: Handle causal split case
        if is_causal:
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(mid * block_M + i >= k * block_N + j, 0,
                                             -T.infinity(acc_s.dtype))
        else:
            T.clear(acc_s)
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def MMA1(
        V: T.Tensor(shape_kv, dtype),  # type: ignore
        V_shared: T.SharedBuffer([block_M, dim], dtype),  # type: ignore
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),  # type: ignore
        acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),  # type: ignore
        k: T.int32,
        hid: T.int32,
        bid: T.int32,
        sid: T.int32,
    ):
        T.copy(
            V[bid, (seqlen_kv // num_split) * sid + k * block_N:(seqlen_kv // num_split) * sid +
              (k + 1) * block_N, hid, :], V_shared)
        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def Softmax(
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),  # type: ignore
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),  # type: ignore
            scores_max: T.FragmentBuffer([block_M], accum_dtype),  # type: ignore
            scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),  # type: ignore
            scores_scale: T.FragmentBuffer([block_M], accum_dtype),  # type: ignore
            scores_sum: T.FragmentBuffer([block_M], accum_dtype),  # type: ignore
            logsum: T.FragmentBuffer([block_M], accum_dtype),  # type: ignore
    ):
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        # To do causal softmax, we need to set the scores_max to 0 if it is -inf
        # This process is called Check_inf in FlashAttention3 code, and it only need to be done
        # in the first ceil_div(kBlockM, kBlockN) steps.
        # for i in T.Parallel(block_M):
        #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
        for i in T.Parallel(block_M):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
        for i, j in T.Parallel(block_M, block_N):
            # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            # max * log_2(e)) This allows the compiler to use the ffma
            # instruction instead of fadd and fmul separately.
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(block_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
        T.copy(acc_s, acc_s_cast)

    @T.macro
    def Rescale(
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),  # type: ignore
            scores_scale: T.FragmentBuffer([block_M], accum_dtype),  # type: ignore
    ):
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] *= scores_scale[i]

    @T.macro
    def flash_attn_split(
            Q: T.Tensor(shape_q, dtype),  # type: ignore
            K: T.Tensor(shape_kv, dtype),  # type: ignore
            V: T.Tensor(shape_kv, dtype),  # type: ignore
            glse: T.Tensor([batch, heads, num_split, seqlen_q], dtype),  # type: ignore
            Output_partial: T.Tensor(part_shape, dtype),  # type: ignore
    ):
        with T.Kernel(
                T.ceildiv(seqlen_q, block_M), heads * batch, num_split,
                threads=128) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            mid = bx
            hid = by % heads
            bid = by // heads
            sid = bz

            T.annotate_layout({Q_shared: tl.layout.make_swizzled_layout(Q_shared)})
            T.copy(Q[bid, mid * block_M:(mid + 1) * block_M, hid, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # TODO: Handle causal split case
            loop_range = (
                T.min(T.ceildiv(seqlen_kv, block_N), T.ceildiv(
                    (mid + 1) * block_M, block_N)) if is_causal else T.ceildiv(
                        (seqlen_kv // num_split), block_N))

            for k in T.Pipelined(loop_range, num_stages=2):
                MMA0(K, Q_shared, K_shared, acc_s, k, mid, hid, bid, sid)
                Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum,
                        logsum)
                Rescale(acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, k, hid, bid, sid)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            for i in T.Parallel(block_M):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, glse[bid, hid, sid, mid * block_M:(mid + 1) * block_M])
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output_partial[bid, mid * block_M:(mid + 1) * block_M, hid, sid, :])

    @T.macro
    def combine(
            glse: T.Tensor([batch, heads, num_split, seqlen_q], dtype),  # type: ignore
            Output_partial: T.Tensor(part_shape, dtype),  # type: ignore
            Output: T.Tensor(shape_q, dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seqlen_q, block_M), heads, batch, threads=128) as (bx, by, bz):
            po_local = T.alloc_fragment([block_M, dim], dtype)
            po_shared = T.alloc_shared([block_M, dim], dtype)
            o_accum_local = T.alloc_fragment([block_M, dim], accum_dtype)
            o_shared = T.alloc_shared([block_M, dim], dtype)
            lse_local = T.alloc_fragment([num_split, block_M], dtype)
            lse_local_split = T.alloc_fragment([block_M], accum_dtype)
            lse_logsum_local = T.alloc_fragment([block_M], accum_dtype)
            lse_max_local = T.alloc_fragment([block_M], accum_dtype)
            scale_local = T.alloc_fragment([block_M], accum_dtype)

            T.annotate_layout({
                o_accum_local: T.Fragment(o_accum_local.shape, forward_thread_fn=lambda i, j: i),
                lse_local_split: T.Fragment(lse_local_split.shape, forward_thread_fn=lambda i: i),
                o_shared: tl.layout.make_swizzled_layout(o_shared),
                po_shared: tl.layout.make_swizzled_layout(po_shared),
            })

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            T.copy(glse[
                bz,
                by,
                :,
                bx * block_M:(bx + 1) * block_M,
            ], lse_local)
            T.reduce_max(lse_local, lse_max_local, dim=0, clear=False)
            for k in T.Pipelined(num_split):
                T.copy(lse_local[k, :], lse_local_split)
                for i in T.Parallel(block_M):
                    lse_logsum_local[i] += T.exp2(lse_local_split[i] - lse_max_local[i])
            for i in T.Parallel(block_M):
                lse_logsum_local[i] = T.log2(lse_logsum_local[i]) + lse_max_local[i]
            for k in T.Pipelined(num_split, num_stages=2):
                T.copy(Output_partial[bz, bx * block_M:(bx + 1) * block_M, by, k, :], po_shared)
                T.copy(po_shared, po_local)
                T.copy(lse_local[k, :], lse_local_split)
                for i in T.Parallel(block_M):
                    scale_local[i] = T.exp2(lse_local_split[i] - lse_logsum_local[i])
                for i, j in T.Parallel(block_M, dim):
                    o_accum_local[i, j] += po_local[i, j] * scale_local[i]
            T.copy(o_accum_local, o_shared)
            T.copy(o_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])

    @T.prim_func
    def main(
            Q: T.Tensor(shape_q, dtype),  # type: ignore
            K: T.Tensor(shape_kv, dtype),  # type: ignore
            V: T.Tensor(shape_kv, dtype),  # type: ignore
            glse: T.Tensor([batch, heads, num_split, seqlen_q], dtype),  # type: ignore
            # [batch, seqlen_q, heads, num_split, dim]
        Output_partial: T.Tensor(part_shape, dtype),  # type: ignore
            Output: T.Tensor(shape_q, dtype),  # type: ignore
    ):
        flash_attn_split(Q, K, V, glse, Output_partial)
        combine(glse, Output_partial, Output)

    return main


class _MHA_infer_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, glse, Output_partial, causal):
        BATCH, Q_CTX, H, D_HEAD = q.shape
        KV_CTX = k.shape[1]
        block_M = 128
        block_N = 64 if D_HEAD <= 128 else 32

        mod = _mha_infer(BATCH, H, Q_CTX, KV_CTX, D_HEAD, causal, block_M, block_N)
        return mod(q, k, v, glse, Output_partial)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass is not implemented for MLA_infer_attention.")


MHA_infer_attention = _MHA_infer_attention.apply


class MHAInferenceKernel(nn.Module):

    def __init__(self,
                 batch_size,
                 num_heads,
                 seqlen_q,
                 seqlen_kv,
                 head_dim,
                 causal,
                 dtype=torch.float16,
                 device="cuda"):
        super().__init__()
        self.attention = MHA_infer_attention
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.head_dim = head_dim
        self.causal = causal
        self.dtype = dtype
        self.device = device
        flops_per_matmul = 2.0 * batch_size * num_heads * seqlen_q * seqlen_kv * head_dim
        self.total_flops = 2 * flops_per_matmul
        if causal:
            self.total_flops *= 0.5

    def forward(self, Q, K, V, glse, Output_partial):  # Layout: BSHD
        return self.attention(Q, K, V, glse, Output_partial, self.causal)

    def ref_program(self, Q, K, V):
        assert self.causal is False
        dim = Q.size(-1)
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)
        scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)
        return output

    def gen_inputs(self):
        shape_q = self.batch_size, self.seqlen_q, self.num_heads, self.head_dim
        shape_kv = self.batch_size, self.seqlen_kv, self.num_heads, self.head_dim
        part_shape = self.batch_size, self.seqlen_q, self.num_heads, num_split, self.head_dim

        Q = torch.randn(shape_q, dtype=self.dtype, device=self.device)
        K = torch.randn(shape_kv, dtype=self.dtype, device=self.device)
        V = torch.randn(shape_kv, dtype=self.dtype, device=self.device)
        glse = torch.randn([self.batch_size, self.num_heads, num_split, self.seqlen_q],
                           dtype=self.dtype,
                           device=self.device)
        Output_partial = torch.randn(part_shape, dtype=self.dtype, device=self.device)
        return Q, K, V, glse, Output_partial

    def check(self):
        Q, K, V, glse, Output_partial = self.gen_inputs()
        o = self.forward(Q, K, V, glse, Output_partial)
        o_ref = self.ref_program(Q, K, V)
        assert torch.allclose(o, o_ref, rtol=1e-2, atol=1e-2), "o does not match reference"
        print("All checks passed! ✅")

    def profile(self, warmup=500):
        Q, K, V, glse, Output_partial = self.gen_inputs()
        with torch.no_grad():
            ref_latency = do_bench(
                lambda: self.ref_program(Q, K, V),
                warmup=warmup,
            )
            print(f'Reference Latency: {ref_latency:.2f} ms')
            print(f"Reference FLOPs: {self.total_flops / ref_latency * 1e-9:.2f} TFLOPs")

            latency = do_bench(
                lambda: self.forward(Q, K, V, glse, Output_partial),
                warmup=warmup,
            )
            print(f'Latency: {latency:.2f} ms')
            print(f"FLOPs: {self.total_flops / latency * 1e-9:.2f} TFLOPs")
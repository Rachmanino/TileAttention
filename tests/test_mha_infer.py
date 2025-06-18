import argparse
from tla import MHAInferenceKernel


def test_mha_inferece_kernel(BATCH, H, Q_CTX, KV_CTX, D_HEAD, causal=False):
    kernel = MHAInferenceKernel(
        batch_size=BATCH,
        num_heads=H,
        seqlen_q=Q_CTX,
        seqlen_kv=KV_CTX,
        head_dim=D_HEAD,
        causal=causal)
    kernel.check()
    kernel.profile(warmup=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--seqlen_q', type=int, default=128, help='sequence length of query')
    parser.add_argument('--seqlen_kv', type=int, default=8192, help='sequence length of key/value')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    args = parser.parse_args()
    BATCH, H, Q_CTX, KV_CTX, D_HEAD = args.batch, args.heads, args.seqlen_q, args.seqlen_kv, args.dim

    test_mha_inferece_kernel(BATCH, H, Q_CTX, KV_CTX, D_HEAD, causal=False)
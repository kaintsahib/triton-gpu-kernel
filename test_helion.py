import torch
import helion
import helion.language as hl

# ऑटोट्यूनर द्वारा खोजी गई बेस्ट गोल्डन कॉन्फ़िगरेशन को यहाँ पास किया गया है
@helion.kernel(
    config=helion.Config(
        atomic_indexing=[], 
        block_sizes=[256], 
        indexing=['block_ptr', 'pointer', 'pointer'], 
        load_eviction_policies=['', 'first'], 
        num_stages=6, 
        num_warps=1, 
        pid_type='flat', 
        range_flattens=[None], 
        range_multi_buffers=[None], 
        range_num_stages=[0], 
        range_unroll_factors=[0], 
        range_warp_specializes=[]
    ), 
    static_shapes=True
)
def add_kernel(a: torch.Tensor, b: torch.Tensor):
    out = torch.empty_like(a)

    for tile in hl.tile(a.numel()):
        out[tile] = a[tile] + b[tile]

    return out

def main():
    assert torch.cuda.is_available()

    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")

    result = add_kernel(a, b)
    torch.testing.assert_close(result, a + b)

    print("Helion GPU kernel passed with optimal autotuned config!")
    print("GPU:", torch.cuda.get_device_name(0))
    print("First 5 values:", result[:5])

if __name__ == "__main__":
    main()

# Check GPU health

```shell
for i in 0 1 2 3 4 5 6 7; do                                                                                                                                                                
    echo "=== GPU $i ==="                                                                                                                                                 
    CUDA_VISIBLE_DEVICES=$i python -c "import torch; x = torch.randn(1024, 1024, device='cuda'); print('OK', x.sum().item())" 2>&1 | tail -1
done

for i in 0 1 2 3; do                                                                                                                                                                
    echo "=== GPU $i ==="                                                                                                                                                 
    CUDA_VISIBLE_DEVICES=$i python -c "import torch; x = torch.randn(1024, 1024, device='cuda'); print('OK', x.sum().item())" 2>&1 | tail -1
done                      
```
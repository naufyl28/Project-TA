import time
import numpy as np
import torch

def evaluate_prediction_latency(trainer, dataset, repeat_times=10):
    latencies = []

    device = next(trainer.model.parameters()).device

    for _ in range(repeat_times):
        # Ambil 1 data random
        idx = np.random.randint(0, len(dataset))
        single_input = dataset[idx]
        inputs = {k: v.unsqueeze(0).to(device) for k, v in single_input.items() if k != 'labels'}
        
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = trainer.model(**inputs)
        end_time = time.perf_counter()
        
        latencies.append(end_time - start_time)
    
    avg_latency = np.mean(latencies)
    print(f"Average latency over {repeat_times} runs: {avg_latency:.6f} seconds")
    return avg_latency
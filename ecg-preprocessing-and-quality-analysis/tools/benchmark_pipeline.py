import time
import cProfile
import pstats
import logging
from pathlib import Path
from scripts.process_daily_ecg import process_one_file, PipelineConfig

def main():
    print("Benchmarking Process_Daily_ECG")
    data_dir = Path("DATA/RAW/NOISE_STRESS_TEST")
    output_base = Path("Processed_ECG/NST_BENCHMARK")
    
    # Clean up output dir if exists to ensure clean benchmark
    if output_base.exists():
        import shutil
        shutil.rmtree(output_base)
        
    files = list(data_dir.glob("*.csv"))[:5] # Process 5 files
    
    cfg = PipelineConfig(target_fs_hz=125.0, segment_seconds=5.0)
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.CRITICAL) # Silence logs for benchmark
    
    print(f"Processing {len(files)} files...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    
    results = []
    for f in files:
        res = process_one_file(
            f,
            input_folder=data_dir,
            output_base=output_base,
            cfg=cfg,
            logger=logger
        )
        results.append(res)
        
    end_time = time.time()
    profiler.disable()
    
    print(f"Total processing time: {end_time - start_time:.4f} seconds")
    
    # Print the top 20 functions by cumulative time
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(25)

if __name__ == "__main__":
    main()

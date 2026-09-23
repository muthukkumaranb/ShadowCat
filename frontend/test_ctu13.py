import pandas as pd
from data_provider import run_core_ml_inference

def main():
    print("Loading CTU-13 dataset...")
    df = pd.read_csv("../scratch/CTU13_Attack_Traffic.csv")
    print(f"Loaded {len(df)} rows.")
    print("Running core ML inference...")
    try:
        result = run_core_ml_inference(df, source_type="csv")
        print("Inference completed.")
        if "_inference_error" in result:
            print("ERROR CAUGHT BY data_provider:")
            print(result["_inference_error"])
            print("TRACEBACK:")
            print(result.get("_inference_traceback", ""))
        else:
            print("SUCCESS")
    except Exception as e:
        print("UNCAUGHT EXCEPTION during inference:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

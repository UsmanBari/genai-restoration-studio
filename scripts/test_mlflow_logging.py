"""
Test MLflow logging and verification.
Logs parameters, metrics, tags, and an artifact, then verifies them using MlflowClient.
"""

import os
import sys
import tempfile
import mlflow
from mlflow.tracking import MlflowClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_mlflow_verification():
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    
    tracking_dir = os.path.abspath("mlruns")
    os.makedirs(tracking_dir, exist_ok=True)
    db_path = os.path.join(tracking_dir, "mlflow.db").replace('\\', '/')
    mlflow_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(mlflow_uri)

    exp_name = "Task1-Universal-Restoration"
    mlflow.set_experiment(exp_name)
    client = MlflowClient(tracking_uri=mlflow_uri)

    print(f"MLflow Tracking URI: {mlflow_uri}")
    print(f"Experiment Name:     {exp_name}")

    # Create dummy artifact file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Sample model config: 128x128 RGB, Autoencoder bottleneck 64")
        temp_artifact_path = f.name

    try:
        with mlflow.start_run(run_name="milestone_1_verification_run") as run:
            run_id = run.info.run_id
            print(f"Started Run ID:      {run_id}")

            # Log parameters
            mlflow.log_param("architecture", "UniversalAutoencoder")
            mlflow.log_param("optimizer", "Adam")
            mlflow.log_param("learning_rate", 0.0002)
            mlflow.log_param("batch_size", 32)

            # Log metrics across steps
            for step in range(5):
                train_loss = 0.50 / (step + 1)
                val_loss = 0.55 / (step + 1)
                val_psnr = 20.0 + step * 2.5
                val_ssim = 0.70 + step * 0.05
                mlflow.log_metric("train_loss", train_loss, step=step)
                mlflow.log_metric("val_loss", val_loss, step=step)
                mlflow.log_metric("val_psnr", val_psnr, step=step)
                mlflow.log_metric("val_ssim", val_ssim, step=step)

            # Log tags and artifact
            mlflow.set_tag("milestone", "1")
            mlflow.set_tag("status", "verified")
            mlflow.log_artifact(temp_artifact_path, artifact_path="configs")

        # Verify using MlflowClient
        logged_run = client.get_run(run_id)
        print("\n=== MLflow Run Verification Output ===")
        print(f"Run ID:        {logged_run.info.run_id}")
        print(f"Run Name:      {logged_run.info.run_name}")
        print(f"Status:        {logged_run.info.status}")
        print(f"Parameters:    {logged_run.data.params}")
        print(f"Metrics (end): {logged_run.data.metrics}")
        print(f"Tags:          {logged_run.data.tags.get('milestone')}")
        
        artifacts = client.list_artifacts(run_id, path="configs")
        print(f"Artifacts:     {[a.path for a in artifacts]}")
        
        assert logged_run.info.status == "FINISHED"
        assert "learning_rate" in logged_run.data.params
        assert "val_psnr" in logged_run.data.metrics
        assert len(artifacts) > 0

        print("\n[OK] MLflow logging and client verification confirmed!")
        return True
    finally:
        if os.path.exists(temp_artifact_path):
            os.remove(temp_artifact_path)


if __name__ == '__main__':
    run_mlflow_verification()

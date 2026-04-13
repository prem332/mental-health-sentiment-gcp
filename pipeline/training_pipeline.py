"""
pipeline/training_pipeline.py
Vertex AI Kubeflow pipeline: Train → Evaluate → Gate → Register
Usage:
    python pipeline/training_pipeline.py \
        --project mental-health-sentiment-gcp \
        --bucket  mh-sentiment-models \
        --region  us-central1 \
        --image   us-central1-docker.pkg.dev/mental-health-sentiment-gcp/mental-health-sentiment-repo/mental-health-sentiment:latest
"""
import argparse, logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ACCURACY_THRESHOLD = 0.95
F1_THRESHOLD       = 0.93

def build_pipeline(project, bucket, region, image, epochs):
    from kfp import dsl
    from kfp.dsl import component, Output, Artifact, Metrics

    @component(base_image=image,
               packages_to_install=["google-cloud-storage","tensorflow==2.15.1","scikit-learn","pandas","numpy"])
    def train_component(bucket: str, epochs: int, metrics_output: Output[Artifact]):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "train_model.py", "--bucket", bucket,
             "--epochs", str(epochs), "--metrics-output", metrics_output.path],
            capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"Training failed:\n{result.stderr}")

    @component(base_image="python:3.10-slim", packages_to_install=["google-cloud-storage"])
    def evaluate_gate_component(bucket: str, accuracy_threshold: float, f1_threshold: float,
                                 metrics_input: "Input[Artifact]", gate_passed: Output[Artifact], eval_metrics: Output[Metrics]):
        import json, os
        with open(metrics_input.path) as f: metrics = json.load(f)
        accuracy, f1_macro, version = metrics["accuracy"], metrics["f1_macro"], metrics["model_version"]
        eval_metrics.log_metric("accuracy", accuracy)
        eval_metrics.log_metric("f1_macro", f1_macro)
        for cls, f1 in metrics["f1_per_class"].items():
            eval_metrics.log_metric(f"f1_{cls.lower()}", f1)
        passed = (accuracy >= accuracy_threshold) and (f1_macro >= f1_threshold)
        print(f"📊 Accuracy={accuracy:.4f} (≥{accuracy_threshold}) | F1={f1_macro:.4f} (≥{f1_threshold})")
        print(f"{'✅ Gate PASSED' if passed else '❌ Gate FAILED'} — version {version}")
        os.makedirs(os.path.dirname(gate_passed.path), exist_ok=True)
        with open(gate_passed.path, "w") as f:
            json.dump({"passed": passed, "accuracy": accuracy, "f1_macro": f1_macro, "version": version, "metrics": metrics}, f, indent=2)

    @component(base_image="python:3.10-slim", packages_to_install=["google-cloud-aiplatform","google-cloud-storage"])
    def register_model_component(project: str, region: str, bucket: str,
                                  gate_result: "Input[Artifact]", registered_model: Output[Artifact]):
        import json, os
        from google.cloud import aiplatform
        with open(gate_result.path) as f: result = json.load(f)
        if not result["passed"]:
            raise RuntimeError(f"Gate failed: accuracy={result['accuracy']:.4f}, f1={result['f1_macro']:.4f}")
        version, metrics = result["version"], result["metrics"]
        aiplatform.init(project=project, location=region)
        artifact_uri = f"gs://{bucket}/models/{version}/"
        model = aiplatform.Model.upload(
            display_name=f"bilstm-sentiment-{version}",
            artifact_uri=artifact_uri,
            serving_container_image_uri=f"us-central1-docker.pkg.dev/{project}/mental-health-sentiment-repo/mental-health-sentiment:latest",
            labels={"accuracy": str(int(metrics["accuracy"]*10000)), "f1_macro": str(int(metrics["f1_macro"]*10000))},
            description=f"BiLSTM | acc={metrics['accuracy']:.4f} | f1={metrics['f1_macro']:.4f}",
        )
        os.makedirs(os.path.dirname(registered_model.path), exist_ok=True)
        with open(registered_model.path, "w") as f:
            json.dump({"resource_name": model.resource_name, "version": version,
                       "accuracy": metrics["accuracy"], "f1_macro": metrics["f1_macro"]}, f, indent=2)
        print(f"✅ Registered: {model.resource_name}")

    @dsl.pipeline(name="mental-health-sentiment-training",
                  description="Train → Evaluate → Gate → Register BiLSTM")
    def sentiment_training_pipeline(bucket: str = bucket, epochs: int = epochs,
                                     accuracy_threshold: float = ACCURACY_THRESHOLD, f1_threshold: float = F1_THRESHOLD):
        train_task  = train_component(bucket=bucket, epochs=epochs).set_display_name("Train BiLSTM").set_cpu_limit("4").set_memory_limit("16G")
        gate_task   = evaluate_gate_component(bucket=bucket, accuracy_threshold=accuracy_threshold, f1_threshold=f1_threshold,
                                               metrics_input=train_task.outputs["metrics_output"]).set_display_name("Evaluate + Gate").after(train_task)
        register_task = register_model_component(project=project, region=region, bucket=bucket,
                                                   gate_result=gate_task.outputs["gate_passed"]).set_display_name("Register in Vertex AI").after(gate_task)

    return sentiment_training_pipeline

def compile_pipeline(pipeline_func, output_path="pipeline.yaml"):
    from kfp import compiler
    compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=output_path)
    logger.info("✅ Pipeline compiled → %s", output_path)
    return output_path

def submit_pipeline(project, region, pipeline_yaml, bucket):
    from google.cloud import aiplatform
    from datetime import datetime
    job_name = f"mh-sentiment-train-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    aiplatform.init(project=project, location=region)
    job = aiplatform.PipelineJob(display_name=job_name, template_path=pipeline_yaml,
                                  pipeline_root=f"gs://{bucket}/pipeline_root/",
                                  parameter_values={"bucket": bucket}, enable_caching=False)
    job.submit()
    logger.info("🚀 Pipeline submitted → https://console.cloud.google.com/vertex-ai/pipelines?project=%s", project)
    return job

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--bucket",  required=True)
    parser.add_argument("--region",  default="us-central1")
    parser.add_argument("--image",   required=True)
    parser.add_argument("--epochs",  type=int, default=10)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    pipeline_func = build_pipeline(args.project, args.bucket, args.region, args.image, args.epochs)
    yaml_path = compile_pipeline(pipeline_func)
    if not args.compile_only:
        submit_pipeline(args.project, args.region, yaml_path, args.bucket)
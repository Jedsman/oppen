# Phase 8: Production ML Observability

## Objective
Monitor model performance in production, detect data drift and model degradation, and enable autonomous root-cause analysis. Move beyond infrastructure metrics to ML-specific observability.

## 1. Model Performance Metrics
Track model behavior on production data.

### Steps:
1. **Inference Logging**:
   - For each prediction, log:
     - Input features (or hash for privacy).
     - Model version.
     - Prediction output.
     - Confidence score (if applicable).
     - Actual label (ground truth, once available).
     - Latency, resource usage.
   - Store in time-series DB or data warehouse (ClickHouse, Parquet files).

2. **Custom Prometheus Metrics**:
   - Export model-specific metrics:
     ```python
     model_accuracy = Counter('model_predictions_correct_total', 'Correct predictions')
     model_latency = Histogram('model_inference_latency_ms', 'Inference latency')
     model_confidence = Histogram('model_prediction_confidence', 'Prediction confidence')
     model_errors = Counter('model_prediction_errors_total', 'Error predictions', ['error_type'])
     ```
   - Label by model name, version, environment.

3. **Grafana Dashboards**:
   - **Real-time Performance**: Accuracy, latency, throughput (sliding 24h window).
   - **Model Comparison**: v1 vs v2 vs v3 accuracy on same test set.
   - **Version Rollout**: Gradual traffic shift to new model (e.g., 10% → 50% → 100%).

## 2. Data Drift Detection
Identify when input data distribution changes (model inputs no longer match training data).

### Steps:
1. **Baseline Statistics**:
   - Calculate training data statistics:
     - Per-feature: mean, std, min, max, percentiles.
     - Categorical features: value distribution.
   - Store as reference in MLflow model metadata.

2. **Production Data Distribution**:
   - Periodically (hourly/daily) compute statistics on production data.
   - Compare to baseline using statistical tests.

3. **Drift Detection Methods** (choose one or combine):

   **Method A: Kolmogorov-Smirnov (KS) Test**:
   - Compares distributions of continuous features.
   - Alert if KS statistic > threshold.

   **Method B: Chi-Square Test**:
   - For categorical features.

   **Method C: Wasserstein Distance**:
   - Measures dissimilarity between distributions.
   - More interpretable than KS test.

   **Method D: Evidently AI Integration**:
   - Use Evidently library for comprehensive drift detection:
   ```python
   from evidently.metric_preset import DataDriftPreset
   from evidently.report import Report

   report = Report(metrics=[DataDriftPreset()])
   report.run(reference_data=training_df, current_data=prod_df)
   ```

4. **Drift Alerting**:
   - Alert if drift detected on critical features.
   - Log to Prometheus and trigger investigation.

## 3. Model Degradation Detection
Identify when model performance drops on production data.

### Steps:
1. **Actual vs Predicted Tracking**:
   - Inference logs contain predicted labels; once ground truth arrives (delayed), compare.
   - Calculate per-period accuracy: `accuracy = (correct_predictions / total_predictions)`.

2. **Performance Thresholds**:
   - Set SLO (Service Level Objective): e.g., "accuracy ≥ 95%".
   - Alert if accuracy drops below SLO.
   - Example Prometheus alert:
     ```yaml
     alert: ModelAccuracyDrop
     expr: model_accuracy < 0.95
     for: 1h
     ```

3. **Performance Comparison**:
   - Track accuracy per model version.
   - Alert if new version underperforms baseline:
     - "v4 accuracy (93%) < v3 baseline (96%), degradation -3%".

4. **Latency & Resource Monitoring**:
   - Track p50, p95, p99 latencies.
   - Alert if latency SLO violated.
   - Correlate with GPU/CPU utilization.

## 4. Root-Cause Analysis Framework
Structure diagnosis of model issues.

### Categories:
1. **Data Issues**:
   - Input drift (distribution shift).
   - Missing values or outliers increasing.
   - New feature values not seen during training.

2. **Model Issues**:
   - Model version bug or incorrect deployment.
   - Weights corrupted or incompatible with new input schema.

3. **Infrastructure Issues**:
   - GPU OOM causing silent failures.
   - Network latency causing timeouts.
   - Resource contention causing degradation.

### Diagnostic Workflow:
1. **Detect**: Alert triggered (accuracy drop, drift, latency spike).
2. **Investigate**:
   - Is it data drift? Check feature distributions.
   - Is it model degradation? Compare version accuracy.
   - Is it infrastructure? Check logs, GPU metrics.
3. **Isolate**: Pin down the root cause.
4. **Remediate**: Trigger fix (retrain, rollback, scale resources).

## 5. Agent Integration (LangGraph)
Extend the agent to diagnose and resolve ML issues autonomously.

### New MCP Server: ML Observability
Expose tools:
- `get_model_accuracy(model_name, version, time_window)`: Latest accuracy.
- `detect_data_drift(model_name, threshold)`: Current drift status.
- `query_inference_logs(model_name, version, filters)`: Sample predictions + errors.
- `compare_model_versions(model_a, model_b)`: Performance comparison.
- `get_feature_distribution(feature_name, time_window)`: Statistical summary.

### Agent Nodes (LangGraph):
1. **Monitor Node**:
   - Continuously polls observability data.
   - Aggregates alerts: "Model accuracy dropped to 92%".

2. **Investigate Node**:
   - Queries inference logs: "What patterns in recent errors?".
   - Checks drift: "Feature X distribution changed by 25%".
   - Compares versions: "v3 has 96% accuracy, v4 has 92%".

3. **Reasoning Node**:
   - Ollama analyzes collected data.
   - Hypothesis: "Accuracy drop is due to input drift (Feature X), not model bug".
   - Evidence: "Feature X mean shifted from 50 → 75 (40% increase)".

4. **Action Node**:
   - Proposes remediation:
     - "Rollback to v3 (immediate, restores accuracy)".
     - "Retrain with recent data (long-term, fixes model)".
   - Executes (with human approval).

5. **Verification Node**:
   - Post-action: "Rolled back to v3, accuracy now 96%".

## 6. Advanced: Model Retraining Triggers
Automate retraining when performance degrades.

### Steps:
1. **Retrain Trigger Conditions**:
   - Accuracy drops below threshold (manual trigger).
   - Drift detected on critical features (automatic).
   - Scheduled (weekly, monthly).
   - On-demand (agent proposal).

2. **Retraining Pipeline** (integrate with Phase 5):
   - Fetch fresh production data.
   - Create new dataset (training + recent production).
   - Launch training job (K8s Job with GPU).
   - Evaluate new model against old model.
   - Register in MLflow Model Registry.
   - Promote to Staging for validation.

3. **Automated Promotion**:
   - If new model > baseline accuracy, automatically promote to Production.
   - Or require human approval (depends on use case criticality).

## 7. Simulation Run: Model Degradation Scenario
Test observability and remediation workflow.

### Scenario:
1. **Baseline**: Deploy model v1, monitor for 1 hour.
   - Accuracy: 96%, latency p95: 50ms.
2. **Inject Drift**: Introduce out-of-distribution data (e.g., change feature scale).
   - Accuracy drops to 88%, drift detected.
3. **Agent Investigation**:
   - Agent detects accuracy drop.
   - Queries drift detection: "Feature X distribution changed 30%".
   - Proposes: "Accuracy drop caused by input drift, not model bug".
4. **Remediation Options**:
   - Option A: Rollback to v0 (temporary, restores accuracy to 96%).
   - Option B: Retrain model v2 with recent data (long-term).
5. **Execution & Verification**:
   - Agent executes rollback (or triggers retraining).
   - Post-action accuracy back to 96%.

## 8. Observability Stack Components
Summary of tools and integrations.

### Components:
- **Inference Logging**: Custom sidecar logging predictions to ClickHouse or file storage.
- **Drift Detection**: Evidently AI or custom statistical checks in Prometheus.
- **Performance Tracking**: Custom Prometheus metrics exported by serving framework.
- **Alerting**: Prometheus AlertManager for drift/degradation alerts.
- **Visualization**: Grafana dashboards per model showing accuracy, drift, latency.
- **Root-Cause Analysis**: Agent + MCP tools querying logs and metrics.

## Deliverables
- Inference logging system (predictions, features, labels, metadata).
- Custom Prometheus metrics for model performance (accuracy, confidence, latency).
- Data drift detection using statistical methods (KS test, Wasserstein, or Evidently).
- Model degradation alerting (accuracy SLO breaches).
- Grafana dashboards: Model performance, data drift, version comparison.
- ML Observability MCP Server with tools for diagnosis.
- Agent nodes for autonomous root-cause analysis.
- Automated retraining trigger integration with Phase 5 training pipeline.
- Simulation: Inject drift, detect, diagnose, and remediate autonomously.

## Learning Outcomes
- ML-specific observability beyond infrastructure metrics.
- Data drift and model degradation detection and diagnosis.
- Root-cause analysis frameworks for production ML systems.
- Integrating inference logging with monitoring and alerting.
- Autonomous remediation for common ML failure modes.
- Designing ML SLOs and observability for production systems.

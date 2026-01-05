# Lab 1.2: Create Your First Cluster

**Estimated Time**: 45 minutes
**Difficulty**: Medium
**Goal**: Create a Kind cluster and understand its structure

## Before You Start

✅ Verify you completed Lab 1.1:
- [ ] `kind version` works
- [ ] `kubectl version --client` works
- [ ] `docker ps` works (Docker running)

## Part 1: Create a Kind Cluster

A Kind cluster runs Kubernetes inside Docker containers. You'll create a cluster called "oppen-lab".

### Simple Creation (Recommended for Learning)

```bash
kind create cluster --name oppen-lab
```

**What happens:**
1. Kind downloads a Kubernetes image (⏱️ ~2-3 minutes on first run)
2. Starts a Docker container running Kubernetes control plane
3. Sets up networking
4. Configures kubectl to talk to this cluster

**Expected output:**
```
Creating cluster "oppen-lab" ...
 ✓ Ensuring node image (kindest/node:v1.28.0) 🖼
 ✓ Preparing nodes 📦
 ✓ Writing configuration 📝
 ✓ Starting control-plane 🕹️
 ✓ Installing CNI 🔌
 ✓ Installing StorageClass 💾
Set kubectl context to "kind-oppen-lab"
```

**⏱️ Wait for the process to complete** (this takes 2-3 minutes on first run, 30 seconds on subsequent runs).

### Advanced Creation (Multi-Node Cluster)

If you want to simulate multiple machines, create a config file:

```bash
cat > kind-config.yaml <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: oppen-lab
nodes:
- role: control-plane
- role: worker
- role: worker
EOF

kind create cluster --config kind-config.yaml
```

This creates 1 control plane + 2 worker nodes (3 Docker containers total). **This is optional** - single-node cluster is fine for learning.

**⏰ Estimated wait time:**
- First run: 2-3 minutes
- Subsequent runs: 30 seconds

## Part 2: Verify Cluster is Running

### Check Cluster Exists

```bash
kind get clusters
```

**Expected output:**
```
oppen-lab
```

✅ Your cluster exists!

### Check Nodes are Ready

```bash
kubectl get nodes
```

**Expected output (single-node):**
```
NAME                     STATUS   ROLES           AGE   VERSION
oppen-lab-control-plane  Ready    control-plane   2m    v1.28.0
```

**Expected output (multi-node):**
```
NAME                      STATUS   ROLES           AGE   VERSION
oppen-lab-control-plane   Ready    control-plane   2m    v1.28.0
oppen-lab-worker          Ready    <none>          2m    v1.28.0
oppen-lab-worker2         Ready    <none>          2m    v1.28.0
```

✅ If STATUS = Ready, your cluster is healthy!

### Check System Pods

```bash
kubectl get pods -n kube-system
```

**Expected output:**
```
NAME                                           READY   STATUS    RESTARTS   AGE
coredns-5d78c0869f-m8qzk                      1/1     Running   0          2m
etcd-oppen-lab-control-plane                  1/1     Running   0          2m
kube-apiserver-oppen-lab-control-plane        1/1     Running   0          2m
kube-controller-manager-oppen-lab-control-plane 1/1   Running   0          2m
kube-proxy-kmcvb                              1/1     Running   0          2m
kube-scheduler-oppen-lab-control-plane        1/1     Running   0          2m
kindnet-vnjdf                                 1/1     Running   0          2m
```

✅ If all pods are Running, Kubernetes is operational!

## Part 3: Understand Cluster Structure

Your cluster has this structure:

```
oppen-lab (Kubernetes Cluster)
│
├─ kube-system (System namespace)
│  ├─ kube-apiserver (Control Plane)
│  ├─ kube-controller-manager (Control Plane)
│  ├─ kube-scheduler (Control Plane)
│  ├─ etcd (Control Plane database)
│  ├─ coredns (DNS service)
│  ├─ kube-proxy (Network routing)
│  └─ kindnet (networking plugin)
│
├─ default (User namespace - empty)
│
└─ kube-public (Public namespace - empty)
```

### Explore Namespaces

```bash
kubectl get namespaces
```

**Expected output:**
```
NAME              STATUS   AGE
default           Active   3m
kube-node-lease   Active   3m
kube-public       Active   3m
kube-system       Active   3m
```

Each namespace is like a virtual cluster within your cluster.

### View All Resources Across All Namespaces

```bash
kubectl get all -A
```

This shows pods, services, deployments, etc. across all namespaces.

## Part 4: Inside the Docker Container

Your Kind cluster is literally running inside a Docker container. You can inspect it:

```bash
docker ps
```

**Expected output (single-node):**
```
CONTAINER ID   IMAGE                  COMMAND                STATUS          PORTS                       NAMES
a1b2c3d4e5f6   kindest/node:v1.28.0   "/usr/local/bin/entrypoint.sh"   Up 3 minutes   127.0.0.1:XXXXX->6443/tcp   oppen-lab-control-plane
```

**Expected output (multi-node):**
```
CONTAINER ID   IMAGE                  COMMAND   STATUS      PORTS      NAMES
a1b2c3d4   kindest/node:v1.28.0   "/usr/local/bin..."   Up 3m   127.0.0.1:XXXXX->6443/tcp   oppen-lab-control-plane
b2c3d4e5   kindest/node:v1.28.0   "/usr/local/bin..."   Up 3m                               oppen-lab-worker
c3d4e5f6   kindest/node:v1.28.0   "/usr/local/bin..."   Up 3m                               oppen-lab-worker2
```

Each line is a Kubernetes node running in Docker! 🐳

## Part 5: Access the Kubernetes Dashboard (Optional)

Kubernetes includes a web UI dashboard. Deploy it:

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml
```

**Wait for dashboard pod to start:**

```bash
kubectl get pods -n kubernetes-dashboard -w
```

Press Ctrl+C when you see `dashboard` pod with STATUS = Running.

**Create a proxy to access it:**

```bash
kubectl proxy
```

**Open browser:**
```
http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/
```

This opens the Kubernetes Dashboard UI where you can visualize your cluster.

**To stop proxy:** Press Ctrl+C in the terminal.

## Part 6: Check Cluster Resources

See how much CPU/memory your cluster can use:

```bash
kubectl top nodes
```

If this fails (common in Kind), don't worry - metrics server takes time to initialize.

### Get Cluster Info

```bash
kubectl cluster-info
```

**Expected output:**
```
Kubernetes control plane is running at https://127.0.0.1:XXXXX
CoreDNS is running at https://127.0.0.1:XXXXX/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy

To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.
```

## Part 7: Context Management

Your kubectl is now configured to talk to the "oppen-lab" cluster.

### View Current Context

```bash
kubectl config current-context
```

**Expected output:**
```
kind-oppen-lab
```

### View All Contexts

```bash
kubectl config get-contexts
```

This shows all Kubernetes clusters your kubectl can talk to. If you had multiple clusters, you could switch between them.

### Switch Context (if you had multiple clusters)

```bash
# Switch to a different cluster
kubectl config use-context kind-other-cluster

# Switch back to oppen-lab
kubectl config use-context kind-oppen-lab
```

## Validation Checkpoint ✅

Complete all of these checks:

- [ ] `kind get clusters` shows "oppen-lab"
- [ ] `kubectl get nodes` shows STATUS = Ready
- [ ] `kubectl get pods -n kube-system` shows all Running pods
- [ ] `docker ps` shows 1+ oppen-lab containers
- [ ] `kubectl config current-context` shows "kind-oppen-lab"
- [ ] `kubectl cluster-info` succeeds

**If all boxes are checked**: Proceed to Lab 1.3 ✅

**If any fail**: See troubleshooting section below.

## Troubleshooting

### Cluster Creation Hangs

**Problem**: `kind create cluster` doesn't complete after 5 minutes.

**Solution:**
1. Press Ctrl+C to stop it
2. Check if container started: `docker ps -a`
3. Check Docker logs: `docker logs oppen-lab-control-plane`
4. If stuck, clean up and retry:
   ```bash
   kind delete cluster --name oppen-lab
   kind create cluster --name oppen-lab
   ```

### "Unable to connect to cluster"

**Problem**: `kubectl get nodes` shows "Unable to connect to Kubernetes API"

**Solution:**
1. Check cluster is still running: `docker ps | grep oppen-lab`
2. Check context: `kubectl config current-context` (should be kind-oppen-lab)
3. If not, switch context: `kubectl config use-context kind-oppen-lab`
4. If Docker container died, recreate cluster:
   ```bash
   kind delete cluster --name oppen-lab
   kind create cluster --name oppen-lab
   ```

### "Insufficient CPU/Memory"

**Problem**: Cluster crashes with "OOMKilled" or similar.

**Solution:**
- Kind cluster needs ~2GB RAM
- Free up memory on your machine
- Or resize Kind cluster in config (advanced)

### "Docker: permission denied"

**Problem**: `docker ps` or `kind create` fails with permission error (Linux).

**Solution:**
```bash
sudo usermod -aG docker $USER
# Log out and back in
docker ps  # Should work now
```

### Delete and Recreate Cluster

If something is broken, you can always start fresh:

```bash
# Delete the cluster
kind delete cluster --name oppen-lab

# Verify it's gone
kind get clusters  # Should be empty

# Create fresh cluster
kind create cluster --name oppen-lab
```

This takes ~30 seconds and gives you a clean cluster.

## How Kind Works (Behind the Scenes)

Kind uses Docker to simulate a Kubernetes cluster:

```
┌─────────────────────────────────┐
│         Docker Engine           │
├─────────────────────────────────┤
│                                 │
│  ┌────────────────────────────┐ │
│  │ Container: oppen-lab-cp    │ │
│  │ (Kubernetes control plane) │ │
│  │ - API server               │ │
│  │ - Scheduler                │ │
│  │ - Controller Manager       │ │
│  │ - etcd                     │ │
│  └────────────────────────────┘ │
│                                 │
│  ┌────────────────────────────┐ │
│  │ Container: oppen-lab-w1    │ │
│  │ (Kubernetes worker node)   │ │
│  │ - kubelet                  │ │
│  │ - Container runtime        │ │
│  └────────────────────────────┘ │
│                                 │
└─────────────────────────────────┘
```

Each Docker container simulates a Kubernetes node.

## Next Steps

Your cluster is now ready! In Lab 1.3, you'll:
1. Deploy nginx to the cluster
2. Create a Service to access it
3. Access it from your browser

→ **Next**: [Lab 1.3: Deploy Your First App](./04-lab3-app.md)

## Quick Reference: Useful Commands

```bash
# Cluster management
kind create cluster --name oppen-lab
kind get clusters
kind delete cluster --name oppen-lab

# View resources
kubectl get nodes
kubectl get pods -n kube-system
kubectl get namespaces
kubectl get all -A

# Cluster info
kubectl cluster-info
kubectl config current-context
kubectl describe node <node-name>

# Cleanup
kind delete cluster --name oppen-lab
docker system prune -a  # Warning: deletes all Docker containers/images
```

---

**Completed Lab 1.2?** → Move on to Lab 1.3 and deploy your first app! 🚀

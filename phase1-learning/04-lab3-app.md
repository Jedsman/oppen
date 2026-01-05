# Lab 1.3: Deploy Your First App

**Estimated Time**: 1 hour
**Difficulty**: Medium
**Goal**: Deploy nginx and access it from your browser

## Before You Start

✅ Verify you completed Lab 1.2:
- [ ] Cluster running: `kind get clusters` shows "oppen-lab"
- [ ] Nodes ready: `kubectl get nodes` shows STATUS = Ready
- [ ] `kubectl get pods -n kube-system` shows all pods Running

## Part 1: Create a Deployment (Imperative)

The simplest way to create a deployment is with the `kubectl create` command:

```bash
kubectl create deployment nginx --image=nginx:alpine
```

**What this does:**
- Creates a Deployment resource named "nginx"
- Uses the `nginx:alpine` image (lightweight, ~40MB)
- By default, runs 1 replica (1 container)

**Expected output:**
```
deployment.apps/nginx created
```

### Verify Deployment Created

```bash
kubectl get deployments
```

**Expected output:**
```
NAME    READY   UP-TO-DATE   AVAILABLE   AGE
nginx   1/1     1            1           30s
```

✅ If READY = 1/1, your deployment is working!

### View Pods

Deployments create Pods automatically:

```bash
kubectl get pods
```

**Expected output:**
```
NAME                     READY   STATUS    RESTARTS   AGE
nginx-748c667d99-abc12   1/1     Running   0          45s
```

✅ Pod is running!

### View Deployment Details

```bash
kubectl describe deployment nginx
```

**Expected output includes:**
```
Name:                   nginx
Namespace:              default
Selector:               app=nginx
Replicas:               1 desired | 1 updated | 1 ready
Pods Status:            1 Running
...
```

## Part 2: Create a Service (How to Access Your App)

Your Pod is running but it's not accessible from outside the cluster. Create a Service to expose it:

```bash
kubectl expose deployment nginx --port=80 --type=NodePort
```

**What this does:**
- Creates a Service named "nginx" (same name as deployment)
- Maps port 80 (nginx's port) to a random NodePort (typically 30000-32767)
- Type=NodePort means "accessible on each node's IP"

**Expected output:**
```
service/nginx exposed
```

### Find the NodePort

```bash
kubectl get services
```

**Expected output:**
```
NAME         TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)        AGE
kubernetes   ClusterIP   10.96.0.1      <none>        443/TCP        30m
nginx        NodePort    10.96.123.45   <none>        80:31234/TCP   10s
```

Look for the nginx service. The port mapping is `80:31234/TCP`:
- `80` = nginx listens on this port inside container
- `31234` = NodePort (external, random each time)

📝 **Write down your NodePort** (in this example: 31234)

## Part 3: Access Your App

There are two ways to access your nginx app.

### Method 1: Port Forward (Recommended for Learning)

Port forward creates a tunnel from your machine to the cluster:

```bash
kubectl port-forward svc/nginx 8080:80
```

**What this does:**
- `svc/nginx` = route to nginx Service
- `8080` = your machine's port
- `:80` = Service's port
- Creates tunnel: localhost:8080 → nginx container port 80

**Expected output:**
```
Forwarding from 127.0.0.1:8080 -> 80
Forwarding from [::1]:8080 -> 80
```

**⏰ Keep this running!** (don't press Ctrl+C yet)

### Open Browser and Visit

Open a new terminal window and run:

```bash
curl http://localhost:8080
```

Or open your browser:
```
http://localhost:8080
```

**Expected output:**
```
<!DOCTYPE html>
<html>
<head>
<title>Welcome to nginx!</title>
...
</head>
<body>
<h1>Welcome to nginx!</h1>
...
</body>
</html>
```

✅ **Congratulations!** You just deployed and accessed your first Kubernetes app! 🎉

**When done exploring, stop port-forward:**
```bash
# In the port-forward terminal, press Ctrl+C
```

### Method 2: NodePort (Alternative)

If you prefer direct access, use the NodePort:

```bash
# Get the NodePort
kubectl get svc nginx -o jsonpath='{.spec.ports[0].nodePort}'
```

This gives you the port (e.g., 31234).

Then access it:
```bash
curl http://localhost:31234
```

Or in browser:
```
http://localhost:31234
```

**Note:** localhost works because Kind exposes NodePorts on localhost. In a real cloud cluster, you'd use the node's IP address.

## Part 4: Scale Your Deployment

Deployments make it easy to run multiple replicas. Scale nginx to 3 replicas:

```bash
kubectl scale deployment nginx --replicas=3
```

**Expected output:**
```
deployment.apps/nginx scaled
```

### Verify Scaling

```bash
kubectl get pods
```

**Expected output:**
```
NAME                     READY   STATUS    RESTARTS   AGE
nginx-748c667d99-abc12   1/1     Running   0          5m
nginx-748c667d99-def45   1/1     Running   0          30s
nginx-748c667d99-ghi78   1/1     Running   0          30s
```

✅ All 3 pods are running!

### Load Balancing

Your Service automatically load balances across all 3 pods:

```bash
# Keep port-forward open: kubectl port-forward svc/nginx 8080:80
# Then in another terminal:

for i in {1..9}; do
  curl -s http://localhost:8080 | grep -o '<title>.*</title>'
done
```

All requests get the same "Welcome to nginx!" response, but the Service is routing to different Pods behind the scenes.

### Scale Back Down

```bash
kubectl scale deployment nginx --replicas=1
```

```bash
kubectl get pods
```

**Expected output:**
```
NAME                     READY   STATUS    RESTARTS   AGE
nginx-748c667d99-abc12   1/1     Running   0          8m
```

✅ 2 pods terminated, 1 remains!

## Part 5: Update Your Deployment

Update nginx to a newer image (without downtime):

```bash
kubectl set image deployment/nginx nginx=nginx:latest
```

**What this does:**
- Updates the image from `nginx:alpine` to `nginx:latest`
- Kubernetes performs a rolling update:
  1. Start new pod with latest image
  2. Verify it's healthy
  3. Terminate old pod
  4. Repeat until all pods updated

### Watch the Update

```bash
kubectl get pods -w
```

You'll see:
```
NAME                     READY   STATUS        RESTARTS   AGE
nginx-748c667d99-abc12   1/1     Terminating   0          10m
nginx-5c4d9b8e9f-def45   1/1     Running       0          5s
```

**Zero-downtime deployment!** ✅ (Service continues serving while pods update)

### Verify Update

```bash
kubectl describe deployment nginx | grep Image
```

**Expected output:**
```
Image:      nginx:latest
```

## Part 6: Check Logs

See what's happening inside your containers:

```bash
kubectl logs -f deployment/nginx
```

**Expected output:**
```
2024-01-15 10:23:45 [notice] nginx: master process started
2024-01-15 10:23:45 [notice] nginx: worker process started
127.0.0.1 - - [15/Jan/2024:10:24:01 +0000] "GET / HTTP/1.1" 200 615 "-" "curl/7.68.0"
```

The last line shows your curl request!

**Stop log following:** Press Ctrl+C

## Part 7: Debugging - Access Pod Directly

If something goes wrong, exec into a pod:

```bash
# Get pod name
kubectl get pods -o name

# Exec into it (replace nginx-748c667d99-abc12 with your pod name)
kubectl exec -it nginx-748c667d99-abc12 -- /bin/sh
```

**You're now inside the pod's container:**

```bash
# List files
ls -la

# Check if nginx is running
ps aux | grep nginx

# View nginx config
cat /etc/nginx/nginx.conf

# Test locally
curl localhost

# Exit
exit
```

This is powerful for debugging!

## Part 8: Delete Everything

When done learning, clean up:

```bash
# Delete service
kubectl delete svc nginx

# Delete deployment (also deletes pods)
kubectl delete deployment nginx

# Verify everything is gone
kubectl get all
```

**Expected output:**
```
NAME              TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
service/kubernetes ClusterIP  10.96.0.1    <none>        443/TCP   1h
```

Only the system "kubernetes" service remains (that's normal).

## Validation Checkpoint ✅

Complete all of these checks:

- [ ] `kubectl create deployment nginx --image=nginx:alpine` succeeded
- [ ] `kubectl get deployments` shows nginx with READY=1/1
- [ ] `kubectl expose deployment nginx --port=80 --type=NodePort` succeeded
- [ ] `kubectl port-forward svc/nginx 8080:80` works
- [ ] Browser shows "Welcome to nginx!" at http://localhost:8080
- [ ] `kubectl scale deployment nginx --replicas=3` succeeded
- [ ] All 3 pods running: `kubectl get pods` shows 3 nginx pods
- [ ] Logs work: `kubectl logs deployment/nginx` shows nginx startup

**If all boxes are checked**: Proceed to Lab 1.4 ✅

## Troubleshooting

### Port-forward Fails: "Unable to connect"

**Problem**: `kubectl port-forward svc/nginx 8080:80` shows error.

**Solution:**
1. Verify service exists: `kubectl get svc nginx`
2. Verify pod is running: `kubectl get pods`
3. If pod is CrashLoopBackOff, check logs: `kubectl logs deployment/nginx`
4. Try without port-forward:
   ```bash
   kubectl get svc nginx -o jsonpath='{.spec.ports[0].nodePort}'
   curl http://localhost:NODEPORT
   ```

### Pod Stuck in "Pending"

**Problem**: `kubectl get pods` shows STATUS=Pending after 1 minute.

**Solution:**
1. Check events: `kubectl describe pod <pod-name>`
2. Check nodes: `kubectl get nodes`
3. If no ready nodes, cluster is broken - delete and recreate:
   ```bash
   kind delete cluster --name oppen-lab
   kind create cluster --name oppen-lab
   ```

### Pod in "CrashLoopBackOff"

**Problem**: Pod keeps crashing and restarting.

**Solution:**
1. Check logs: `kubectl logs deployment/nginx`
2. Common issues:
   - Image pull failed → check image name
   - Port already in use → use different port
   - Memory/CPU limit → increase limits
3. Recreate deployment:
   ```bash
   kubectl delete deployment nginx
   kubectl create deployment nginx --image=nginx:alpine
   ```

### Browser Shows "Connection Refused"

**Problem**: `http://localhost:8080` gives connection error.

**Solution:**
1. Verify port-forward is running:
   ```bash
   kubectl port-forward svc/nginx 8080:80
   # Output should show "Forwarding from 127.0.0.1:8080 -> 80"
   ```
2. If not running, the tunnel is closed - restart it
3. Try different port: `kubectl port-forward svc/nginx 9090:80`
4. Then visit: `http://localhost:9090`

### How to Wipe and Start Over

```bash
# Delete everything in default namespace
kubectl delete all --all

# Or delete entire cluster and recreate
kind delete cluster --name oppen-lab
kind create cluster --name oppen-lab
```

## Understanding What Happened

```
kubectl create deployment nginx --image=nginx:alpine
        ↓
  Creates Deployment "nginx"
        ↓
  Deployment creates Pod "nginx-748..."
        ↓
  Pod runs nginx container
        ↓
kubectl expose deployment nginx --port=80 --type=NodePort
        ↓
  Creates Service "nginx"
        ↓
  Service routes traffic to pods with label app=nginx
        ↓
kubectl port-forward svc/nginx 8080:80
        ↓
  Creates tunnel: localhost:8080 → Service port 80 → Pod port 80
        ↓
Browser request to http://localhost:8080
        ↓
Nginx returns "Welcome to nginx!" response
```

## Advanced: Declarative Approach

In Part 1, you used the **imperative** approach (telling Kubernetes what to do with commands).

The **declarative** approach uses YAML files (more professional):

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:latest
        ports:
        - containerPort: 80
---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx
spec:
  type: NodePort
  selector:
    app: nginx
  ports:
  - port: 80
    targetPort: 80
```

Then apply both:
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

You'll use this approach in Lab 1.4 when connecting the AI agent!

## Next Steps

Your first deployment is complete! In Lab 1.4, you'll:
1. Install Ollama (local LLM)
2. Set up MCP servers
3. Build an AI agent
4. Connect the agent to your cluster

→ **Next**: [Lab 1.4: Connect Agent to Kubernetes](./05-lab4-agent.md)

## Quick Reference: Common Commands

```bash
# Deployments
kubectl create deployment <name> --image=<image>
kubectl get deployments
kubectl describe deployment <name>
kubectl delete deployment <name>
kubectl scale deployment <name> --replicas=N
kubectl set image deployment/<name> <container>=<image>

# Pods
kubectl get pods
kubectl logs deployment/<name>
kubectl exec -it <pod-name> -- /bin/sh
kubectl describe pod <pod-name>

# Services
kubectl expose deployment <name> --port=80 --type=NodePort
kubectl get services
kubectl port-forward svc/<name> <local-port>:<svc-port>
kubectl delete svc <name>

# Debugging
kubectl events
kubectl get all
kubectl describe <resource-type> <name>
```

---

**Completed Lab 1.3?** → You're ready to connect an AI agent in Lab 1.4! 🤖

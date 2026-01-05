# Phase 1 Troubleshooting Guide

This guide covers common issues encountered in Phase 1 labs. Problems are organized by lab.

**Quick Navigation:**
- [Lab 1.1: Installation Issues](#lab-11-installation-issues)
- [Lab 1.2: Cluster Creation Issues](#lab-12-cluster-creation-issues)
- [Lab 1.3: Deployment Issues](#lab-13-deployment-issues)
- [Lab 1.4: Agent Issues](#lab-14-agent-issues)
- [General Kubernetes Debugging](#general-kubernetes-debugging)
- [Windows-Specific Issues](#windows-specific-issues)
- [Getting Help](#getting-help)

---

## Lab 1.1: Installation Issues

### "command not found: kind"

**Problem**: After installing Kind, `kind version` returns "command not found"

**Causes:**
1. Installation didn't complete
2. Binary not in PATH
3. Wrong shell (terminal doesn't see new PATH)

**Solutions:**

**Option A: Verify installation location**
```bash
# Find where kind was installed
which kind
find ~ -name "kind" -type f 2>/dev/null

# If found, add to PATH
export PATH="$HOME/.local/bin:$PATH"  # If installed there
# Or
export PATH="/usr/local/bin:$PATH"    # If installed there
```

**Option B: Reinstall via Homebrew (macOS/Linux)**
```bash
brew install kind
# Homebrew automatically puts it in /usr/local/bin
```

**Option C: Manual download (all platforms)**
```bash
# macOS (Apple Silicon)
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-darwin-arm64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# macOS (Intel)
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-darwin-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# Linux
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
```

**Option D: Permanently add to PATH (bash/zsh)**
```bash
# Edit your shell config
nano ~/.bashrc  # or ~/.zshrc, ~/.bash_profile

# Add this line
export PATH="/usr/local/bin:$PATH"

# Save and reload
source ~/.bashrc
```

### "kubectl: command not found"

**Problem**: After installing kubectl, it's not found

**Solutions:**
Same as kind above - it's a PATH issue.

```bash
# Quick fix
export PATH="/usr/local/bin:$PATH"
kubectl version --client

# Permanent fix
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### "Docker Desktop not running" or "Cannot connect to Docker daemon"

**Problem**: Tools installed but Docker is not accessible

**Solutions:**

**macOS/Windows:**
1. Open Docker Desktop application from Applications folder
2. Wait for whale icon to show "Docker is running"
3. Try again: `docker ps`

**Linux:**
```bash
# Start Docker service
sudo systemctl start docker

# Check status
sudo systemctl status docker

# Enable on boot (optional)
sudo systemctl enable docker
```

**Test Docker:**
```bash
docker ps
# Should show (empty or list of containers, no error)
```

### "Docker: permission denied" (Linux)

**Problem**: Get "permission denied while trying to connect to Docker daemon"

**Solution:**
```bash
# Add your user to docker group
sudo usermod -aG docker $USER

# Apply group changes (choose one)
# Option 1: Log out and back in
# Option 2: Restart terminal
# Option 3: Apply group immediately
newgrp docker

# Verify
docker ps  # Should work now
```

### "No space left on device"

**Problem**: Installation fails with disk space error

**Solutions:**
```bash
# Check free space
df -h

# You need ~30GB free (10GB for Kind, 20GB for Ollama model)

# Free space options:
# 1. Delete unused Docker images
docker system prune -a  # ⚠️ Deletes ALL unused images/containers

# 2. Delete old Docker volumes
docker volume prune

# 3. Clean up system files (OS-specific)
# macOS: Clear cache, empty trash
# Linux: sudo apt-get clean && sudo apt-get autoclean
# Windows: Disk Cleanup utility
```

### "Installer hangs or times out"

**Problem**: Download or installation takes forever

**Solutions:**
```bash
# Check internet connection
ping google.com

# If slow, try downloading from browser instead
# Then install from downloaded file

# Or use alternative package manager:
# macOS: MacPorts
sudo port install kind

# Linux: Different distro repos
sudo snap install kind --classic  # Ubuntu/Debian with snap
```

---

## Lab 1.2: Cluster Creation Issues

### "kind create cluster" hangs for >5 minutes

**Problem**: Cluster creation starts but never completes

**Causes:**
1. Downloading Kubernetes image (normal, takes 2-3 min on first run)
2. Docker daemon slow/unresponsive
3. Not enough disk space

**Solutions:**
```bash
# Option 1: Let it finish (first run takes time)
# Wait up to 5 minutes

# Option 2: Force cancel and check what happened
# Press Ctrl+C

# Check if container started
docker ps -a

# View container logs
docker logs oppen-lab-control-plane

# If stuck, clean up and retry
kind delete cluster --name oppen-lab
kind create cluster --name oppen-lab
```

### "Unable to connect to the server" after cluster creation

**Problem**: Cluster was created but `kubectl get nodes` fails

**Causes:**
1. kubectl not configured for the cluster
2. Cluster crashed after creation
3. Wrong context selected

**Solutions:**

**Option 1: Switch to correct context**
```bash
# Check available contexts
kubectl config get-contexts

# Use oppen-lab context
kubectl config use-context kind-oppen-lab

# Try again
kubectl get nodes
```

**Option 2: Check if cluster is still running**
```bash
# List Kind clusters
kind get clusters

# Check if Docker container exists
docker ps | grep oppen-lab

# If missing, recreate
kind create cluster --name oppen-lab
```

**Option 3: Check kubeconfig**
```bash
# View kubeconfig
cat ~/.kube/config

# Should have entry for "kind-oppen-lab"
# If missing, regenerate
kind create cluster --name oppen-lab --config kind-config.yaml
```

### "nodes are NotReady" or "nodes are Unknown"

**Problem**: `kubectl get nodes` shows STATUS = NotReady

**Solutions:**
```bash
# Check what's wrong
kubectl describe node oppen-lab-control-plane

# Common issues:
# 1. CNI (networking) plugin not installed - wait 30 seconds
# 2. kubelet crashed - check logs
docker logs oppen-lab-control-plane | grep -i error

# Nuclear option: delete and recreate
kind delete cluster --name oppen-lab
kind create cluster --name oppen-lab
```

### "Cannot pull image" error during node startup

**Problem**: Cluster creation fails because it can't download Kubernetes image

**Solutions:**
```bash
# Pre-download image manually
docker pull kindest/node:v1.28.0

# Then try cluster creation
kind create cluster --name oppen-lab

# If network is slow, allow more time
kind create cluster --name oppen-lab --wait 10m
```

### "Insufficient CPU/Memory" or "OOMKilled"

**Problem**: Cluster crashes immediately or nodes are NotReady

**Causes:**
- Kind cluster needs ~2GB RAM
- You have less than 4GB available

**Solutions:**
```bash
# Check available memory
free -h  # Linux
vm_stat  # macOS
tasklist  # Windows

# Free up memory:
# 1. Close unused applications
# 2. Increase Docker memory allocation
#    - Docker Desktop → Preferences → Resources → Memory
# 3. Create single-node cluster (not multi-node)

# If still stuck:
kind delete cluster --name oppen-lab
# Close other apps
kind create cluster --name oppen-lab
```

---

## Lab 1.3: Deployment Issues

### "Pod stuck in Pending status"

**Problem**: `kubectl get pods` shows STATUS = Pending after 1-2 minutes

**Causes:**
1. Nodes not ready
2. Image can't be pulled
3. Insufficient resources

**Solutions:**
```bash
# Get details
kubectl describe pod <pod-name>

# Check events section for specific error

# Most common fix:
kubectl get nodes  # Should all be Ready

# If nodes not ready:
kind delete cluster --name oppen-lab
kind create cluster --name oppen-lab
```

### "ImagePullBackOff" or "ErrImagePull"

**Problem**: Pod fails to start because image can't be found

**Causes:**
1. Image name typo
2. Image doesn't exist on Docker Hub
3. Network issue

**Solutions:**
```bash
# For our labs, use official images:
kubectl create deployment nginx --image=nginx:alpine
# (not "ngnix" or "nginx:latest" or custom images)

# Test if image is available locally:
docker images | grep nginx

# If missing, pull first:
docker pull nginx:alpine

# Then load into Kind:
kind load docker-image nginx:alpine --name oppen-lab

# Then create deployment
kubectl create deployment nginx --image=nginx:alpine
```

### "Port already in use" or "Address in use"

**Problem**: `kubectl port-forward svc/nginx 8080:80` fails with port error

**Solutions:**
```bash
# Use different port
kubectl port-forward svc/nginx 9090:80  # Instead of 8080

# Or kill process using port
# macOS/Linux
lsof -i :8080
kill -9 <PID>

# Windows
netstat -ano | findstr :8080
taskkill /PID <PID> /F
```

### "Connection refused" when visiting localhost:8080

**Problem**: Browser shows "connection refused" or timeout

**Causes:**
1. port-forward not running
2. service doesn't exist
3. pod not ready

**Solutions:**
```bash
# Make sure port-forward is running
kubectl port-forward svc/nginx 8080:80
# Should show: "Forwarding from 127.0.0.1:8080 -> 80"

# In another terminal, verify service exists:
kubectl get svc nginx

# And pod is running:
kubectl get pods

# If pod not running, check status:
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

### "service [NAME] not found"

**Problem**: `kubectl expose deployment` or `kubectl get svc` can't find service

**Solutions:**
```bash
# Create service
kubectl expose deployment nginx --port=80 --type=NodePort

# Verify it exists
kubectl get svc

# If still missing, create with different approach:
kubectl expose deployment nginx --type=NodePort
```

### Pod keeps crashing (CrashLoopBackOff)

**Problem**: `kubectl get pods` shows STATUS = CrashLoopBackOff, RESTARTS > 0

**Solutions:**
```bash
# Check logs
kubectl logs <pod-name>

# Get more details
kubectl describe pod <pod-name>

# Look for:
# - Error messages in logs
# - "Last State: Terminated" with error reason

# For nginx specifically, shouldn't happen
# If it does, delete and recreate:
kubectl delete deployment nginx
kubectl create deployment nginx --image=nginx:alpine
```

### Scale command fails

**Problem**: `kubectl scale deployment` doesn't work

**Solutions:**
```bash
# Syntax is:
kubectl scale deployment [NAME] --replicas=[NUMBER]

# Not:
kubectl scale pod ...  # Wrong
kubectl scale service ...  # Wrong

# Verify deployment exists first:
kubectl get deployments
```

---

## Lab 1.4: Agent Issues

### "Ollama is not running"

**Problem**: `curl http://localhost:11434/api/tags` fails

**Solutions:**

**macOS/Linux:**
```bash
# Start Ollama
ollama serve &

# Wait 5 seconds
sleep 5

# Test
curl http://localhost:11434/api/tags
```

**Windows:**
1. Open Start Menu
2. Search for Ollama
3. Click to launch
4. Or check if already running in system tray

**Verify it's listening:**
```bash
# Port 11434 should be listening
lsof -i :11434  # macOS/Linux
netstat -ano | findstr :11434  # Windows
```

### "model not found" when running agent

**Problem**: Agent fails because llama3.2:3b isn't downloaded

**Solutions:**
```bash
# Check if model exists
ollama list

# If not listed, download it
ollama pull llama3.2:3b

# This takes 2-10 minutes depending on internet

# Verify download completed
ollama list
# Should show: llama3.2:3b 2.0 GB
```

### "ModuleNotFoundError: No module named 'langchain_ollama'"

**Problem**: Python can't import LangChain modules

**Solutions:**
```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows

# Reinstall packages
pip install --upgrade pip
pip install -r requirements.txt

# Verify
python3 -c "from langchain_ollama import ChatOllama; print('OK')"
```

### "kubectl: command not found" in agent

**Problem**: Agent script fails because kubectl not accessible

**Causes:**
1. kubectl not installed (see Lab 1.1 troubleshooting)
2. Virtual environment doesn't see kubectl in PATH

**Solutions:**
```bash
# Verify kubectl works in terminal
kubectl version --client

# Then run agent
source venv/bin/activate
python3 agent.py
```

### Agent hangs for 30+ seconds

**Problem**: Agent is very slow responding

**Causes:**
1. Ollama loading model into memory (normal on first request)
2. Kubernetes API slow
3. Subprocess call blocked

**Solutions:**
```bash
# First response takes ~30 seconds while model loads
# Subsequent responses should be less than 5 seconds

# If still slow:
# 1. Check Ollama status
curl http://localhost:11434/api/tags

# 2. Check Kubernetes responsiveness
kubectl get pods

# 3. Monitor system resources
top  # or Task Manager on Windows
# Look for high CPU or memory usage
```

### Agent responds incorrectly

**Problem**: Agent misunderstands questions or gives wrong answers

**Solutions:**
```bash
# This is normal with small 3B model
# Tips to get better answers:

# 1. Ask more specific questions
Ask: "List all pods in the default namespace"
Not: "What's running?"

# 2. Use exact resource names
Ask: "Describe pod nginx-748c667d99-abc12"
Not: "Tell me about nginx"

# 3. Give agent context
Ask: "Show me all Kubernetes deployments"
Not: "What do I have?"

# For production, use larger models (7B, 13B)
# Those are more accurate but need more RAM/GPU
```

### "from k8s_tools import TOOLS" fails

**Problem**: Agent can't find the k8s_tools module

**Solutions:**
```bash
# Make sure k8s_tools.py is in same directory as agent.py
ls -la k8s_tools.py agent.py
# Both should be listed

# If k8s_tools.py doesn't exist, create it
# Follow Lab 1.4 Part 3

# If in wrong directory, move to same folder:
mv /path/to/k8s_tools.py ./k8s_tools.py
```

---

## General Kubernetes Debugging

### Understanding Pod States

```
STATUS          MEANING                          ACTION
────────────────────────────────────────────────────────────
Pending         Waiting for resources/scheduling  Wait or check events
Running         Container is executing           Normal, everything OK
Succeeded       Container exited cleanly         Check logs if unexpected
Failed          Container exited with error      Check logs and events
Unknown         Can't determine state            Check cluster health
CrashLoopBackOff Pod keeps crashing             Check logs, restart policy
```

### Debugging Checklist

When something goes wrong:

```bash
# 1. Check pod status
kubectl get pods

# 2. Describe the pod (shows events)
kubectl describe pod <pod-name>

# 3. Check logs
kubectl logs <pod-name>

# 4. Check node status
kubectl get nodes

# 5. Check events across cluster
kubectl get events --sort-by='.lastTimestamp'

# 6. Nuclear option (fresh start)
kubectl delete all --all
# or
kind delete cluster --name oppen-lab
kind create cluster --name oppen-lab
```

### Useful Debug Commands

```bash
# See what resources exist
kubectl get all

# Show what a resource looks like
kubectl get pod [NAME] -o yaml

# Edit a resource live
kubectl edit deployment [NAME]

# Execute command in pod
kubectl exec -it <pod-name> -- /bin/sh

# Copy file from pod
kubectl cp <pod-name>:/path/to/file ./local/file

# Port forward (bypass service)
kubectl port-forward pod/[NAME] 8080:8000

# Follow logs in real-time
kubectl logs -f deployment/[NAME]

# Watch resource changes
kubectl get pods -w
```

---

## Windows-Specific Issues

### PowerShell Execution Policy Error

**Problem**: Running PowerShell script gives "cannot be loaded because running scripts is disabled"

**Solution:**
```powershell
# Temporarily allow for current session
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# Then run script
.\validate.ps1
```

### WSL2 vs Hyper-V Conflicts

**Problem**: Docker/Kind doesn't work, Windows running both WSL2 and Hyper-V

**Causes:**
Can't run multiple hypervisors simultaneously

**Solutions:**
1. Use Docker Desktop with WSL2 (easiest)
   - Docker Desktop → Settings → Resources → WSL integration
2. Or use WSL2 exclusively (disable Hyper-V)
   - Settings → Turn Windows features on/off → Uncheck Hyper-V

### kubectl in WSL but Kind on Windows

**Problem**: Running Kind on Windows but kubectl in WSL

**Solution:**
Keep both in same environment (both Windows or both WSL2)

```bash
# Option 1: Use Windows terminal, PowerShell, Windows tools
# Option 2: Use WSL2 terminal, Linux tools

# Don't mix
```

### Docker Desktop Resource Limits

**Problem**: Kind cluster OOMKilled but you have 16GB RAM

**Solution:**
Docker Desktop has separate resource allocation

1. Docker Desktop icon → Settings
2. Resources → Memory: Increase to 8GB+ (if available)
3. Resources → CPUs: Increase to 4+ cores
4. Recreate cluster

---

## Getting Help

### Check Error Messages Carefully

1. Read the full error (not just first line)
2. Look for "reason:" or "error:" keywords
3. Search the error message online

### How to Report Issues

When reporting issues:

1. **Include error output:**
   ```bash
   # Save complete output
   kubectl describe pod [NAME] > error.txt
   kubectl logs [NAME] >> error.txt
   ```

2. **Include command that failed:**
   ```bash
   $ kind create cluster --name oppen-lab
   error message here
   ```

3. **Include environment:**
   - OS (macOS/Linux/Windows)
   - RAM and disk free space
   - Docker Desktop version
   - Kind version
   - kubectl version

### Useful Resources

- **Kubernetes Official Docs**: https://kubernetes.io/docs/
- **Kind Documentation**: https://kind.sigs.k8s.io/
- **Ollama Documentation**: https://ollama.com
- **LangChain Docs**: https://python.langchain.com/

### Quick Fix Templates

**Everything broken, want fresh start:**
```bash
kind delete cluster --name oppen-lab
docker system prune -a  # WARNING: Deletes all Docker images/containers
kind create cluster --name oppen-lab
kubectl create deployment nginx --image=nginx:alpine
kubectl expose deployment nginx --port=80 --type=NodePort
```

**Stuck on a specific lab:**
1. Delete just the problematic resource: `kubectl delete deployment/pod/service [NAME]`
2. Reread the lab instructions carefully
3. Follow each step exactly
4. Check validation checkpoints

**Stuck on installation:**
1. Verify each tool individually:
   ```bash
   kind version
   kubectl version --client
   docker ps
   ```
2. If one fails, look up that tool's troubleshooting
3. Usually it's a PATH issue (see "command not found" section)

---

## FAQ

**Q: Can I delete the cluster and start over?**

A: Yes, completely safe!
```bash
kind delete cluster --name oppen-lab
kind create cluster --name oppen-lab
```

**Q: Will I lose my data?**

A: By design, yes. Kind clusters are ephemeral (temporary). That's fine for learning.
For production (Phase 2+), you'd add persistent storage.

**Q: Can I have multiple Kind clusters?**

A: Yes!
```bash
kind create cluster --name lab1
kind create cluster --name lab2
kubectl config use-context kind-lab1
# Now connected to lab1
```

**Q: What if I restart my computer?**

A: Kind cluster will still exist (saved in Docker).
```bash
kind get clusters
# Shows oppen-lab still exists

kubectl get nodes
# Still works
```

**Q: Can I uninstall and reinstall everything?**

A: Completely safe!
```bash
# Delete cluster
kind delete cluster --name oppen-lab

# Uninstall tools (varies by platform)
# Then reinstall from scratch
```

---

## Still Stuck?

1. **Reread the lab** - Instructions are very detailed
2. **Check troubleshooting** - Your issue is probably here
3. **Try fresh start** - Delete cluster and recreate
4. **Take a break** - Many issues resolve after sleep!
5. **Review theory** - Sometimes helps understand concepts

Remember: **Kubernetes is complex. Struggling is normal!** Every expert was once a beginner. 🚀

---

**Keep going! Phase 1 labs are learnable.** → Return to [Phase 1 Overview](./README.md)

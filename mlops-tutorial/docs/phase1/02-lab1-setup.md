# Lab 1.1: Set Up Your Environment

**Estimated Time**: 1 hour
**Difficulty**: Easy
**Goal**: Install Kind, kubectl, and helm on your machine

## Prerequisites Checklist

Before starting, verify you have:

- [ ] **Docker Desktop** or **Rancher Desktop** installed and running
- [ ] **Terminal/CLI** access (Terminal on Mac/Linux, PowerShell on Windows)
- [ ] **Python 3.10+** installed (`python --version` to check)
- [ ] **20GB free disk space**
- [ ] **4GB RAM available** (K8s cluster will use ~2GB)

If any of these are missing, install them first before continuing.

## Part 1: Install Kind (Kubernetes in Docker)

Kind is a tool to run Kubernetes clusters using Docker containers. Perfect for learning and local development.

### macOS

```bash
# Using Homebrew (if installed)
brew install kind

# Or download binary directly
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-darwin-arm64
# (Use darwin-amd64 if on Intel Mac)
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
```

### Linux

```bash
# Using package manager (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y kind

# Or download binary
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
```

### Windows

```powershell
# Using Chocolatey (if installed)
choco install kind

# Or download binary
curl.exe -Lo kind-windows-amd64.exe https://kind.sigs.k8s.io/dl/v0.20.0/kind-windows-amd64.exe
# Move to a directory in your PATH, e.g., C:\Program Files\kind\
```

### Verify Installation

```bash
kind version
```

**Expected output:**
```
kind v0.20.0 go1.20.0 windows/amd64
```

✅ If you see a version number, Kind is installed!

## Part 2: Install kubectl

kubectl is the command-line tool to interact with Kubernetes clusters.

### macOS

```bash
# Using Homebrew
brew install kubectl

# Or download binary
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/darwin/arm64/kubectl"
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin/kubectl
```

### Linux

```bash
# Using snap (Ubuntu)
sudo snap install kubectl --classic

# Or download binary
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin/kubectl
```

### Windows

```powershell
# Using Chocolatey
choco install kubernetes-cli

# Or download from: https://dl.k8s.io/release/stable.txt (get version)
# Then download: https://dl.k8s.io/release/v1.28.0/bin/windows/amd64/kubectl.exe
```

### Verify Installation

```bash
kubectl version --client
```

**Expected output:**
```
Client Version: v1.28.0
Kustomize Version: v5.0.0
```

✅ If you see a version number, kubectl is installed!

## Part 3: Install Helm (Optional but Recommended)

Helm is a package manager for Kubernetes. Makes deploying complex apps easier. We'll use it in later phases.

### macOS

```bash
brew install helm
```

### Linux

```bash
# Using package manager
sudo apt-get install -y helm

# Or download binary
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### Windows

```powershell
choco install kubernetes-helm
```

### Verify Installation

```bash
helm version
```

**Expected output:**
```
version.BuildInfo{Version:"v3.12.0", ...}
```

✅ If you see a version number, Helm is installed!

## Part 4: Verify Docker is Running

Your tools need Docker to function. Verify it's running:

```bash
docker ps
```

**Expected output:**
```
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
(empty - no containers running yet, which is fine)
```

**If you get an error:**
```
Cannot connect to Docker daemon...
```

👉 Open **Docker Desktop** (Mac/Windows) or start **Docker service** (Linux):

```bash
# Linux only
sudo systemctl start docker
```

## Part 5: Validate Your Installation

Run this script to verify everything is installed correctly:

### macOS/Linux

```bash
#!/bin/bash
echo "=== Kubernetes Tools Installation Validation ==="
echo ""

echo "✓ Checking Kind..."
if kind version > /dev/null 2>&1; then
  echo "  ✅ Kind is installed: $(kind version | head -1)"
else
  echo "  ❌ Kind not found. Please install it."
  exit 1
fi

echo ""
echo "✓ Checking kubectl..."
if kubectl version --client > /dev/null 2>&1; then
  echo "  ✅ kubectl is installed: $(kubectl version --client --short)"
else
  echo "  ❌ kubectl not found. Please install it."
  exit 1
fi

echo ""
echo "✓ Checking Helm..."
if helm version > /dev/null 2>&1; then
  echo "  ✅ Helm is installed: $(helm version --short)"
else
  echo "  ⚠️  Helm not installed (optional, but recommended)"
fi

echo ""
echo "✓ Checking Docker..."
if docker ps > /dev/null 2>&1; then
  echo "  ✅ Docker is running and accessible"
else
  echo "  ❌ Docker not running. Please start Docker Desktop or Docker daemon."
  exit 1
fi

echo ""
echo "=== All checks passed! ✅ ==="
echo "Ready to create your first Kubernetes cluster in Lab 1.2"
```

Save this as `validate.sh` and run:

```bash
chmod +x validate.sh
./validate.sh
```

### Windows (PowerShell)

```powershell
# Save as validate.ps1
Write-Host "=== Kubernetes Tools Installation Validation ===" -ForegroundColor Green
Write-Host ""

Write-Host "Checking Kind..." -ForegroundColor Cyan
try {
  kind version | Out-Null
  Write-Host "  ✅ Kind is installed: $(kind version)" -ForegroundColor Green
} catch {
  Write-Host "  ❌ Kind not found" -ForegroundColor Red
  exit 1
}

Write-Host ""
Write-Host "Checking kubectl..." -ForegroundColor Cyan
try {
  kubectl version --client | Out-Null
  Write-Host "  ✅ kubectl is installed" -ForegroundColor Green
} catch {
  Write-Host "  ❌ kubectl not found" -ForegroundColor Red
  exit 1
}

Write-Host ""
Write-Host "Checking Docker..." -ForegroundColor Cyan
try {
  docker ps | Out-Null
  Write-Host "  ✅ Docker is running" -ForegroundColor Green
} catch {
  Write-Host "  ❌ Docker not running" -ForegroundColor Red
  exit 1
}

Write-Host ""
Write-Host "=== All checks passed! ✅ ===" -ForegroundColor Green
```

Run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\validate.ps1
```

## Validation Checkpoint ✅

Complete all of these checks:

- [ ] `kind version` returns v0.20.0 or higher
- [ ] `kubectl version --client` returns v1.28.0 or higher
- [ ] `docker ps` runs without error (Docker running)
- [ ] (Optional) `helm version` returns v3.x.x or higher
- [ ] You have at least 20GB free disk space
- [ ] You have at least 4GB RAM available

**If all boxes are checked**: Proceed to Lab 1.2 ✅

**If any fail**: See troubleshooting section below.

## Troubleshooting

### "kind: command not found"

**Problem**: Kind is installed but not in your PATH.

**Solution (macOS)**:
```bash
# Check where it was installed
which kind
# If not in /usr/local/bin, move it there:
sudo mv ~/.kind/kind /usr/local/bin/kind

# Or reinstall via Homebrew:
brew install kind
```

**Solution (Linux)**:
```bash
# Check if it's in /usr/local/bin
ls -la /usr/local/bin/kind

# If not, download again to correct location:
curl -Lo /usr/local/bin/kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
sudo chmod +x /usr/local/bin/kind
```

**Solution (Windows)**:
```powershell
# Check if kind is in PATH
where kind

# If not found, download and place in C:\Program Files\kind\
# Then add C:\Program Files\kind\ to your PATH environment variable
```

### "kubectl: command not found"

**Problem**: kubectl is not installed or not in PATH.

**Solution**: Follow the install instructions above for your OS, or:

```bash
# Test if it was installed but not in PATH
find / -name "kubectl" 2>/dev/null | head -1

# Then move to /usr/local/bin:
sudo mv /path/to/kubectl /usr/local/bin/kubectl
sudo chmod +x /usr/local/bin/kubectl
```

### "Cannot connect to Docker daemon"

**Problem**: Docker is not running.

**Solution**:
- **macOS/Windows**: Open Docker Desktop application
- **Linux**: Run `sudo systemctl start docker`
- **All**: Verify with `docker ps`

If still failing:
```bash
# Check Docker status (Linux)
sudo systemctl status docker

# Start Docker (Linux)
sudo systemctl start docker

# Enable on boot (Linux)
sudo systemctl enable docker
```

### "Docker: permission denied"

**Problem**: User not in docker group (Linux only).

**Solution**:
```bash
# Add your user to docker group
sudo usermod -aG docker $USER

# Log out and back in (or restart terminal)
# Verify
docker ps
```

### Installer Says "No Space Left on Device"

**Problem**: Not enough disk space.

**Solution**:
- Check free space: `df -h`
- Clear Docker cache: `docker system prune -a` (⚠️ deletes unused containers/images)
- Free up space on your system (at least 20GB needed)

### Installation Hangs or Times Out

**Problem**: Network connection issue or slow download.

**Solution**:
1. Check internet connection
2. Try downloading again
3. Use a faster mirror if available
4. Contact your network admin if on a corporate network

### "Version is too old"

**Problem**: You have an old version installed from years ago.

**Solution**:
```bash
# Remove old version
brew uninstall kind  # macOS
sudo apt-get remove kind  # Linux

# Install new version
brew install kind
# or follow fresh install instructions above
```

## What's Next?

Now that your environment is set up, you're ready for **Lab 1.2: Create Your First Cluster**.

In the next lab, you'll:
1. Create a local Kind cluster named "oppen-lab"
2. Explore its structure with kubectl
3. Access the Kubernetes dashboard (optional)

→ **Next**: [Lab 1.2: Create Your First Cluster](./03-lab2-cluster.md)

## Quick Reference: Installation Checklist

```
Development Machine Setup Checklist:

☐ Docker Desktop / Rancher Desktop running
☐ Kind installed (v0.20.0+)
☐ kubectl installed (v1.28.0+)
☐ Helm installed (v3.0+) [optional but recommended]
☐ Python 3.10+ installed
☐ 20GB+ free disk space
☐ 4GB+ RAM available
☐ All tools verified with validation script

Ready to proceed: ✅
```

## Advanced: Manual PATH Configuration

If tools don't work after installation, you may need to add them to your PATH manually.

### macOS/Linux

```bash
# Edit ~/.bash_profile or ~/.zshrc
nano ~/.zshrc

# Add this line (if tools are in /usr/local/bin):
export PATH="/usr/local/bin:$PATH"

# Save (Ctrl+O, Enter, Ctrl+X)
# Reload shell
source ~/.zshrc
```

### Windows

1. Press `Win + X` → "System"
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "User variables", click "New"
5. Variable name: `PATH`
6. Variable value: `C:\Program Files\kind` (or wherever kind.exe is)
7. Click "OK" → "OK" → Restart PowerShell

---

**Completed Lab 1.1?** → You're ready for Lab 1.2! 🚀

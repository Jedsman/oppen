import subprocess
import random
import time
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [CHAOS] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

NAMESPACE = "apps"
LABEL_SELECTOR = "app.kubernetes.io/name=podinfo"
INTERVAL_SECONDS = 30

def get_pods():
    """Get list of pods matching selector"""
    try:
        cmd = [
            "kubectl", "get", "pods", 
            "-n", NAMESPACE, 
            "-l", LABEL_SELECTOR, 
            "-o", "json"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return [pod['metadata']['name'] for pod in data['items']]
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get pods: {e.stderr}")
        return []
    except Exception as e:
        logger.error(f"Error: {e}")
        return []

def delete_pod(pod_name):
    """Delete a specific pod"""
    try:
        logger.info(f"Targeting victim: {pod_name}")
        cmd = ["kubectl", "delete", "pod", pod_name, "-n", NAMESPACE, "--grace-period=0", "--force"]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"KILLED: {pod_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to delete {pod_name}: {e.stderr}")
        return False

def main():
    logger.info(f"Chaos Monkey started. Hunting in namespace '{NAMESPACE}' for '{LABEL_SELECTOR}'...")
    logger.info(f"Interval: {INTERVAL_SECONDS} seconds.")
    
    try:
        while True:
            pods = get_pods()
            if not pods:
                logger.warning("No pods found matching selector. Waiting...")
            else:
                victim = random.choice(pods)
                delete_pod(victim)
            
            logger.info(f"Sleeping for {INTERVAL_SECONDS}s...")
            time.sleep(INTERVAL_SECONDS)
            
    except KeyboardInterrupt:
        logger.info("Chaos Monkey stopped by user.")

if __name__ == "__main__":
    main()

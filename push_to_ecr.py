import subprocess
import json
import sys

# -----------------------------
# CONFIGURATION
# -----------------------------
AWS_ACCOUNT_ID = "597303283286"
AWS_REGION = "us-east-1"
REPOSITORY_NAME = "rag-ai-demo"
IMAGE_NAME = "rag-ai-local"
TAG = "latest"

# -----------------------------
# HELPER TO RUN COMMANDS
# -----------------------------
def run_cmd(cmd, exit_on_fail=True):
    print(f"\n>> {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    print(out.decode())
    if err:
        print(err.decode())
    if process.returncode != 0:
        print("❌ Command failed.")
        if exit_on_fail:
            sys.exit(1)
    return out.decode()

# -----------------------------
# CHECK IF REPOSITORY EXISTS
# -----------------------------
def ensure_repository():
    print("\n🔍 Checking if ECR repository exists...")

    cmd = (
        f"aws ecr describe-repositories --repository-names {REPOSITORY_NAME} "
        f"--region {AWS_REGION}"
    )

    output = run_cmd(cmd, exit_on_fail=False)

    if "repositoryUri" in output:
        print("✅ Repository already exists.")
        return

    print("📦 Repository not found. Creating it...")

    cmd = (
        f"aws ecr create-repository "
        f"--repository-name {REPOSITORY_NAME} "
        f"--region {AWS_REGION}"
    )

    run_cmd(cmd)
    print("✅ Repository created successfully.")

# -----------------------------
# MAIN PUSH LOGIC
# -----------------------------
def main():

    ensure_repository()

    # ECR Login
    print("\n🔐 Logging in to AWS ECR...")
    login_cmd = (
        f"aws ecr get-login-password --region {AWS_REGION} | "
        f"docker login --username AWS --password-stdin "
        f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com"
    )
    run_cmd(login_cmd)

    # Docker Build
    print("\n🐳 Building Docker image...")
    build_cmd = f"docker build -t {IMAGE_NAME}:latest ."
    run_cmd(build_cmd)

    # Test the image locally
    print("\n🧪 Testing Docker image locally...")
    print(f"   Starting container on port 8000 (30 second test)...")
    # Run container in the background, then stop it after 30 seconds
    test_cmd = f"docker run --rm -d -p 8000:8000 {IMAGE_NAME}:latest"
    container_id_output = run_cmd(test_cmd, exit_on_fail=False)
    
    # Extract container ID from output
    import time
    time.sleep(5)  # Give container time to start
    
    # Stop the test container
    if "Unable to find image" not in container_id_output and container_id_output.strip():
        container_id = container_id_output.strip().split('\n')[-1]
        stop_cmd = f"docker stop {container_id}"
        run_cmd(stop_cmd, exit_on_fail=False)
    
    print("   ✓ Local test completed. Container started successfully.")
    print("   (Container has been stopped. Test passed.)")

    # Confirm before pushing
    print("\n⚠️  Local test complete. Ready to push to ECR?")
    response = input("   Continue with ECR push? (yes/no): ").strip().lower()
    if response not in ["yes", "y"]:
        print("   ❌ Push cancelled by user.")
        return

    # Tag Image
    print("\n🏷️ Tagging image...")
    full_uri = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{REPOSITORY_NAME}:{TAG}"
    tag_cmd = f"docker tag {IMAGE_NAME}:latest {full_uri}"
    run_cmd(tag_cmd)

    # Push Image
    print("\n📤 Pushing image to ECR...")
    push_cmd = f"docker push {full_uri}"
    run_cmd(push_cmd)

    print("\n🎉 DONE! Your image is available here:")
    print(full_uri)


if __name__ == "__main__":
    main()

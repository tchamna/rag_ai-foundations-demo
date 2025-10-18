#!/usr/bin/env python3
"""
Build, push Docker image to Azure Container Registry (ACR), and optionally configure
an Azure Web App to use the pushed image.

Intended to be called from CI (GitHub Actions) or locally. Uses the az CLI and docker.
"""

import argparse
import subprocess
import sys
import shlex


def run(cmd, capture=False):
    print(f"> {cmd}")
    if capture:
        proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            print(proc.stdout, file=sys.stderr)
            raise SystemExit(proc.returncode)
        return proc.stdout.strip()
    else:
        subprocess.run(shlex.split(cmd), check=True)


def get_acr_login_server(acr_name):
    return run(f"az acr show --name {acr_name} --query loginServer -o tsv", capture=True)


def acr_login(acr_name):
    print("Logging into ACR with az acr login...")
    run(f"az acr login --name {acr_name}")


def build_image(image_full, dockerfile, context):
    print(f"Building image {image_full} (dockerfile={dockerfile})...")
    run(f"docker build -f {dockerfile} -t {image_full} {context}")


def push_image(image_full):
    print(f"Pushing image {image_full} ...")
    run(f"docker push {image_full}")


def configure_webapp(app_name, resource_group, image_full, login_server):
    print(f"Configuring Azure Web App '{app_name}' (rg={resource_group}) to use image {image_full} ...")
    run(
        f"az webapp config container set --name {app_name} --resource-group {resource_group} --docker-custom-image-name {image_full} --docker-registry-server-url https://{login_server}"
    )
    run(
        f"az webapp config appsettings set --resource-group {resource_group} --name {app_name} --settings WEBSITES_PORT=8000"
    )
    print("Configured web app container settings.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--acr-name", required=True, help="ACR name (short name, not login server)")
    p.add_argument("--repo", required=True, help="Repository/image name to use (e.g. owner/repo)")
    p.add_argument("--tag", default="latest", help="Image tag (default: latest)")
    p.add_argument("--dockerfile", default="Dockerfile.backend", help="Path to Dockerfile")
    p.add_argument("--context", default=".", help="Docker build context (default: .)")
    p.add_argument("--configure-webapp", action="store_true", help="After push, configure Azure Web App to use the image")
    p.add_argument("--app-name", help="Azure Web App name (required if --configure-webapp)")
    p.add_argument("--resource-group", help="Azure resource group for the Web App (required if --configure-webapp)")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        login_server = get_acr_login_server(args.acr_name)
    except SystemExit:
        print("Failed to resolve ACR login server. Ensure your Azure CLI is logged in and ACR name is correct.", file=sys.stderr)
        sys.exit(1)

    image_full = f"{login_server}/{args.repo}:{args.tag}"

    try:
        acr_login(args.acr_name)
    except subprocess.CalledProcessError as e:
        print("ACR login failed.", file=sys.stderr)
        sys.exit(e.returncode)

    try:
        build_image(image_full, args.dockerfile, args.context)
    except subprocess.CalledProcessError as e:
        print("Docker build failed.", file=sys.stderr)
        sys.exit(e.returncode)

    try:
        push_image(image_full)
    except subprocess.CalledProcessError as e:
        print("Docker push failed.", file=sys.stderr)
        sys.exit(e.returncode)

    print(f"Successfully pushed {image_full}")

    if args.configure_webapp:
        if not args.app_name or not args.resource_group:
            print("When using --configure-webapp you must specify --app-name and --resource-group", file=sys.stderr)
            sys.exit(2)
        try:
            configure_webapp(args.app_name, args.resource_group, image_full, login_server)
        except subprocess.CalledProcessError as e:
            print("Failed to configure webapp.", file=sys.stderr)
            sys.exit(e.returncode)


if __name__ == "__main__":
    main()

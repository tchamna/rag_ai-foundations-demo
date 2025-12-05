# AWS EC2 Deployment Guide

This guide walks you through deploying the RAG AI Foundations Demo to AWS EC2 with automated CI/CD, custom domain, and HTTPS.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Step 1: Set Up AWS Resources](#step-1-set-up-aws-resources)
- [Step 2: Prepare EC2 Instance](#step-2-prepare-ec2-instance)
- [Step 3: Configure GitHub Secrets](#step-3-configure-github-secrets)
- [Step 4: Set Up CI/CD Pipeline](#step-4-set-up-cicd-pipeline)
- [Step 5: Configure Custom Domain](#step-5-configure-custom-domain)
- [Step 6: Set Up HTTPS with Let's Encrypt](#step-6-set-up-https-with-lets-encrypt)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- AWS account with appropriate permissions
- GitHub account with repository access
- Domain name with DNS management access
- Basic knowledge of SSH and terminal commands

## Step 1: Set Up AWS Resources

### 1.1 Create ECR Repository

```bash
aws ecr create-repository --repository-name rag-ai-demo --region us-east-1
```

Note your repository URI (e.g., `597303283286.dkr.ecr.us-east-1.amazonaws.com/rag-ai-demo`)

### 1.2 Launch EC2 Instance

1. Go to AWS EC2 Console
2. Launch a new instance:
   - **AMI**: Amazon Linux 2023 or Ubuntu 22.04 LTS
   - **Instance Type**: t2.medium or larger (recommended for ML workloads)
   - **Storage**: 20 GB minimum
3. Create or select a key pair for SSH access
4. Configure Security Group with the following inbound rules:
   - Port 22 (SSH) - Your IP
   - Port 80 (HTTP) - 0.0.0.0/0
   - Port 443 (HTTPS) - 0.0.0.0/0
   - Port 8080 (App) - 0.0.0.0/0

Note your instance's public IP address (e.g., `13.220.111.43`)

### 1.3 Create IAM User for GitHub Actions

1. Create IAM user with programmatic access
2. Attach policies:
   - `AmazonEC2ContainerRegistryFullAccess`
   - Or create custom policy with ECR push/pull permissions
3. Save the Access Key ID and Secret Access Key

## Step 2: Prepare EC2 Instance

### 2.1 Connect to EC2

```bash
ssh -i "your-key.pem" ec2-user@13.220.111.43
```

### 2.2 Run Setup Script

```bash
# Clone your repository
git clone https://github.com/tchamna/rag_ai-foundations-demo.git
cd rag_ai-foundations-demo

# Make setup script executable
chmod +x scripts/setup_ec2.sh

# Run the setup script
./scripts/setup_ec2.sh
```

This script will:
- Install Docker and Docker Compose
- Install AWS CLI v2
- Configure Docker permissions
- Set up AWS credentials

### 2.3 Configure AWS Credentials

When prompted by the setup script, enter:
- AWS Access Key ID
- AWS Secret Access Key
- Default region: `us-east-1`
- Default output format: `json`

### 2.4 Test ECR Login

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  597303283286.dkr.ecr.us-east-1.amazonaws.com
```

You should see "Login Succeeded"

## Step 3: Configure GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add the following secrets:

| Secret Name | Value | Description |
|------------|-------|-------------|
| `AWS_ACCESS_KEY_ID` | Your IAM access key | For ECR push |
| `AWS_SECRET_ACCESS_KEY` | Your IAM secret key | For ECR push |
| `AWS_ACCOUNT_ID` | Your AWS account ID | 12-digit number |
| `EC2_HOST` | Your EC2 public IP | e.g., 13.220.111.43 |
| `EC2_USER` | SSH user | `ec2-user` or `ubuntu` |
| `EC2_SSH_PRIVATE_KEY` | Your private key | Full contents of .pem file |

## Step 4: Set Up CI/CD Pipeline

The repository includes two GitHub Actions workflows:

### 4.1 ECR Deployment (`.github/workflows/deploy-ecr.yml`)

Automatically builds and pushes Docker image to ECR on every push to `main`:

```yaml
name: Deploy to AWS ECR

on:
  push:
    branches: [ main ]

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1
      
      - name: Build and push Docker image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: rag-ai-demo
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
```

### 4.2 EC2 Deployment (`.github/workflows/deploy-ec2.yml`)

Automatically deploys to EC2 after ECR push succeeds:

```yaml
name: Deploy to EC2

on:
  workflow_run:
    workflows: ["Deploy to AWS ECR"]
    types:
      - completed

jobs:
  deploy:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    steps:
      - name: Deploy to EC2
        uses: appleboy/ssh-action@master
        env:
          AWS_ACCOUNT_ID: ${{ secrets.AWS_ACCOUNT_ID }}
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ${{ secrets.EC2_USER }}
          key: ${{ secrets.EC2_SSH_PRIVATE_KEY }}
          envs: AWS_ACCOUNT_ID
          script: |
            aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
            docker pull $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/rag-ai-demo:latest
            docker stop rag-ai-demo || true
            docker rm rag-ai-demo || true
            docker run -d --name rag-ai-demo --restart unless-stopped -p 8080:8080 $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/rag-ai-demo:latest
```

### 4.3 Test the Pipeline

1. Make a small change to your code
2. Commit and push to `main`:
   ```bash
   git add .
   git commit -m "Test deployment pipeline"
   git push
   ```
3. Go to GitHub Actions tab to watch the deployment
4. Verify at `http://your-ec2-ip:8080`

## Step 5: Configure Custom Domain

### 5.1 Set Up DNS Records

In your domain registrar's DNS settings (e.g., Domain.com, GoDaddy, Cloudflare):

1. Create an A record:
   - **Host**: `rag` (or your subdomain)
   - **Type**: `A`
   - **Value**: Your EC2 IP (e.g., `13.220.111.43`)
   - **TTL**: 300 (5 minutes)

2. (Optional) Create www subdomain:
   - **Host**: `www`
   - **Type**: `A`
   - **Value**: Your EC2 IP
   - **TTL**: 300

### 5.2 Verify DNS Propagation

Wait 5-10 minutes, then test:

```bash
nslookup rag.tchamna.com
```

You should see your EC2 IP address in the response.

## Step 6: Set Up HTTPS with Let's Encrypt

### 6.1 SSH to EC2 Instance

```bash
ssh -i "your-key.pem" ec2-user@13.220.111.43
```

### 6.2 Run HTTPS Setup Script

```bash
cd rag_ai-foundations-demo

# Make script executable
chmod +x scripts/setup_nginx_https.sh

# Run with your domain and email
./scripts/setup_nginx_https.sh rag.tchamna.com your-email@example.com
```

The script will:
1. Install Nginx
2. Stop any conflicting services on port 80
3. Start your Docker container on port 8080
4. Configure Nginx as reverse proxy (port 80 → 8080)
5. Install Certbot
6. Obtain Let's Encrypt SSL certificate
7. Configure HTTPS (port 443 → 8080)
8. Set up automatic certificate renewal

### 6.3 Verify HTTPS

Visit your domain:
- `https://rag.tchamna.com` ✅

You should see:
- Valid SSL certificate (green padlock)
- Your Streamlit application running
- Secure WebSocket connections

## Verification

### Test All Endpoints

```bash
# HTTP (redirects to HTTPS)
curl -I http://rag.tchamna.com

# HTTPS
curl -I https://rag.tchamna.com

# Direct app access (if security group allows)
curl -I http://13.220.111.43:8080
```

### Check Docker Container

```bash
docker ps
docker logs rag-ai-demo
```

### Check Nginx Status

```bash
sudo systemctl status nginx
sudo nginx -t  # Test configuration
```

### Check SSL Certificate

```bash
sudo certbot certificates
```

## Troubleshooting

### Issue: Port 8080 not accessible

**Solution**: Verify Docker container is running on port 8080:
```bash
docker ps
docker logs rag-ai-demo
netstat -tlnp | grep 8080
```

### Issue: HTTPS not working

**Solution**: Check Nginx configuration and certificate:
```bash
sudo nginx -t
sudo certbot certificates
sudo tail -f /var/log/nginx/error.log
```

### Issue: SSL certificate renewal fails

**Solution**: Certbot auto-renewal is configured, but you can manually test:
```bash
sudo certbot renew --dry-run
```

### Issue: Application won't start

**Solution**: Check Docker logs:
```bash
docker logs rag-ai-demo --tail 100
```

Common issues:
- Missing vectorstore files
- Insufficient memory (upgrade to t2.medium or larger)
- Port conflicts

### Issue: CI/CD pipeline fails

**Solution**: Check GitHub Actions logs and verify:
- All secrets are correctly configured
- EC2 instance has Docker and AWS CLI installed
- ECR repository exists and is accessible
- SSH key is valid and has proper permissions

### Issue: DNS not resolving

**Solution**: 
```bash
# Check DNS propagation
nslookup rag.tchamna.com

# Clear local DNS cache (Windows)
ipconfig /flushdns

# Wait longer - DNS can take up to 48 hours, though usually 5-10 minutes
```

## Cost Optimization Tips

1. **Use Spot Instances**: Save up to 90% on EC2 costs
2. **Stop instance when not in use**: Only pay for storage
3. **Use Application Load Balancer**: Better for production with auto-scaling
4. **Enable CloudWatch alarms**: Monitor costs and usage
5. **Use ECR lifecycle policies**: Auto-delete old images

## Security Best Practices

1. **Restrict SSH access**: Update security group to allow only your IP
2. **Use IAM roles**: Attach IAM role to EC2 instead of storing credentials
3. **Enable CloudWatch Logs**: Monitor application and Nginx logs
4. **Set up CloudWatch Alarms**: Alert on high CPU, memory, or costs
5. **Regular updates**: Keep OS, Docker, and dependencies updated
6. **Use secrets manager**: For sensitive environment variables
7. **Enable VPC Flow Logs**: Monitor network traffic

## Production Deployment Checklist

- [ ] EC2 instance type sized appropriately (t2.medium or larger)
- [ ] Security groups configured with minimal access
- [ ] IAM role attached to EC2 instance
- [ ] CloudWatch monitoring enabled
- [ ] SSL certificate installed and auto-renewal configured
- [ ] Backups configured (EBS snapshots)
- [ ] Health checks configured
- [ ] Error tracking set up (e.g., Sentry)
- [ ] Cost alerts configured
- [ ] Documentation updated with production URLs

## Related Documentation

- [Azure Deployment Guide](DEPLOY_AZURE.md)
- [ECS Fargate Deployment](../docs/DEPLOY_ECS.md)
- [Architecture Overview](ARCHITECTURE.md)
- [Usage Guide](USAGE.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/tchamna/rag_ai-foundations-demo/issues
- Main README: [README.md](../README.md)

## Live Demo

🚀 **Production URL**: https://rag.tchamna.com

Deployed on: December 4, 2025

# EC2 Deployment Setup

This guide explains how to set up automatic deployment to EC2 after the Docker image is pushed to ECR.

## Prerequisites

1. **EC2 Instance** with:
   - Docker installed
   - AWS CLI installed and configured
   - Security group allowing inbound traffic on port 80 (HTTP)
   - IAM role or credentials with ECR pull permissions

2. **GitHub Secrets** configured in your repository:
   - `AWS_ACCESS_KEY_ID` - AWS access key (already configured for ECR)
   - `AWS_SECRET_ACCESS_KEY` - AWS secret key (already configured for ECR)
   - `EC2_HOST` - Public IP or DNS of your EC2 instance
   - `EC2_USER` - SSH username (typically `ec2-user` for Amazon Linux or `ubuntu` for Ubuntu)
   - `EC2_SSH_PRIVATE_KEY` - Private SSH key for EC2 access

## EC2 Instance Setup

### 1. Launch EC2 Instance

```bash
# Using AWS CLI (example for Amazon Linux 2023)
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxx \
  --subnet-id subnet-xxxxxxxxx \
  --iam-instance-profile Name=ECRAccessRole \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=rag-ai-demo}]'
```

### 2. Connect to EC2 and Install Dependencies

```bash
# SSH into your instance
ssh -i your-key.pem ec2-user@your-ec2-ip

# Update system
sudo yum update -y

# Install Docker
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Log out and back in for docker group to take effect
exit
```

### 3. Configure IAM Role for ECR Access

Create an IAM role with this policy and attach it to your EC2 instance:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    }
  ]
}
```

Or configure AWS credentials manually:

```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter default region: us-east-1
# Enter default output format: json
```

### 4. Configure Security Group

Ensure your EC2 security group allows:
- **Inbound**: Port 80 (HTTP) from 0.0.0.0/0 (or your specific IP range)
- **Inbound**: Port 22 (SSH) from your IP for management

```bash
# Example using AWS CLI
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxxx \
  --protocol tcp \
  --port 80 \
  --cidr 0.0.0.0/0
```

## GitHub Secrets Setup

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

1. **EC2_HOST**
   - Value: Your EC2 public IP or DNS (e.g., `ec2-xx-xx-xx-xx.compute-1.amazonaws.com` or `54.123.45.67`)

2. **EC2_USER**
   - Value: SSH username
   - Amazon Linux: `ec2-user`
   - Ubuntu: `ubuntu`
   - Red Hat: `ec2-user`

3. **EC2_SSH_PRIVATE_KEY**
   - Value: Content of your private key file
   - To get the value: `cat your-key.pem`
   - Paste the entire key including `-----BEGIN RSA PRIVATE KEY-----` and `-----END RSA PRIVATE KEY-----`

## Deployment Workflow

The deployment happens automatically:

1. Code is pushed to `main` branch
2. **Build and Push to AWS ECR** workflow runs first
3. Once ECR push succeeds, **Deploy to EC2** workflow triggers automatically
4. EC2 pulls the latest image from ECR
5. Stops old container, starts new container
6. Application is available at `http://your-ec2-ip`

### Manual Deployment

You can also trigger deployment manually:

1. Go to **Actions** tab in GitHub
2. Select **Deploy to EC2** workflow
3. Click **Run workflow**

## Configuration

Edit `.github/workflows/deploy-ec2.yml` to customize:

- `HOST_PORT: 80` - Change to expose on different port (e.g., 8080)
- `CONTAINER_PORT: 8000` - Internal container port (matches Dockerfile)
- `CONTAINER_NAME: rag-ai-app` - Docker container name

## Monitoring

### Check deployment status
- GitHub Actions tab shows deployment progress
- Deployment summary appears after successful deployment

### SSH into EC2 to check container
```bash
ssh -i your-key.pem ec2-user@your-ec2-ip

# Check running containers
docker ps

# View container logs
docker logs rag-ai-app

# Follow logs in real-time
docker logs -f rag-ai-app

# Restart container manually
docker restart rag-ai-app

# Stop container
docker stop rag-ai-app
```

### Access application
```bash
# Check if app is responding
curl http://your-ec2-ip

# Open in browser
# http://your-ec2-ip
```

## Troubleshooting

### Container not starting
```bash
# Check container logs
docker logs rag-ai-app

# Check if image was pulled
docker images | grep rag-ai-demo

# Try running interactively
docker run -it --rm -p 80:8000 YOUR_ECR_URI
```

### ECR authentication issues
```bash
# Manually login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  597303283286.dkr.ecr.us-east-1.amazonaws.com
```

### Port already in use
```bash
# Check what's using port 80
sudo lsof -i :80

# Kill the process or change HOST_PORT in workflow
```

### SSH connection issues
- Verify security group allows SSH (port 22) from your IP
- Check that private key matches the EC2 key pair
- Verify EC2_HOST is correct (public IP or DNS)
- Ensure private key has correct permissions: `chmod 600 your-key.pem`

## Cost Optimization

- Use **t3.small** or **t3.medium** for production
- Use **t2.micro** for testing (free tier eligible)
- Set up CloudWatch alarms for CPU/memory usage
- Consider using AWS App Runner or ECS Fargate for auto-scaling

## Next Steps

After deployment:

1. **Set up a domain**: Use Route 53 to point a domain to your EC2 IP
2. **Enable HTTPS**: Use Let's Encrypt with nginx reverse proxy
3. **Set up monitoring**: CloudWatch or Datadog for metrics
4. **Configure backups**: Snapshot the vectorstore data periodically
5. **Load balancing**: Use Application Load Balancer for multiple instances

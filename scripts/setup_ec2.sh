#!/bin/bash
# EC2 Instance Setup Script for RAG AI Demo
# Run this on your EC2 instance to prepare it for deployment

set -e

echo "=========================================="
echo "Setting up EC2 for RAG AI Demo Deployment"
echo "=========================================="

# Update system
echo "📦 Updating system packages..."
sudo yum update -y

# Install Docker
echo "🐳 Installing Docker..."
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Install AWS CLI v2
echo "☁️ Installing AWS CLI v2..."
if ! command -v aws &> /dev/null; then
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf aws awscliv2.zip
else
    echo "AWS CLI already installed"
fi

# Configure AWS credentials (if not using IAM role)
echo ""
echo "=========================================="
echo "AWS Configuration"
echo "=========================================="
echo "If your EC2 instance has an IAM role with ECR permissions, you can skip this."
echo "Otherwise, configure AWS credentials now:"
read -p "Do you want to configure AWS credentials? (y/n): " configure_aws

if [ "$configure_aws" = "y" ]; then
    aws configure
fi

# Test Docker
echo ""
echo "=========================================="
echo "Testing Docker Installation"
echo "=========================================="
docker --version
docker ps

# Test ECR login
echo ""
echo "=========================================="
echo "Testing ECR Login"
echo "=========================================="
AWS_REGION="us-east-1"
ECR_REGISTRY="597303283286.dkr.ecr.us-east-1.amazonaws.com"

echo "Attempting to login to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY

if [ $? -eq 0 ]; then
    echo "✅ ECR login successful!"
else
    echo "❌ ECR login failed. Check your AWS credentials or IAM role."
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ EC2 Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Log out and back in for docker group to take effect: exit"
echo "2. Ensure your security group allows inbound traffic on port 80"
echo "3. Push your code to trigger the deployment workflow"
echo "4. Or manually trigger the 'Deploy to EC2' workflow in GitHub Actions"
echo ""
echo "Your application will be available at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo ""

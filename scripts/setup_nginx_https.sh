#!/bin/bash
# Nginx + HTTPS Setup Script for RAG AI Demo on EC2
# This script sets up Nginx as a reverse proxy with optional HTTPS via Let's Encrypt

set -e

echo "=========================================="
echo "Nginx + HTTPS Setup for RAG AI Demo"
echo "=========================================="

# Variables
DOMAIN=${1:-""}
EMAIL=${2:-""}

# Install Nginx
echo "📦 Installing Nginx..."
sudo yum install -y nginx

# Start and enable Nginx
sudo systemctl start nginx
sudo systemctl enable nginx

echo "✅ Nginx installed and started"

# Stop old container if running
echo "🐳 Stopping old container..."
docker stop rag-ai-app 2>/dev/null || true
docker rm rag-ai-app 2>/dev/null || true

# Start app on port 8080
echo "🐳 Starting app container on port 8080..."
docker run -d --name rag-ai-app \
  --restart unless-stopped \
  -p 8080:8080 \
  597303283286.dkr.ecr.us-east-1.amazonaws.com/rag-ai-demo:latest

echo "✅ App container started on port 8080"

# Wait for container to be ready
sleep 5

# Test if app is responding
if curl -s http://localhost:8080 > /dev/null; then
    echo "✅ App is responding on localhost:8080"
else
    echo "⚠️ App might not be ready yet, but continuing..."
fi

# Configure Nginx
echo "⚙️ Configuring Nginx..."

# Backup default config
sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup

# Create Nginx config for the app
sudo tee /etc/nginx/conf.d/rag-ai-app.conf > /dev/null <<'EOF'
server {
    listen 80;
    server_name _;

    # Increase buffer sizes for Streamlit
    proxy_buffer_size 128k;
    proxy_buffers 4 256k;
    proxy_busy_buffers_size 256k;

    location / {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }

    # Streamlit health check endpoint
    location /_stcore/health {
        proxy_pass http://localhost:8080/_stcore/health;
        proxy_http_version 1.1;
    }
}
EOF

# Test Nginx config
echo "🧪 Testing Nginx configuration..."
sudo nginx -t

# Reload Nginx
echo "🔄 Reloading Nginx..."
sudo systemctl reload nginx

echo "✅ Nginx configured and reloaded"

# Get public IP
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Your app is now accessible at:"
echo "  HTTP: http://${PUBLIC_IP}"
echo ""

# Check if domain and email provided for HTTPS
if [ -n "$DOMAIN" ] && [ -n "$EMAIL" ]; then
    echo "=========================================="
    echo "Setting up HTTPS with Let's Encrypt..."
    echo "=========================================="
    
    # Install Certbot
    echo "📦 Installing Certbot..."
    sudo yum install -y certbot python3-certbot-nginx
    
    # Update Nginx config with domain
    sudo sed -i "s/server_name _;/server_name ${DOMAIN};/" /etc/nginx/conf.d/rag-ai-app.conf
    sudo systemctl reload nginx
    
    # Get SSL certificate
    echo "🔒 Obtaining SSL certificate..."
    sudo certbot --nginx -d ${DOMAIN} --non-interactive --agree-tos -m ${EMAIL}
    
    # Set up auto-renewal
    echo "⏰ Setting up auto-renewal..."
    sudo systemctl enable certbot-renew.timer
    
    echo ""
    echo "=========================================="
    echo "✅ HTTPS Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Your app is now accessible at:"
    echo "  HTTP:  http://${DOMAIN}"
    echo "  HTTPS: https://${DOMAIN}"
    echo ""
    echo "SSL certificate will auto-renew via systemd timer"
    
else
    echo "=========================================="
    echo "Optional: Enable HTTPS"
    echo "=========================================="
    echo ""
    echo "To enable HTTPS, you need:"
    echo "  1. A domain name pointing to this EC2 IP (${PUBLIC_IP})"
    echo "  2. Your email address for Let's Encrypt notifications"
    echo ""
    echo "Then run:"
    echo "  ./setup_nginx_https.sh your-domain.com your-email@example.com"
    echo ""
fi

echo "=========================================="
echo "Useful Commands:"
echo "=========================================="
echo ""
echo "Check Nginx status:"
echo "  sudo systemctl status nginx"
echo ""
echo "View Nginx logs:"
echo "  sudo tail -f /var/log/nginx/access.log"
echo "  sudo tail -f /var/log/nginx/error.log"
echo ""
echo "Check app container:"
echo "  docker logs -f rag-ai-app"
echo ""
echo "Restart Nginx:"
echo "  sudo systemctl restart nginx"
echo ""
echo "Restart app container:"
echo "  docker restart rag-ai-app"
echo ""

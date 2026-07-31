# AWS EC2 Deployment Guide

Complete step-by-step guide for deploying the Search Relevancy application on AWS EC2.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [AWS Setup](#aws-setup)
3. [EC2 Instance Configuration](#ec2-instance-configuration)
4. [Application Deployment](#application-deployment)
5. [Production Configuration](#production-configuration)
6. [Monitoring and Scaling](#monitoring-and-scaling)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- AWS Account with EC2 access
- SSH client (PuTTY on Windows, Terminal on Mac/Linux)
- Domain name (optional, for production)
- SSL certificate (optional, for HTTPS)

---

## AWS Setup

### 1. Create Key Pair

1. Go to AWS Console → EC2 → Key Pairs
2. Click "Create Key Pair"
3. Name: `search-relevancy-key`
4. Key type: RSA
5. Download and save securely

On Linux/Mac:
```bash
chmod 400 search-relevancy-key.pem
```

### 2. Create Security Group

1. Go to EC2 → Security Groups
2. Click "Create security group"
3. Name: `search-relevancy-sg`
4. Add inbound rules:

| Protocol | Port | Source | Purpose |
|---|---|---|---|
| SSH | 22 | Your IP | SSH access |
| TCP | 5000 | 0.0.0.0/0 | Flask API |
| TCP | 443 | 0.0.0.0/0 | HTTPS (optional) |
| TCP | 80 | 0.0.0.0/0 | HTTP (optional) |

5. Click "Create security group"

---

## EC2 Instance Configuration

### 1. Launch EC2 Instance

1. Go to AWS Console → EC2 → Instances
2. Click "Launch Instance"

**AMI Selection:**
- Ubuntu Server 22.04 LTS (Free tier eligible)
- Region: Choose closest to your location

**Instance Type:**
- Development: `t3.medium` (4 GB RAM, 2 vCPU)
- Production: `t3.large` or `t3.xlarge`

**Storage:**
- 50-100 GB gp3 (General Purpose SSD)

**Security Group:**
- Select `search-relevancy-sg`

**Key Pair:**
- Select `search-relevancy-key`

3. Click "Launch Instance"

### 2. Get Instance Details

1. Note the **Public IPv4 address** (e.g., `54.123.45.67`)
2. Note the **Private IPv4 address** (for internal use)

---

## Application Deployment

### Step 1: Connect to Instance

```bash
# On your local machine
ssh -i search-relevancy-key.pem ubuntu@YOUR_EC2_PUBLIC_IP

# Example
ssh -i search-relevancy-key.pem ubuntu@54.123.45.67
```

### Step 2: Update System

```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y git curl wget
```

### Step 3: Install Docker

```bash
# Download Docker installation script
curl -fsSL https://get.docker.com -o get-docker.sh

# Install Docker
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker ubuntu

# Verify installation
docker --version
```

Log out and log back in for group changes to take effect:
```bash
exit
# Reconnect
ssh -i search-relevancy-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

### Step 4: Install Docker Compose

```bash
# Download latest Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Make executable
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker-compose --version
```

### Step 5: Clone Project

```bash
# Clone repository
git clone https://github.com/your-username/search-relevancy.git
cd search-relevancy

# Create necessary directories
mkdir -p data/raw data/processed models
```

### Step 6: Prepare Data

Option A: Upload from local machine
```bash
# On your local machine
scp -i search-relevancy-key.pem /path/to/news_articles.csv \
    ubuntu@YOUR_EC2_PUBLIC_IP:/home/ubuntu/search-relevancy/data/raw/
```

Option B: Generate sample data
```bash
# On EC2 instance
python generate_sample_data.py
```

### Step 7: Build & Deploy Application

```bash
# Navigate to project directory
cd ~/search-relevancy

# Build Docker image
docker-compose build

# Start application
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f search-api
```

### Step 8: Verify Deployment

```bash
# Test health endpoint
curl http://localhost:5000/health

# Test info endpoint
curl http://localhost:5000/info

# Or from local machine
curl http://YOUR_EC2_PUBLIC_IP:5000/health
```

---

## Production Configuration

### 1. Configure Environment

Create `.env` file:

```bash
# On EC2 instance
cat > .env << EOF
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=0
AWS_REGION=us-east-1
SBERT_MODEL=all-MiniLM-L6-v2
DEFAULT_NUM_RESULTS=10
MAX_NUM_RESULTS=50
EOF
```

### 2. Setup Reverse Proxy (Nginx)

```bash
# Install Nginx
sudo apt-get install -y nginx

# Create Nginx configuration
sudo tee /etc/nginx/sites-available/search-relevancy > /dev/null <<EOF
upstream search_api {
    server localhost:5000;
}

server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://search_api;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/search-relevancy /etc/nginx/sites-enabled/

# Remove default site
sudo rm /etc/nginx/sites-enabled/default

# Test configuration
sudo nginx -t

# Start Nginx
sudo systemctl start nginx
sudo systemctl enable nginx
```

### 3. Setup SSL/TLS (Optional but Recommended)

Using Let's Encrypt and Certbot:

```bash
# Install Certbot
sudo apt-get install -y certbot python3-certbot-nginx

# Get certificate (requires domain name)
sudo certbot certonly --nginx -d your-domain.com

# Update Nginx configuration with SSL
sudo certbot --nginx -d your-domain.com
```

### 4. Setup Auto-restart

Docker Compose handles restart policies. Verify in docker-compose.yml:

```yaml
services:
  search-api:
    restart: unless-stopped
```

### 5. Setup Logs

```bash
# Create log directory
mkdir -p logs

# Update docker-compose.yml to mount logs
# volumes:
#   - ./logs:/var/log/app

# View logs
docker-compose logs search-api
tail -f logs/app.log
```

---

## Monitoring and Scaling

### 1. CloudWatch Monitoring

```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i -E ./amazon-cloudwatch-agent.deb

# Configure agent
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -s
```

### 2. Application Metrics

```bash
# Install Prometheus (optional)
docker run -d \
    --name prometheus \
    -p 9090:9090 \
    -v ./prometheus.yml:/etc/prometheus/prometheus.yml \
    prom/prometheus
```

### 3. Disk Space Management

```bash
# Check disk usage
df -h

# Clean up Docker
docker system prune -a

# Clean up old logs
sudo journalctl --vacuum=time=30d
```

### 4. Scaling to Multiple Instances

For horizontal scaling:

1. **Create AMI from current instance**
   - Select instance → Image and templates → Create image

2. **Create Auto Scaling Group**
   - Use created AMI
   - Set min/max instances
   - Configure load balancer

3. **Setup Application Load Balancer**
   - Point to auto scaling group
   - Health check: `/health` on port 5000

---

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs search-api

# Restart container
docker-compose restart search-api

# Rebuild if needed
docker-compose down
docker-compose build
docker-compose up -d
```

### Memory issues

```bash
# Check system memory
free -h

# Check container memory usage
docker stats

# Increase swap (temporary solution)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Port conflicts

```bash
# Find process using port 5000
sudo lsof -i :5000

# Kill process
sudo kill -9 <PID>
```

### API not responding

```bash
# Test local connectivity
curl http://localhost:5000/health

# Check security group
# AWS Console → EC2 → Security Groups → search-relevancy-sg

# Check Nginx
sudo systemctl status nginx
sudo nginx -t

# Check firewall
sudo ufw status
sudo ufw allow 5000/tcp
```

### High latency

```bash
# Check instance type
# If using t3.small/medium, consider upgrading to t3.large

# Optimize ANNOY parameters
# Reduce num_trees in config.py
# Increase Docker memory allocation

# Use CloudWatch to identify bottlenecks
```

### Data synchronization issues

```bash
# Verify ANNOY index integrity
python -c "from annoy import AnnoyIndex; idx = AnnoyIndex(384, 'angular'); idx.load('models/articles_index.annoy'); print(f'Index loaded: {idx.get_n_items()} items')"

# Rebuild if corrupted
python src/build_annoy_index.py
docker-compose restart search-api
```

---

## Security Best Practices

1. **Use IAM roles instead of access keys**
   - Attach EC2 instance role with necessary permissions

2. **Regular security updates**
   ```bash
   sudo apt-get update && sudo apt-get upgrade
   ```

3. **Restrict security group access**
   - Only allow necessary ports
   - Restrict source IPs where possible

4. **Enable CloudTrail logging**
   - AWS Console → CloudTrail → Create trail

5. **Use VPC with private subnets**
   - Place database/models in private subnets
   - Use NAT gateway for outbound access

6. **Backup strategies**
   ```bash
   # Create EBS snapshots
   # AWS Console → EC2 → Snapshots → Create snapshot
   ```

7. **Rotate credentials regularly**
   - Change SSH keys
   - Rotate AWS access keys

---

## Cost Optimization

1. **Use Reserved Instances** for long-term deployments
2. **Enable auto-scaling** to match demand
3. **Use S3 for model storage** instead of EBS
4. **Consider Spot Instances** for non-critical environments
5. **Monitor CloudWatch metrics** to right-size instances

---

## Maintenance Schedule

- **Daily**: Monitor logs and metrics
- **Weekly**: Check disk usage and clean up
- **Monthly**: Security updates and backups
- **Quarterly**: Review performance and optimize

---

## Support Resources

- AWS Documentation: https://docs.aws.amazon.com/ec2/
- Docker Documentation: https://docs.docker.com/
- Flask Documentation: https://flask.palletsprojects.com/
- SBERT Documentation: https://www.sbert.net/

For issues specific to this project, refer to [README.md](README.md)

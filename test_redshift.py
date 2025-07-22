import socket
from sqlalchemy import create_engine
from urllib.parse import urlparse, quote_plus
import os

# --- Configuration ---
# It's good practice to separate credentials
user = "demo_user"
password = "kopkYc-jocnyp-8vywwe"
host = "demo-workgroup.910606336720.eu-north-1.redshift-serverless.amazonaws.com"
port = 5439
database = "dev"

# URL-encode the password to handle any special characters
encoded_password = quote_plus(password)
conn_string = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{database}"

# --- Step-by-Step Network Diagnostics ---
print("--- Running Network Connection Test ---")
print(f"Target Host: {host}")
print(f"Target Port: {port}")
print("-" * 35)

# Step 1: DNS Resolution
print("Step 1: Resolving DNS...")
try:
    ip_address = socket.gethostbyname(host)
    print(f"✅ DNS resolution successful. Host '{host}' resolved to IP: {ip_address}")
except socket.gaierror as e:
    print(f"❌ FATAL: DNS resolution failed for host '{host}'.")
    print(f"   Error: {e}")
    print("\n--- TROUBLESHOOTING ---")
    print("This means your computer cannot find the server address. The issue is not a firewall.")
    print("1. Double-check the Redshift endpoint URL for any typos.")
    print(f"2. Try to `ping {host}` from your terminal. If it fails, there might be a local DNS issue or a problem with the Redshift endpoint itself.")
    exit() # Stop further tests if DNS fails

# Step 2: Socket Connection (Network Reachability)
print("\nStep 2: Attempting direct socket connection...")
try:
    with socket.create_connection((host, port), timeout=15) as s:
        print(f"✅ Socket connection successful! A network path to {host}:{port} is open.")
except socket.timeout:
    print(f"❌ FATAL: Socket connection timed out after 15 seconds.")
    print("\n--- TROUBLESHOOTING ---")
    print("This confirms a network block. Your request is being blocked by AWS networking rules.")
    print("Check these in your AWS Console, in this order of likelihood:")
    print("  1. ➡️  **VPC Route Tables & Subnets**: This is the most likely cause.")
    print("     - Your Redshift workgroup is associated with one or more subnets.")
    print("     - **EVERY SINGLE ONE** of these subnets must be 'public'.")
    print("     - A subnet is public if its associated Route Table has a route: `Destination: 0.0.0.0/0` -> `Target: igw-xxxxxxxx` (an Internet Gateway).")
    print("     - If even one associated subnet is private (routes to a NAT Gateway or has no internet route), the endpoint may be placed there and become unreachable.")
    print("  2. **Security Group IP**: Is your IP address correct in the Security Group inbound rule? Your IP can change.")
    print("  3. **Network ACL (NACL)**: The subnet's NACL must have Inbound AND Outbound rules allowing traffic from/to your IP.")
    print("Also check your local environment:")
    print("  4. **Local/Corporate Firewall**: Ensure your local machine or office network isn't blocking outbound port 5439.")
    exit()
except Exception as e:
    print(f"❌ FATAL: Socket connection failed with an unexpected error.")
    print(f"   Error: {e}")
    exit()

# Step 3: SQLAlchemy Connection (Database Handshake)
print("\nStep 3: Attempting database connection with SQLAlchemy...")
try:
    engine = create_engine(conn_string, connect_args={'connect_timeout': 20})
    with engine.connect() as conn:
        result = conn.execute("SELECT 1;")
        print("✅ SQLAlchemy connection successful! Credentials and database name are correct.")
        print(f"   Result from 'SELECT 1;': {result.scalar()}")
except Exception as e:
    print(f"❌ SQLAlchemy connection failed.")
    print(f"   Error: {e}")
    print("\n--- TROUBLESHOOTING ---")
    print("Since the network path is open (Step 2 succeeded), this error is likely due to:")
    print("  1. Incorrect Username or Password in your connection string.")
    print(f"  2. Incorrect Database Name ('{database}' in this case).")
    print("  3. An SSL/TLS configuration issue (less common with default settings).")
    print("  4. A driver-specific problem (`psycopg2`). Ensure it's installed correctly.")
    exit()

print("\n--- ✅ All connection tests passed successfully! ---")
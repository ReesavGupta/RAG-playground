#!/usr/bin/env python3
"""
Setup script for the Customer Support Ticketing System
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Setting up Customer Support Ticketing System Backend")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("pyproject.toml"):
        print("❌ Please run this script from the backend directory!")
        sys.exit(1)
    
    # Install dependencies
    if not run_command("uv sync", "Installing dependencies"):
        print("❌ Failed to install dependencies. Make sure you have UV installed.")
        sys.exit(1)
    
    # Check if .env exists
    if not os.path.exists(".env"):
        print("⚠️  .env file not found. Please create one with your configuration.")
        print("   You can copy the example from the README.md")
        return
    
    # Initialize database
    print("\n🗄️  Setting up database...")
    if not run_command("python init_db.py", "Initializing database"):
        print("❌ Failed to initialize database. Please check your PostgreSQL connection.")
        print("   Make sure PostgreSQL is running and your .env file has correct credentials.")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Make sure PostgreSQL is running")
    print("   2. Update your .env file with correct database credentials")
    print("   3. Run: python run.py")
    print("   4. Visit: http://localhost:8000/docs")

if __name__ == "__main__":
    main() 
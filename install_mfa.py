"""
Quick script to install Montreal Forced Aligner (MFA)
"""

import subprocess
import sys

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}...")
    print(f"   Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            check=True,
            text=True
        )
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"   Error code: {e.returncode}")
        return False

def check_mfa():
    """Check if MFA is already installed"""
    try:
        result = subprocess.run(
            ["mfa", "version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"✅ MFA is already installed: {result.stdout.strip()}")
            return True
    except:
        pass
    return False

def main():
    print("="*70)
    print("📦 MONTREAL FORCED ALIGNER (MFA) INSTALLATION")
    print("="*70)
    
    # Check if already installed
    if check_mfa():
        print("\n✅ MFA is already installed. Nothing to do!")
        return True
    
    print("\n❌ MFA is not installed")
    print("\n📥 Installing MFA via conda...")
    print("   This may take a few minutes...\n")
    
    # Install MFA
    if not run_command(
        ["conda", "install", "-c", "conda-forge", "montreal-forced-aligner", "-y"],
        "Installing Montreal Forced Aligner"
    ):
        print("\n❌ Installation failed!")
        print("\n💡 Please try manually:")
        print("   conda install -c conda-forge montreal-forced-aligner")
        return False
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    if check_mfa():
        print("\n" + "="*70)
        print("✅ MFA INSTALLATION COMPLETE!")
        print("="*70)
        print("\n📥 Next: Download the acoustic model")
        print("   mfa model download acoustic english_mfa")
        print("\n🚀 Then run the pipeline:")
        print("   python run_complete_pipeline.py")
        return True
    else:
        print("\n❌ Installation verification failed")
        print("   Please restart your terminal and try again")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

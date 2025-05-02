{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Python
    (python311.withPackages (ps: with ps; [
      pip
      virtualenv
      pytest
      pytest-cov
      mypy
      black
      flake8
      isort
      # Documentation
      sphinx
      sphinx-rtd-theme
    ]))
    
    # Developer tools
    git
    gnumake
  ];

  shellHook = ''
    # Create a Python virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
      echo "Creating virtual environment..."
      virtualenv venv
    fi
    
    # Activate the virtual environment
    source venv/bin/activate
    
    # Install the package in development mode if not already installed
    if ! pip list | grep -q "^pylean "; then
      echo "Installing pylean in development mode..."
      pip install -e .
    fi
    
    # Print welcome message
    echo "Pylean development environment initialized!"
    echo "Run 'deactivate' to exit the virtual environment."
  '';
}

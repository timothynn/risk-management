{
  description = "Risk Management System with VaR, Monte Carlo, and Stress Testing";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          numpy
          scipy
          pandas
          matplotlib
          seaborn
          numba
          joblib
          pytest
          black
          pylint
        ]);

      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          name = "risk-management-system";
          src = ./.;
          
          buildInputs = [ pythonEnv ];
          
          installPhase = ''
            mkdir -p $out/bin
            mkdir -p $out/lib
            
            cp -r src/* $out/lib/
            
            cat > $out/bin/risk-system <<EOF
            #!${pkgs.bash}/bin/bash
            export PYTHONPATH=$out/lib:\$PYTHONPATH
            ${pythonEnv}/bin/python $out/lib/main.py "\$@"
            EOF
            
            chmod +x $out/bin/risk-system
          '';
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.git
          ];

          shellHook = ''
            echo "Risk Management System Development Environment"
            echo "Python: $(python --version)"
            echo ""
            echo "Available commands:"
            echo "  python src/main.py           - Run the risk management system"
            echo "  pytest tests/                - Run tests"
            echo "  black src/                   - Format code"
            echo ""
          '';
        };

        apps.default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/risk-system";
        };
      }
    );
}

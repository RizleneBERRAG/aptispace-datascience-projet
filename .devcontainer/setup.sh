#!/usr/bin/env bash
set -e

echo "Mise à jour des paquets..."
sudo apt-get update

echo "Installation des outils système..."
sudo apt-get install -y curl wget ca-certificates gnupg

echo "Installation de Task..."
if ! command -v task >/dev/null 2>&1; then
  curl -1sLf 'https://dl.cloudsmith.io/public/task/task/setup.deb.sh' | sudo -E bash
  sudo apt-get update
  sudo apt-get install -y task
else
  echo "Task est déjà installé."
fi

echo "Installation de Quarto..."
if ! command -v quarto >/dev/null 2>&1; then
  cd /tmp
  wget -O quarto.deb https://github.com/quarto-dev/quarto-cli/releases/download/v1.9.37/quarto-1.9.37-linux-amd64.deb
  sudo apt-get install -y ./quarto.deb
  cd -
else
  echo "Quarto est déjà installé."
fi

echo "Installation des dépendances Python..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Environnement prêt."

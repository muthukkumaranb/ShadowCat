#!/bin/bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io docker-compose-v2 jq curl dos2unix
cd /d/sih2026/fabric-experiment/fabric-samples/test-network
export PATH=$PWD/../bin:$PATH
dos2unix *.sh
dos2unix scripts/*.sh
dos2unix organizations/fabric-ca/*.sh
dos2unix addOrg3/*.sh
./network.sh up createChannel -c shadowcat-notary-channel -ca

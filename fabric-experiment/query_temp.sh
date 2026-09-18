#!/bin/bash
cd /d/sih2026/fabric-experiment/fabric-samples/test-network
export PATH=$PWD/../bin:$PATH
export FABRIC_CFG_PATH=$PWD/../config/

# Set up Org1 Admin environment
export CORE_PEER_TLS_ENABLED=true
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${PWD}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${PWD}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=peer0.org1.example.com:7051

peer chaincode query -C shadowcat-notary-channel -n shadowcat_notary -c '{"function": "QueryAlert", "Args": ["874893f65af4302f"]}'

#!/bin/bash
cd /d/sih2026/fabric-experiment/fabric-samples/test-network
export PATH=$PWD/../bin:$PATH
./network.sh deployCC -ccn shadowcat_notary -ccp ../../chaincode/shadowcat_notary -ccl go -c shadowcat-notary-channel

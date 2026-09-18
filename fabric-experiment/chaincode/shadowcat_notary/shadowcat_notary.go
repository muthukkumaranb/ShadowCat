package main

import (
	"encoding/json"
	"fmt"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// SmartContract provides functions for managing a notary
type SmartContract struct {
	contractapi.Contract
}

// ModelProvenance describes basic details of what makes up a model
type ModelProvenance struct {
	ModelID        string `json:"model_id"`
	CheckpointPath string `json:"checkpoint_path"`
	SchemaPath     string `json:"schema_path"`
}

// AlertRecord describes an alert triggered during inference
type AlertRecord struct {
	AlertHash string `json:"alert_hash"`
	Severity  string `json:"severity"`
	Timestamp string `json:"timestamp"`
}

// InitLedger adds a base set of data to the ledger
func (s *SmartContract) InitLedger(ctx contractapi.TransactionContextInterface) error {
	return nil
}

// RecordModelProvenance adds a model provenance record to the ledger
func (s *SmartContract) RecordModelProvenance(ctx contractapi.TransactionContextInterface, modelID string, checkpointPath string, schemaPath string) error {
	provenance := ModelProvenance{
		ModelID:        modelID,
		CheckpointPath: checkpointPath,
		SchemaPath:     schemaPath,
	}

	provenanceJSON, err := json.Marshal(provenance)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState(modelID, provenanceJSON)
}

// NotarizeAlert adds an alert record to the ledger
func (s *SmartContract) NotarizeAlert(ctx contractapi.TransactionContextInterface, alertHash string, severity string, timestamp string) error {
	alert := AlertRecord{
		AlertHash: alertHash,
		Severity:  severity,
		Timestamp: timestamp,
	}

	alertJSON, err := json.Marshal(alert)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState(alertHash, alertJSON)
}

// QueryAlert returns the alert stored in the world state with given id
func (s *SmartContract) QueryAlert(ctx contractapi.TransactionContextInterface, alertHash string) (*AlertRecord, error) {
	alertJSON, err := ctx.GetStub().GetState(alertHash)
	if err != nil {
		return nil, fmt.Errorf("failed to read from world state: %v", err)
	}
	if alertJSON == nil {
		return nil, fmt.Errorf("the alert %s does not exist", alertHash)
	}

	var alert AlertRecord
	err = json.Unmarshal(alertJSON, &alert)
	if err != nil {
		return nil, err
	}

	return &alert, nil
}

func main() {
	chaincode, err := contractapi.NewChaincode(&SmartContract{})
	if err != nil {
		fmt.Printf("Error creating shadowcat_notary chaincode: %s", err.Error())
		return
	}

	if err := chaincode.Start(); err != nil {
		fmt.Printf("Error starting shadowcat_notary chaincode: %s", err.Error())
	}
}

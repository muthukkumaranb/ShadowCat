package main

import (
	"encoding/json"
	"fmt"
	"strings"

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

// PredictionLineage binds all four pipeline stages together in an atomic ledger record
type PredictionLineage struct {
	LineageID      string `json:"lineage_id"`
	RawDataHash    string `json:"raw_data_hash"`
	FeatureHash    string `json:"feature_hash"`
	ModelID        string `json:"model_id"`
	PredictionHash string `json:"prediction_hash"`
	Severity       string `json:"severity"`
	Timestamp      string `json:"timestamp"`
	TargetNode     string `json:"target_node,omitempty"`
}

// IncidentResponseRecord describes an autonomous on-chain incident containment action
type IncidentResponseRecord struct {
	IncidentID        string `json:"incident_id"`
	LineageID         string `json:"lineage_id"`
	RecommendedAction string `json:"recommended_action"`
	TriggeredAt       string `json:"triggered_at"`
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

// Helper to record lineage and handle auto-trigger inside Go chaincode
func (s *SmartContract) recordLineageInternal(
	ctx contractapi.TransactionContextInterface,
	lineageID string,
	rawDataHash string,
	featureHash string,
	modelID string,
	predictionHash string,
	severity string,
	timestamp string,
	targetNode string,
) error {
	if strings.TrimSpace(targetNode) == "" {
		targetNode = "172.31.69.21"
	}

	lineage := PredictionLineage{
		LineageID:      lineageID,
		RawDataHash:    rawDataHash,
		FeatureHash:    featureHash,
		ModelID:        modelID,
		PredictionHash: predictionHash,
		Severity:       severity,
		Timestamp:      timestamp,
		TargetNode:     targetNode,
	}

	lineageJSON, err := json.Marshal(lineage)
	if err != nil {
		return fmt.Errorf("failed to marshal lineage: %v", err)
	}

	// 1. Write atomic lineage record
	err = ctx.GetStub().PutState(lineageID, lineageJSON)
	if err != nil {
		return fmt.Errorf("failed to put lineage state: %v", err)
	}

	// 2. CHAINCODE-NATIVE AUTO-TRIGGER
	// Decision MUST be made inside chaincode (Go): if severity is HIGH or CRITICAL,
	// chaincode autonomously creates an IncidentResponseRecord in the same transaction.
	normSev := strings.ToUpper(strings.TrimSpace(severity))
	if normSev == "HIGH" || normSev == "CRITICAL" {
		incidentID := fmt.Sprintf("incident_%s", lineageID)
		action := fmt.Sprintf("ISOLATE_HOST:%s", targetNode)
		incident := IncidentResponseRecord{
			IncidentID:        incidentID,
			LineageID:         lineageID,
			RecommendedAction: action,
			TriggeredAt:       timestamp,
		}

		incidentJSON, err := json.Marshal(incident)
		if err != nil {
			return fmt.Errorf("failed to marshal incident response: %v", err)
		}

		err = ctx.GetStub().PutState(incidentID, incidentJSON)
		if err != nil {
			return fmt.Errorf("failed to put incident response state: %v", err)
		}

		// Emit on-chain event as well
		_ = ctx.GetStub().SetEvent("IncidentAutoTriggered", incidentJSON)
	}

	return nil
}

// RecordPredictionLineage records 4-stage pipeline lineage and auto-triggers containment for HIGH/CRITICAL (7 arguments)
func (s *SmartContract) RecordPredictionLineage(
	ctx contractapi.TransactionContextInterface,
	lineageID string,
	rawDataHash string,
	featureHash string,
	modelID string,
	predictionHash string,
	severity string,
	timestamp string,
) error {
	targetNode := "172.31.69.21"
	if strings.Contains(lineageID, "::") {
		parts := strings.Split(lineageID, "::")
		if len(parts) > 1 && parts[1] != "" {
			targetNode = parts[1]
		}
	}
	return s.recordLineageInternal(ctx, lineageID, rawDataHash, featureHash, modelID, predictionHash, severity, timestamp, targetNode)
}

// RecordPredictionLineageWithTarget records 4-stage pipeline lineage with explicit target node (8 arguments)
func (s *SmartContract) RecordPredictionLineageWithTarget(
	ctx contractapi.TransactionContextInterface,
	lineageID string,
	rawDataHash string,
	featureHash string,
	modelID string,
	predictionHash string,
	severity string,
	timestamp string,
	targetNode string,
) error {
	return s.recordLineageInternal(ctx, lineageID, rawDataHash, featureHash, modelID, predictionHash, severity, timestamp, targetNode)
}

// QueryPredictionLineage returns the prediction lineage record for a given lineageID
func (s *SmartContract) QueryPredictionLineage(ctx contractapi.TransactionContextInterface, lineageID string) (*PredictionLineage, error) {
	lineageJSON, err := ctx.GetStub().GetState(lineageID)
	if err != nil {
		return nil, fmt.Errorf("failed to read from world state: %v", err)
	}
	if lineageJSON == nil {
		return nil, fmt.Errorf("the lineage %s does not exist", lineageID)
	}

	var lineage PredictionLineage
	err = json.Unmarshal(lineageJSON, &lineage)
	if err != nil {
		return nil, err
	}

	return &lineage, nil
}

// QueryIncidentResponse returns the incident response record for a given incidentID
func (s *SmartContract) QueryIncidentResponse(ctx contractapi.TransactionContextInterface, incidentID string) (*IncidentResponseRecord, error) {
	incidentJSON, err := ctx.GetStub().GetState(incidentID)
	if err != nil {
		return nil, fmt.Errorf("failed to read from world state: %v", err)
	}
	if incidentJSON == nil {
		return nil, fmt.Errorf("the incident %s does not exist", incidentID)
	}

	var incident IncidentResponseRecord
	err = json.Unmarshal(incidentJSON, &incident)
	if err != nil {
		return nil, err
	}

	return &incident, nil
}

// QueryAllIncidents queries all IncidentResponseRecord objects from the ledger
func (s *SmartContract) QueryAllIncidents(ctx contractapi.TransactionContextInterface) ([]*IncidentResponseRecord, error) {
	resultsIterator, err := ctx.GetStub().GetStateByRange("incident_", "incident_\uffff")
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()

	var incidents []*IncidentResponseRecord
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}

		var inc IncidentResponseRecord
		err = json.Unmarshal(queryResponse.Value, &inc)
		if err != nil {
			continue
		}
		incidents = append(incidents, &inc)
	}

	return incidents, nil
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

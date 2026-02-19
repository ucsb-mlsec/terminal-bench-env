#!/usr/bin/env python3

import time
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Participant:
    """Mock participant representing a database node"""
    
    def __init__(self, name: str, prepare_response: bool = True, response_delay: float = 0.0):
        self.name = name
        self.prepare_response = prepare_response
        self.response_delay = response_delay
        self.state = "IDLE"
        self.prepared_transactions = set()
        
    def prepare(self, transaction_id: str, operation: Dict) -> Optional[bool]:
        """Prepare phase - vote YES/NO on transaction"""
        logger.info(f"Participant {self.name} received PREPARE for transaction {transaction_id}")
        
        if self.response_delay > 0:
            time.sleep(self.response_delay)
            
        if self.response_delay >= 2.0:  # Simulate timeout
            logger.warning(f"Participant {self.name} timed out for transaction {transaction_id}")
            return None
            
        if self.prepare_response:
            self.prepared_transactions.add(transaction_id)
            self.state = "PREPARED"
            logger.info(f"Participant {self.name} voted YES for transaction {transaction_id}")
            return True
        else:
            logger.info(f"Participant {self.name} voted NO for transaction {transaction_id}")
            return False
    
    def commit(self, transaction_id: str) -> bool:
        """Commit the transaction"""
        logger.info(f"Participant {self.name} COMMITTING transaction {transaction_id}")
        self.state = "COMMITTED"
        if transaction_id in self.prepared_transactions:
            self.prepared_transactions.remove(transaction_id)
        return True
    
    def abort(self, transaction_id: str) -> bool:
        """Abort the transaction"""
        logger.info(f"Participant {self.name} ABORTING transaction {transaction_id}")
        self.state = "ABORTED"
        if transaction_id in self.prepared_transactions:
            self.prepared_transactions.remove(transaction_id)
        return True


class TransactionCoordinator:
    """Two-phase commit coordinator with intentional bugs"""
    
    def __init__(self, participants: List[Participant], timeout: float = 1.5):
        self.participants = participants
        self.timeout = timeout
        self.transaction_log = {}
        
    def execute_transaction(self, transaction_id: str, operation: Dict) -> bool:
        """Execute a distributed transaction using 2PC"""
        logger.info(f"Starting transaction {transaction_id} with operation: {operation}")
        
        # Phase 1: Prepare
        prepare_result = self.prepare_phase(transaction_id, operation)
        
        # Phase 2: Commit or Abort
        if prepare_result:
            return self.commit_phase(transaction_id)
        else:
            return self.abort_all(transaction_id)
    
    def prepare_phase(self, transaction_id: str, operation: Dict) -> bool:
        """Phase 1: Send PREPARE to all participants and collect votes"""
        logger.info(f"PREPARE phase for transaction {transaction_id}")
        
        votes = []
        responses_received = 0
        
        # BUG 1: Doesn't wait for ALL participants - breaks early
        for participant in self.participants:
            start_time = time.time()
            vote = participant.prepare(transaction_id, operation)
            elapsed = time.time() - start_time
            
            # BUG 2: Timeout handling is broken - doesn't check properly
            if elapsed < self.timeout:
                votes.append(vote)
                responses_received += 1
                
                # BUG 3: Breaking early without checking all participants!
                if vote == False:
                    break  # This breaks the loop prematurely!
            
            # BUG 4: Missing else clause for timeout case - doesn't track timeouts
        
        # BUG 5: Not verifying we got responses from ALL participants
        all_yes = all(v == True for v in votes)
        
        logger.info(f"Prepare phase results: {len(votes)} votes collected, all_yes={all_yes}")
        return all_yes
    
    def commit_phase(self, transaction_id: str) -> bool:
        """Phase 2: Send COMMIT to all participants"""
        logger.info(f"COMMIT phase for transaction {transaction_id}")
        
        # BUG 6: Sends commit without double-checking prepare results
        for participant in self.participants:
            participant.commit(transaction_id)
        
        self.transaction_log[transaction_id] = "COMMITTED"
        logger.info(f"Transaction {transaction_id} COMMITTED")
        return True
    
    def abort_all(self, transaction_id: str) -> bool:
        """Rollback all participants"""
        logger.info(f"ABORT phase for transaction {transaction_id}")
        
        # BUG 7: Only aborts participants that were iterated in prepare_phase
        # Missing participants that didn't get prepare might be in inconsistent state
        for participant in self.participants:
            participant.abort(transaction_id)
        
        self.transaction_log[transaction_id] = "ABORTED"
        logger.info(f"Transaction {transaction_id} ABORTED")
        return False


if __name__ == "__main__":
    # Quick test
    participants = [
        Participant("DB1", prepare_response=True),
        Participant("DB2", prepare_response=True),
        Participant("DB3", prepare_response=True)
    ]
    
    coordinator = TransactionCoordinator(participants)
    operation = {'action': 'reserve', 'item': 'ABC', 'quantity': 5}
    result = coordinator.execute_transaction("txn_001", operation)
    
    print(f"\nTransaction result: {result}")
    print(f"Participant states: {[(p.name, p.state) for p in participants]}")